"""Run local release checks and atomically bundle hash-pinned evidence."""

from __future__ import annotations

import argparse
import http.client
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import venv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import release_gate

EVIDENCE_SCHEMA = "rlaif.local_app.release_evidence"
EVIDENCE_VERSION = 1
CHECK_COMMANDS: dict[str, tuple[str, ...]] = {
    "local": ("-m", "unittest", "discover", "-s", "local_app/tests", "-t", "."),
    "rebuild": ("-m", "unittest", "discover", "-s", "rlaif/tests", "-t", "."),
    "browser": ("-m", "unittest", "local_app.tests.test_browser_regressions"),
    "catalog": ("-m", "unittest", "local_app.tests.test_regression_catalog", "local_app.tests.test_regression_catalog_scenarios"),
    "clean_package": ("-m", "unittest", "local_app.tests.test_packaging"),
    "ops": ("-m", "unittest", "local_app.tests.test_operations", "local_app.tests.test_operations_contract"),
}
COMMAND_LABELS = {
    "local": "project-python -m unittest discover -s local_app/tests -t .",
    "rebuild": "project-python -m unittest discover -s rlaif/tests -t . (working directory: rebuild)",
    "browser": "project-python -m unittest local_app.tests.test_browser_regressions",
    "catalog": "project-python -m unittest local_app.tests.test_regression_catalog local_app.tests.test_regression_catalog_scenarios",
    "clean_package": "build wheel and sdist; fresh temporary venv; offline install; isolated import, localhost health, and one-scenario evaluation",
    "ops": "project-python -m unittest local_app.tests.test_operations local_app.tests.test_operations_contract",
}


@dataclass(frozen=True)
class CleanPackageResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    subchecks: Mapping[str, str]
    failure_code: str | None = None


class CleanPackageStepFailure(RuntimeError):
    def __init__(self, step: str, returncode: int, stdout: bytes = b"", stderr: bytes = b"",
                 failure_code: str = "distribution_build_failed"):
        super().__init__(step)
        self.step = step
        self.returncode = returncode or 1
        self.stdout = stdout
        self.stderr = stderr
        self.failure_code = failure_code


def _build_distributions(source: Path, distribution_dir: Path, executable: str) -> tuple[Path, Path, bytes, bytes]:
    """Build sdist and wheel using the available setuptools backend, offline."""
    script = (
        "import re, sys, importlib.metadata as m; from pathlib import Path; "
        "v=m.version('setuptools'); match=re.match(r'(\\d+)\\.(\\d+)', v); "
        "assert match and tuple(map(int, match.groups())) >= (68, 0), 'setuptools>=68 required'; "
        "import setuptools.build_meta as b; "
        "d=Path(sys.argv[1]); d.mkdir(parents=True, exist_ok=True); "
        "b.build_sdist(str(d)); b.build_wheel(str(d))"
    )
    result = subprocess.run((executable, "-c", script, str(distribution_dir)), cwd=source,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False)
    if result.returncode:
        build_error = result.stderr or b""
        missing_backend = (b"No module named 'setuptools'" in build_error
                           or b"No package metadata was found for setuptools" in build_error)
        raise CleanPackageStepFailure("build_wheel_and_sdist", result.returncode,
                                      result.stdout or b"", result.stderr or b"",
                                      "build_backend_unavailable" if missing_backend else "distribution_build_failed")
    wheels = sorted(distribution_dir.glob("*.whl"))
    sdists = sorted(distribution_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise CleanPackageStepFailure("build_wheel_and_sdist", 1,
                                      stderr=b"build did not produce exactly one wheel and one sdist",
                                      failure_code="distribution_artifacts_missing")
    return wheels[0], sdists[0], result.stdout or b"", result.stderr or b""


def _offline_install(python: Path, wheel: Path, cwd: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        (str(python), "-I", "-m", "pip", "install", "--disable-pip-version-check",
         "--no-index", "--no-deps", "--no-build-isolation", str(wheel)),
        cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )


def _verify_installed_imports(python: Path, cwd: Path) -> subprocess.CompletedProcess[bytes]:
    script = (
        "from pathlib import Path; import local_app, local_app.api, local_app.server, "
        "local_app.evaluation; from local_app.knowledge import DEFAULT_CORPUS; "
        "p=Path(local_app.__file__).resolve(); v=Path(__import__('sys').prefix).resolve(); "
        "assert p.is_relative_to(v), (p, v); assert DEFAULT_CORPUS.is_file(); "
        "print('installed distribution imports verified')"
    )
    return subprocess.run((str(python), "-I", "-c", script), cwd=cwd,
                          stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False)


def _verify_local_health(python: Path, cwd: Path) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    server = subprocess.Popen(
        (str(python), "-I", "-u", "-m", "local_app.server", "--host", "127.0.0.1",
         "--port", str(port)), cwd=cwd, stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    health_ok = False
    try:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and server.poll() is None:
            try:
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=0.5)
                connection.request("GET", "/health")
                response = connection.getresponse()
                body = response.read()
                health_ok = response.status == 200 and json.loads(body) == {
                    "status": "ok", "simulator": True,
                }
                connection.close()
                if health_ok:
                    break
            except (OSError, http.client.HTTPException, json.JSONDecodeError):
                time.sleep(0.05)
    finally:
        server.terminate()
        try:
            server.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=3)
    return health_ok


def _run_minimal_evaluation(python: Path, cwd: Path) -> subprocess.CompletedProcess[bytes]:
    script = (
        "from local_app.evaluation import load_scenarios, run_scenario; "
        "s=load_scenarios(count=1)[0]; r=run_scenario(s); "
        "assert r['task_success'] and r['completed']; print('one-scenario evaluation passed')"
    )
    return subprocess.run((str(python), "-I", "-c", script), cwd=cwd,
                          stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False)


def _run_clean_package(root: Path, source_python: str) -> CleanPackageResult:
    """Build and validate the package without source imports or network access."""
    output = bytearray()
    errors = bytearray()
    subchecks: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="rlaif-clean-package-") as temp_name:
        scratch = Path(temp_name)
        source = scratch / "source"
        source.mkdir()
        shutil.copytree(root / "local_app", source / "local_app",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for filename in ("pyproject.toml", "LOCAL_CONVERSATION_SYSTEM_SPEC.md"):
            shutil.copy2(root / filename, source / filename)
        distribution_dir = scratch / "dist"
        try:
            wheel, sdist, build_out, build_err = _build_distributions(
                source, distribution_dir, source_python)
        except CleanPackageStepFailure as exc:
            subchecks[exc.step] = "failed"
            return CleanPackageResult(exc.returncode, exc.stdout, exc.stderr, subchecks, exc.failure_code)
        except Exception as exc:
            subchecks["build_wheel_and_sdist"] = "failed"
            errors.extend(type(exc).__name__.encode("ascii", errors="replace"))
            return CleanPackageResult(1, bytes(output), bytes(errors), subchecks, "distribution_build_exception")
        subchecks["build_wheel_and_sdist"] = "passed"
        output.extend(build_out)
        errors.extend(build_err)

        environment = scratch / "venv"
        try:
            venv.EnvBuilder(with_pip=True, clear=True).create(environment)
        except Exception as exc:
            subchecks["fresh_venv"] = "failed"
            errors.extend(type(exc).__name__.encode("ascii", errors="replace"))
            return CleanPackageResult(1, bytes(output), bytes(errors), subchecks, "fresh_venv_creation_failed")
        subchecks["fresh_venv"] = "passed"
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        verification_cwd = scratch / "verify"
        verification_cwd.mkdir()

        install = _offline_install(python, wheel, verification_cwd)
        output.extend(install.stdout or b"")
        errors.extend(install.stderr or b"")
        subchecks["offline_install"] = "passed" if install.returncode == 0 else "failed"
        if install.returncode:
            return CleanPackageResult(install.returncode, bytes(output), bytes(errors), subchecks,
                                      "offline_install_failed")

        imported = _verify_installed_imports(python, verification_cwd)
        output.extend(imported.stdout or b"")
        errors.extend(imported.stderr or b"")
        subchecks["isolated_imports"] = "passed" if imported.returncode == 0 else "failed"
        if imported.returncode:
            return CleanPackageResult(imported.returncode, bytes(output), bytes(errors), subchecks,
                                      "isolated_import_failed")

        try:
            health_ok = _verify_local_health(python, verification_cwd)
        except Exception:
            health_ok = False
        subchecks["localhost_health"] = "passed" if health_ok else "failed"
        if not health_ok:
            return CleanPackageResult(1, bytes(output), bytes(errors), subchecks, "localhost_health_failed")

        evaluated = _run_minimal_evaluation(python, verification_cwd)
        output.extend(evaluated.stdout or b"")
        errors.extend(evaluated.stderr or b"")
        subchecks["minimal_evaluation"] = "passed" if evaluated.returncode == 0 else "failed"
        return CleanPackageResult(evaluated.returncode, bytes(output), bytes(errors), subchecks,
                                  None if evaluated.returncode == 0 else "minimal_evaluation_failed")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(payload, encoding="utf-8")


def _safe_relative(path: Path, root: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the project root") from exc


def _execute(command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(command, cwd=cwd, stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def record_evidence(
    *,
    project_root: str | Path,
    integrated_matrix: str | Path,
    output_parent: str | Path,
    checks: Sequence[str] | None = None,
    runner: Callable[[Sequence[str], Path], subprocess.CompletedProcess[bytes]] = _execute,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    timer: Callable[[], float] = time.monotonic,
) -> Path:
    """Run selected check commands, then atomically publish a complete bundle.

    `checks` exists for targeted local diagnosis. Only a full six-check run can
    be used by the release gate because omitted evidence will fail its contract.
    Process output is hashed in memory and never persisted.
    """
    root = Path(project_root).resolve()
    matrix = Path(integrated_matrix).resolve()
    parent_candidate = Path(output_parent)
    parent = (root / parent_candidate).resolve() if not parent_candidate.is_absolute() else parent_candidate.resolve()
    _safe_relative(parent, root, "output directory")
    selected = tuple(checks or CHECK_COMMANDS)
    unknown = sorted(set(selected) - CHECK_COMMANDS.keys())
    if unknown:
        raise ValueError("unknown release evidence checks: " + ", ".join(unknown))
    if len(set(selected)) != len(selected):
        raise ValueError("release evidence checks must be unique")
    matrix_ref = _safe_relative(matrix, root, "integrated matrix artifact")
    if not matrix.is_file():
        raise ValueError("integrated matrix artifact does not exist")
    # Reject incomplete, controller-only, synthetic, or tampered matrix inputs
    # before running checks. This validates supplied bytes; it never runs models.
    release_gate._verify_matrix(matrix)
    matrix_digest = _sha256_file(matrix)
    python = root / ".venv" / "bin" / "python"
    executable = str(python) if python.is_file() else sys.executable
    results: dict[str, dict[str, Any]] = {}
    for name in selected:
        cwd = root / "rebuild" if name == "rebuild" else root
        started_at = clock().astimezone(timezone.utc).isoformat()
        started = timer()
        try:
            result = (_run_clean_package(root, executable) if name == "clean_package"
                      else runner((executable, *CHECK_COMMANDS[name]), cwd))
            out = result.stdout or b""
            err = result.stderr or b""
            code = int(result.returncode)
        except Exception as exc:
            # Exception messages can contain environment paths or credentials;
            # retain only the exception class in the durable check record.
            out, err, code = b"", b"", 127
            error_kind = type(exc).__name__
        else:
            error_kind = None
        elapsed_ms = max(0, round((timer() - started) * 1000))
        results[name] = {
            "schema": "rlaif.local_app.release_check",
            "version": 1,
            "check": name,
            "command": COMMAND_LABELS[name],
            "working_directory": "rebuild" if name == "rebuild" else ".",
            "status": "passed" if code == 0 and error_kind is None else "failed",
            "exit_code": code,
            "started_at": started_at,
            "duration_ms": elapsed_ms,
            "stdout_sha256": _sha256_bytes(out),
            "stderr_sha256": _sha256_bytes(err),
            "stdout_bytes": len(out),
            "stderr_bytes": len(err),
        }
        if error_kind:
            results[name]["execution_error"] = error_kind
        subchecks = getattr(result, "subchecks", None) if error_kind is None else None
        if subchecks is not None:
            results[name]["subchecks"] = dict(subchecks)
        failure_code = getattr(result, "failure_code", None) if error_kind is None else None
        if failure_code:
            results[name]["failure_code"] = failure_code

    # Each run gets an immutable versioned directory. Publish with one atomic
    # rename, so consumers never see a partially written evidence bundle.
    parent.mkdir(parents=True, exist_ok=True)
    stamp = clock().astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = parent / f"evidence-v{EVIDENCE_VERSION}-{stamp}"
    temp_dir = Path(tempfile.mkdtemp(prefix=".release-evidence-", dir=parent))
    try:
        check_dir = temp_dir / "checks"
        check_dir.mkdir()
        check_refs = {}
        for name, record in results.items():
            check_path = check_dir / f"{name}.json"
            _atomic_json(check_path, record)
            check_refs[name] = {
                "path": _safe_relative(destination / "checks" / check_path.name, root, "output path"),
                "sha256": _sha256_file(check_path),
            }
        bundle = {
            "schema": EVIDENCE_SCHEMA,
            "version": EVIDENCE_VERSION,
            "created_at": clock().astimezone(timezone.utc).isoformat(),
            "checks": check_refs,
            "integrated_matrix": {"path": matrix_ref, "sha256": matrix_digest},
        }
        bundle_path = temp_dir / "evidence-v1.json"
        _atomic_json(bundle_path, bundle)
        if destination.exists():
            raise FileExistsError("release evidence version already exists")
        os.replace(temp_dir, destination)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return destination / "evidence-v1.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path,
                        help="existing integrated matrix/release JSON artifact inside the project")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="project root")
    parser.add_argument("--output-parent", type=Path, default=Path("artifacts/release"))
    parser.add_argument("--checks", nargs="+", choices=tuple(CHECK_COMMANDS),
                        help="run selected local checks (release gate requires all six)")
    args = parser.parse_args(argv)
    try:
        bundle = record_evidence(project_root=args.root, integrated_matrix=args.matrix,
                                 output_parent=args.output_parent, checks=args.checks)
    except (OSError, ValueError) as exc:
        # Some filesystem/parser exceptions include absolute paths. Keep the
        # CLI diagnostic useful without copying those paths into logs.
        detail = (str(exc) if isinstance(exc, ValueError)
                  and "must be inside the project root" in str(exc) else type(exc).__name__)
        print(f"release evidence error: {detail}", file=sys.stderr)
        return 2
    print(bundle.relative_to(Path(args.root).resolve()).as_posix())
    records = json.loads(bundle.read_text(encoding="utf-8"))["checks"]
    return 0 if len(records) == len(CHECK_COMMANDS) and all(
        json.loads((bundle.parent / "checks" / f"{name}.json").read_text(encoding="utf-8"))["status"] == "passed"
        for name in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
