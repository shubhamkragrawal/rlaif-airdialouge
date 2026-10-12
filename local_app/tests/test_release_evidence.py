import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_app.release_evidence import (
    CHECK_COMMANDS,
    CleanPackageResult,
    CleanPackageStepFailure,
    _build_distributions,
    _run_clean_package,
    record_evidence,
)


class ReleaseEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "rebuild").mkdir()
        self.matrix = self.root / "artifacts/matrix/integrated.json"
        self.matrix.parent.mkdir(parents=True)
        self.matrix.write_text('{"schema":"synthetic-test-input"}', encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_records_actual_results_as_hash_pinned_sanitized_bundle_atomically(self):
        calls = []

        def validate_matrix(path):
            self.assertEqual(path, self.matrix.resolve())
            return {"cohort": {"scenario_count": 50}}

        def runner(command, cwd):
            calls.append((tuple(command), cwd))
            return subprocess.CompletedProcess(command, 0, b"tests passed at /private/machine/path\n", b"")

        with patch("local_app.release_evidence.release_gate._verify_matrix", side_effect=validate_matrix), \
                patch("local_app.release_evidence._run_clean_package", return_value=CleanPackageResult(
                    0, b"package checks passed", b"", {
                        "build_wheel_and_sdist": "passed", "fresh_venv": "passed",
                        "offline_install": "passed", "isolated_imports": "passed",
                        "localhost_health": "passed", "minimal_evaluation": "passed",
                    })) as package_check:
            bundle_path = record_evidence(project_root=self.root, integrated_matrix=self.matrix,
                                          output_parent="artifacts/release", runner=runner)

        self.assertEqual(len(calls), 5)
        package_check.assert_called_once()
        self.assertEqual(bundle_path.parent.parent, (self.root / "artifacts/release").resolve())
        bundle = json.loads(bundle_path.read_text())
        self.assertEqual(set(bundle["checks"]), set(CHECK_COMMANDS))
        self.assertEqual(bundle["integrated_matrix"]["path"], "artifacts/matrix/integrated.json")
        self.assertEqual(bundle["integrated_matrix"]["sha256"], hashlib.sha256(self.matrix.read_bytes()).hexdigest())
        for name, ref in bundle["checks"].items():
            path = self.root.resolve() / ref["path"]
            record = json.loads(path.read_text())
            self.assertEqual(record["status"], "passed")
            self.assertEqual(record["exit_code"], 0)
            self.assertEqual(ref["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(record["check"], name)
            self.assertNotIn("/private/", path.read_text())
            self.assertNotIn(str(self.root), path.read_text())
        clean_record = json.loads((self.root.resolve() / bundle["checks"]["clean_package"]["path"]).read_text())
        self.assertEqual(clean_record["subchecks"]["build_wheel_and_sdist"], "passed")
        self.assertEqual(clean_record["subchecks"]["offline_install"], "passed")
        for command, _cwd in calls:
            self.assertFalse(any("gcloud" in part or "curl" in part or "http" in part for part in command))
        self.assertEqual(len(list(bundle_path.parent.parent.glob("evidence-v1-*"))), 1)

    def test_failure_is_recorded_and_subset_is_marked_non_release_bundle(self):
        runs = 0

        def runner(command, cwd):
            nonlocal runs
            runs += 1
            code = 2 if runs == 2 else 0
            return subprocess.CompletedProcess(command, code, b"", b"failure detail with /private/path")

        with patch("local_app.release_evidence.release_gate._verify_matrix", return_value={}):
            bundle_path = record_evidence(project_root=self.root, integrated_matrix=self.matrix,
                output_parent="artifacts/release", checks=("local", "rebuild"), runner=runner)
        bundle = json.loads(bundle_path.read_text())
        self.assertEqual(set(bundle["checks"]), {"local", "rebuild"})
        rebuild_path = self.root / bundle["checks"]["rebuild"]["path"]
        rebuild = json.loads(rebuild_path.read_text())
        self.assertEqual((rebuild["status"], rebuild["exit_code"]), ("failed", 2))
        self.assertNotIn("failure detail", rebuild_path.read_text())

    def test_rejects_artifact_outside_project_and_invalid_check_before_running(self):
        called = []
        with self.assertRaisesRegex(ValueError, "inside the project root"):
            record_evidence(project_root=self.root, integrated_matrix=Path(self.temp.name).parent / "elsewhere.json",
                            output_parent="artifacts/release", runner=lambda *_: called.append(True))
        with self.assertRaisesRegex(ValueError, "unknown release evidence checks"):
            record_evidence(project_root=self.root, integrated_matrix=self.matrix,
                            output_parent="artifacts/release", checks=("cloud",), runner=lambda *_: called.append(True))
        self.assertEqual(called, [])

    def test_clean_package_runs_build_fresh_venv_offline_install_import_health_and_eval(self):
        (self.root / "local_app").mkdir()
        (self.root / "local_app/__init__.py").write_text("", encoding="utf-8")
        (self.root / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
        (self.root / "LOCAL_CONVERSATION_SYSTEM_SPEC.md").write_text("spec", encoding="utf-8")
        from local_app import release_evidence

        def build(source, distribution_dir, executable):
            distribution_dir.mkdir(parents=True)
            wheel = distribution_dir / "fixture.whl"
            sdist = distribution_dir / "fixture.tar.gz"
            wheel.write_bytes(b"wheel")
            sdist.write_bytes(b"sdist")
            return wheel, sdist, b"built", b""

        def create_env(path):
            python = path / "bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("", encoding="utf-8")

        success = subprocess.CompletedProcess([], 0, b"ok", b"")
        with patch.object(release_evidence, "_build_distributions", side_effect=build) as build_mock, \
                patch.object(release_evidence.venv.EnvBuilder, "create", side_effect=create_env), \
                patch.object(release_evidence, "_offline_install", return_value=success) as install_mock, \
                patch.object(release_evidence, "_verify_installed_imports", return_value=success), \
                patch.object(release_evidence, "_verify_local_health", return_value=True), \
                patch.object(release_evidence, "_run_minimal_evaluation", return_value=success):
            result = _run_clean_package(self.root, "python-under-test")

        build_mock.assert_called_once()
        self.assertEqual(build_mock.call_args.args[2], "python-under-test")
        install_mock.assert_called_once()
        install_args = install_mock.call_args.args
        self.assertEqual(install_args[0].name, "python")
        self.assertTrue(install_args[1].name.endswith(".whl"))
        self.assertEqual(result.returncode, 0)
        self.assertEqual(set(result.subchecks.values()), {"passed"})

    def test_distribution_build_invokes_both_backend_builds_and_preserves_failure_evidence(self):
        source = self.root / "staged-source"
        source.mkdir()
        output = self.root / "dist"

        def successful_build(command, cwd, **kwargs):
            self.assertEqual(cwd, source)
            self.assertIn("build_sdist", command[2])
            self.assertIn("build_wheel", command[2])
            self.assertNotIn("pip", command)
            output.mkdir()
            (output / "sample.whl").write_bytes(b"wheel")
            (output / "sample.tar.gz").write_bytes(b"sdist")
            return subprocess.CompletedProcess(command, 0, b"build output", b"")

        with patch("local_app.release_evidence.subprocess.run", side_effect=successful_build):
            wheel, sdist, stdout, stderr = _build_distributions(source, output, "builder-python")
        self.assertEqual((wheel.name, sdist.name), ("sample.whl", "sample.tar.gz"))
        self.assertEqual((stdout, stderr), (b"build output", b""))

        failed = subprocess.CompletedProcess(["build"], 1, b"out", b"backend failed")
        with patch("local_app.release_evidence.subprocess.run", return_value=failed):
            with self.assertRaises(CleanPackageStepFailure) as caught:
                _build_distributions(source, self.root / "failed-dist", "builder-python")
        self.assertEqual(caught.exception.returncode, 1)
        self.assertEqual(caught.exception.stderr, b"backend failed")

        missing = subprocess.CompletedProcess(
            ["build"], 1, b"", b"importlib.metadata.PackageNotFoundError: No package metadata was found for setuptools"
        )
        with patch("local_app.release_evidence.subprocess.run", return_value=missing):
            with self.assertRaises(CleanPackageStepFailure) as caught:
                _build_distributions(source, self.root / "no-backend", "builder-python")
        self.assertEqual(caught.exception.failure_code, "build_backend_unavailable")


if __name__ == "__main__":
    unittest.main()
