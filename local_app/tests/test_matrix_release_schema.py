"""Schema checks for the emitted integrated release and its companion manifests."""

import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from local_app.controller import ADAPTERS
from local_app.evaluation import DEFAULT_CORPUS
from local_app.matrix import run_matrix, run_release


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "artifacts/matrix/schema-v1.json"


def _json_type_matches(value, expected):
    checks = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "string": lambda v: isinstance(v, str),
        "boolean": lambda v: isinstance(v, bool),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "null": lambda v: v is None,
    }
    return any(checks[kind](value) for kind in expected) if isinstance(expected, list) else checks[expected](value)


def _schema_errors(value, subschema, root, location="$", *, collect_properties=False):
    """Small stdlib-only validator for the JSON Schema keywords used by this schema."""
    errors = []
    if "$ref" in subschema:
        name = subschema["$ref"].rsplit("/", 1)[-1]
        return _schema_errors(value, root["$defs"][name], root, location,
                              collect_properties=collect_properties)
    if "type" in subschema and not _json_type_matches(value, subschema["type"]):
        return [f"{location}: expected {subschema['type']}"]
    if "const" in subschema and value != subschema["const"]:
        errors.append(f"{location}: const mismatch")
    if "enum" in subschema and value not in subschema["enum"]:
        errors.append(f"{location}: enum mismatch")
    if "required" in subschema and isinstance(value, dict):
        errors.extend(f"{location}: missing {key}" for key in subschema["required"] if key not in value)
    if isinstance(value, str):
        if "minLength" in subschema and len(value) < subschema["minLength"]:
            errors.append(f"{location}: string too short")
        if "pattern" in subschema and re.search(subschema["pattern"], value) is None:
            errors.append(f"{location}: pattern mismatch")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in subschema and value < subschema["minimum"]:
            errors.append(f"{location}: below minimum")
        if "maximum" in subschema and value > subschema["maximum"]:
            errors.append(f"{location}: above maximum")
    if isinstance(value, list):
        if len(value) < subschema.get("minItems", 0) or len(value) > subschema.get("maxItems", float("inf")):
            errors.append(f"{location}: array length out of bounds")
        if subschema.get("uniqueItems") and len({json.dumps(v, sort_keys=True) for v in value}) != len(value):
            errors.append(f"{location}: duplicate array items")
        if "items" in subschema:
            for index, item in enumerate(value):
                errors.extend(_schema_errors(item, subschema["items"], root, f"{location}[{index}]"))
    if isinstance(value, dict):
        properties = subschema.get("properties", {})
        for key, child_schema in properties.items():
            if key in value:
                errors.extend(_schema_errors(value[key], child_schema, root, f"{location}.{key}"))
        additional = subschema.get("additionalProperties", True)
        for key in value.keys() - properties.keys():
            if additional is False:
                errors.append(f"{location}: unexpected property {key}")
            elif isinstance(additional, dict):
                errors.extend(_schema_errors(value[key], additional, root, f"{location}.{key}"))
        if subschema.get("unevaluatedProperties") is False:
            evaluated = set(properties)
            for child in subschema.get("allOf", []):
                evaluated.update(_evaluated_properties(child, root))
            for key in value.keys() - evaluated:
                errors.append(f"{location}: unevaluated property {key}")
    for child in subschema.get("allOf", []):
        errors.extend(_schema_errors(value, child, root, location))
    if "oneOf" in subschema:
        matches = sum(not _schema_errors(value, option, root, location)
                      for option in subschema["oneOf"])
        if matches != 1:
            errors.append(f"{location}: oneOf matched {matches} branches")
    if "if" in subschema and not _schema_errors(value, subschema["if"], root, location):
        if "then" in subschema:
            errors.extend(_schema_errors(value, subschema["then"], root, location))
    return errors


def _evaluated_properties(subschema, root):
    if "$ref" in subschema:
        return _evaluated_properties(root["$defs"][subschema["$ref"].rsplit("/", 1)[-1]], root)
    names = set(subschema.get("properties", {}))
    for child in subschema.get("allOf", []):
        names.update(_evaluated_properties(child, root))
    return names


class StubModel:
    last_usage = {"input_tokens": 4, "output_tokens": 3}

    def generate(self, _messages, _flights):
        return "I can help with that request."


class StubRuntimeManager:
    def __init__(self, paths):
        self.config = SimpleNamespace(
            base_model="stub/Qwen", revision="revision-test",
            max_new_tokens=160, temperature=0.0,
            adapter_paths={name: str(path) for name, path in paths.items()},
        )
        self.current = None

    def get(self, adapter, **_kwargs):
        self.current = adapter
        return StubModel()

    def status(self, adapter):
        return {"adapter": adapter,
                "lifecycle_status": "ready" if self.current == adapter else "available",
                "mode": "local_model", "available": True,
                "provenance_status": "complete" if self.current == adapter else "partial"}

    def provenance(self, adapter):
        return {"adapter": adapter,
                "lifecycle_status": "ready" if self.current == adapter else "available",
                "mode": "local_model", "available": True,
                "provenance_status": "complete" if self.current == adapter else "partial",
                "adapter_sha256": hashlib.sha256(adapter.encode()).hexdigest(),
                "base_model": self.config.base_model,
                "base_model_revision": self.config.revision,
                "generation_settings": {"max_new_tokens": 160, "temperature": 0.0},
                "device": "cpu", "precision": "float32", "peak_memory_bytes": 4096}

    def release(self):
        self.current = None


class MatrixReleaseSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    def assert_schema_valid(self, value):
        errors = _schema_errors(value, self.schema, self.schema)
        self.assertEqual(errors, [], "\n".join(errors))

    def make_release(self, directory):
        paths = {}
        for adapter in ADAPTERS:
            path = Path(directory) / f"adapter-{adapter}"
            path.mkdir()
            (path / "adapter_config.json").write_text("{}", encoding="utf-8")
            paths[adapter] = path
        manager = StubRuntimeManager(paths)
        output_dir = Path(directory) / "release"
        result = run_release(runtime_manager=manager, corpus_path=DEFAULT_CORPUS,
                             output_dir=output_dir, arm_runner=run_matrix)
        return result, output_dir

    def test_generated_release_and_emitted_checkpoint_and_final_manifest_validate(self):
        with tempfile.TemporaryDirectory() as directory:
            release, output_dir = self.make_release(directory)
            self.assertEqual(tuple(release["adapters"]), ADAPTERS)
            self.assert_schema_valid(release)
            manifest = json.loads((output_dir / "manifest-v1.json").read_text(encoding="utf-8"))
            self.assert_schema_valid(manifest)
            self.assertEqual(manifest["sha256"], release["sha256"])
            self.assertEqual(manifest["identity"], release["identity"])
            checkpoint = json.loads((output_dir / "checkpoints/sft.json").read_text(encoding="utf-8"))
            self.assert_schema_valid(checkpoint)
            self.assertEqual(checkpoint["adapter"], "sft")
            self.assertEqual(checkpoint["arm"]["layer"], "integrated")
            self.assertEqual(checkpoint["arm"]["status"], "complete")
            self.assertEqual(checkpoint["arm"]["fingerprint"],
                             release["adapters"]["sft"]["fingerprint"])

    def test_malformed_release_envelope_and_identity_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            release, _output_dir = self.make_release(directory)
        malformed = json.loads(json.dumps(release))
        malformed.pop("identity")
        self.assertTrue(_schema_errors(malformed, self.schema, self.schema))
        malformed = json.loads(json.dumps(release))
        malformed["identity"].pop("controller_sha256")
        self.assertTrue(_schema_errors(malformed, self.schema, self.schema))
        malformed = json.loads(json.dumps(release))
        malformed["unexpected"] = True
        self.assertTrue(_schema_errors(malformed, self.schema, self.schema))
        malformed = json.loads(json.dumps(release))
        malformed["sha256"] = "not-a-sha256"
        self.assertTrue(_schema_errors(malformed, self.schema, self.schema))


if __name__ == "__main__":
    unittest.main()
