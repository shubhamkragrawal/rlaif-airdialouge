import unittest

from local_app.observability import (
    EvaluationArtifact, RunRecord, TurnRecord, canonical_json, record_json,
    record_sha256, validate_redacted,
)


class ObservabilityTests(unittest.TestCase):
    def test_turn_is_immutable_and_redacts_passenger_data(self):
        source = {"passenger": "Amy Gonzalez", "text": "Book for Amy Gonzalez"}
        turn = TurnRecord(
            run_id="run-1", scenario_id="scenario-3", controller_version="abc123",
            tool_inputs=(source,), tool_outputs=({"message": "Amy Gonzalez booked"},),
            transaction_evidence_ids=("hold-1", "booking-1"), validation_result="valid",
            input_tokens=12, output_tokens=8, latency_ms=23.5,
            peak_memory_bytes=1024, final_outcome="confirmed",
            adapter={"name": "demo", "weights_sha256": "a" * 64},
            hardware={"device": "cpu"}, generation_settings={"temperature": 0},
        )
        source["text"] = "mutated"
        serialized = record_json(turn)
        self.assertNotIn("Amy Gonzalez", serialized)
        self.assertIn("[REDACTED]", serialized)
        self.assertEqual(turn.tool_inputs[0]["text"], "Book for [REDACTED]")
        with self.assertRaises(TypeError):
            turn.adapter["name"] = "changed"
        validate_redacted(turn)

    def test_canonical_json_and_hash_are_stable_across_mapping_order(self):
        first = RunRecord("r", "controller", adapter={"name": "a", "sha256": "b"})
        second = RunRecord("r", "controller", adapter={"sha256": "b", "name": "a"})
        self.assertEqual(record_json(first), record_json(second))
        self.assertEqual(record_sha256(first), record_sha256(second))
        self.assertEqual(canonical_json({"é": 1, "a": 2}), '{"a":2,"é":1}')

    def test_evaluation_artifact_is_separate_and_versioned(self):
        artifact = EvaluationArtifact("r", "s", "v1", final_outcome="success")
        self.assertNotEqual(artifact.record_type, RunRecord("r", "v1").record_type)
        self.assertEqual(artifact.schema_version, 1)
        with self.assertRaises(ValueError):
            RunRecord("r", "v1", schema_version=2)

    def test_redaction_validator_rejects_unredacted_identity_field(self):
        with self.assertRaises(ValueError):
            validate_redacted({"passenger": "Amy Gonzalez"})

    def test_redaction_marker_cannot_hide_pii_in_another_field(self):
        with self.assertRaises(ValueError):
            validate_redacted({"passenger": "[REDACTED]", "notes": "Amy Gonzalez"})

    def test_redaction_validation_checks_nested_values_and_keys(self):
        with self.assertRaises(ValueError):
            validate_redacted({"metadata": [{"notes": "My name is Amy Gonzalez"}]})
        with self.assertRaises(ValueError):
            validate_redacted({"metadata": {"Amy Gonzalez": "[REDACTED]"}})

    def test_redaction_validation_detects_unicode_name_candidates(self):
        for value in (
            "Passenger José García",
            "My name is Jose\u0301 Garci\u0301a",
            "Jean‑Luc Picard",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_redacted({"nested": {"text": value}})

    def test_record_construction_redacts_unanchored_names_with_other_pii(self):
        record = TurnRecord(
            run_id="r", scenario_id="s", controller_version="v1",
            tool_inputs=({"nested": {"notes": "Amy Gonzalez, email amy@example.com, card 4111 1111 1111 1111"}},),
        )
        serialized = record_json(record)
        for pii in ("Amy Gonzalez", "amy@example.com", "4111 1111 1111 1111"):
            self.assertNotIn(pii, serialized)
        self.assertIn("[REDACTED]", serialized)

    def test_record_construction_redacts_multiple_unicode_names_in_free_text(self):
        record = TurnRecord(
            run_id="r", scenario_id="s", controller_version="v1",
            tool_outputs=({"notes": "Amy Gonzalez and José O’Neil are traveling"},),
        )
        self.assertNotIn("Amy Gonzalez", record_json(record))
        self.assertNotIn("José O’Neil", record_json(record))

    def test_record_json_redacts_uppercase_cjk_and_decomposed_names_with_other_pii(self):
        record = TurnRecord(
            run_id="r", scenario_id="s", controller_version="v1",
            tool_outputs=({"nested": [{"notes":
                "Passengers AMY GONZALEZ, 李 雷, and Jose\u0301 Garci\u0301a are traveling; "
                "email amy@example.com; card 4111 1111 1111 1111"}]},),
        )
        serialized = record_json(record)
        for fragment in ("AMY", "GONZALEZ", "李", "雷", "Jose\u0301", "Garci\u0301a",
                         "amy@example.com", "4111 1111 1111 1111"):
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, serialized)
        self.assertIn("traveling", serialized)

    def test_validator_rejects_uppercase_cjk_and_decomposed_name_survivors(self):
        for name in ("Passengers AMY GONZALEZ are traveling",
                     "Passengers 李 雷 are traveling",
                     "Passengers Jose\u0301 Garci\u0301a are traveling"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_redacted({"nested": [{"notes": name}]})

    def test_record_json_redacts_names_in_other_record_fields(self):
        record = TurnRecord(
            run_id="r", scenario_id="s", controller_version="v1",
            validation_result=("Passenger AMY GONZALEZ, 李 雷, and "
                               "Jose\u0301 Garci\u0301a are traveling; "
                               "amy@example.com; 4111 1111 1111 1111"),
        )
        serialized = record_json(record)
        for fragment in ("AMY", "GONZALEZ", "李", "雷", "Jose\u0301", "Garci\u0301a",
                         "amy@example.com", "4111 1111 1111 1111"):
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, serialized)
        self.assertIn("[REDACTED]", serialized)

    def test_validator_rejects_case_and_script_name_survivors(self):
        for text in ("Passenger AMY GONZALEZ", "Passenger 李 雷",
                     "Passenger Jose\u0301 Garci\u0301a"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate_redacted({"nested": {"text": text}})

    def test_redaction_validation_rejects_unanchored_names_in_keys_and_values(self):
        for payload in (
            {"nested": {"notes": "Amy Gonzalez, email [REDACTED], card [REDACTED]"}},
            {"nested": {"José O’Neil": "[REDACTED]"}},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_redacted(payload)

    def test_redaction_validation_allows_ordinary_text_and_redacted_names(self):
        validate_redacted({
            "notes": "Search JFK to LAX; the flight departs at noon.",
            "passenger": "[REDACTED]",
            "nested": ["[REDACTED]", "validation completed successfully"],
        })

    def test_rejects_invalid_measurements(self):
        with self.assertRaises(ValueError):
            TurnRecord("r", "s", "v1", input_tokens=-1)
        with self.assertRaises(ValueError):
            TurnRecord("r", "s", "v1", latency_ms=float("nan"))


if __name__ == "__main__":
    unittest.main()
