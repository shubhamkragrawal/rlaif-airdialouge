"""CPU checks for immutable closeout preparation and dataset provenance."""
import tempfile
from datetime import date
from pathlib import Path
import unittest

from cloud.identity import validate_identity
from rlaif.shared.data import verify_dataset_manifest


class CloseoutTests(unittest.TestCase):
    def test_historical_image_and_suffix_are_rejected(self):
        today = date.today().strftime("%Y%m%d")
        with self.assertRaisesRegex(ValueError, "historical (date|image version)"):
            validate_identity(today + "-v13", today, "closeout-v1", minimum_version=14)
        with self.assertRaisesRegex(ValueError, "historical run suffix"):
            validate_identity(today + "-v14", today, "adaptive-v4")

    def test_tag_and_date_must_match(self):
        today = date.today().strftime("%Y%m%d")
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_identity(today + "-v14", "20991231", "closeout-v1", minimum_version=14)

    def test_checked_in_dataset_manifest_matches_files(self):
        manifest = verify_dataset_manifest(require_sft_examples=True)
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(set(manifest["files"]), {"source", "splits", "sft_examples"})


if __name__ == "__main__":
    unittest.main()
