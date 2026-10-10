import io
import json
from pathlib import Path
import tempfile
import unittest

from local_app.regression_cases import (
    BROWSER_CATALOG_PATH,
    CASES_BY_FAMILY,
    CASES_BY_ID,
    EXPECTED_BROWSER_CLAIMS,
    REGRESSION_CASES,
    REQUIRED_BEHAVIORS,
    REQUIRED_FAMILIES,
    RegressionCase,
    load_browser_catalog,
    validate_catalog,
)


class RegressionCatalogTests(unittest.TestCase):
    def test_catalog_covers_every_required_family_behavior_exactly_once(self):
        validate_catalog()
        self.assertEqual(set(CASES_BY_FAMILY), set(REQUIRED_FAMILIES))
        self.assertEqual(set(CASES_BY_ID), {case.case_id for case in REGRESSION_CASES})
        for family, required_behaviors in REQUIRED_BEHAVIORS.items():
            with self.subTest(family=family):
                self.assertEqual(
                    {case.behavior for case in CASES_BY_FAMILY[family]},
                    set(required_behaviors),
                )

    def test_composite_families_use_distinct_executable_targets(self):
        composite_families = {
            family for family, behaviors in REQUIRED_BEHAVIORS.items()
            if len(behaviors) > 1
        }
        for family in composite_families:
            with self.subTest(family=family):
                cases = CASES_BY_FAMILY[family]
                self.assertEqual(len(cases), len(REQUIRED_BEHAVIORS[family]))
                self.assertEqual(len({case.test_id for case in cases}), len(cases))

    def test_missing_family_validation_identifies_the_missing_family(self):
        with self.assertRaisesRegex(ValueError, "no matching flights"):
            validate_catalog(
                cases=(case for case in REGRESSION_CASES
                       if case.family != "no matching flights"),
            )

    def test_catalog_rejects_duplicate_stable_ids(self):
        duplicate = REGRESSION_CASES[0]
        with self.assertRaisesRegex(ValueError, duplicate.case_id):
            validate_catalog((*REGRESSION_CASES, duplicate))

    def test_catalog_rejects_missing_sub_behavior(self):
        with self.assertRaisesRegex(ValueError, "invented flight, airline, or price: invented_airline"):
            validate_catalog(
                case for case in REGRESSION_CASES
                if case.behavior != "invented_airline"
            )

    def test_catalog_rejects_reused_test_target(self):
        original = REGRESSION_CASES[-1]
        replacement = RegressionCase(
            original.case_id,
            original.family,
            original.behavior,
            REGRESSION_CASES[-2].test_id,
            original.browser_ids,
        )
        cases = (*REGRESSION_CASES[:-1], replacement)
        with self.assertRaisesRegex(ValueError, "duplicate regression test targets"):
            validate_catalog(cases)

    def test_browser_case_ids_match_task13_artifact_exactly(self):
        artifact = load_browser_catalog(BROWSER_CATALOG_PATH)
        artifact_ids = {case["case_id"] for case in artifact["cases"]}
        mapped_ids = {
            browser_id for case in REGRESSION_CASES for browser_id in case.browser_ids
        }
        self.assertEqual(mapped_ids, artifact_ids)
        self.assertEqual(artifact_ids, set(EXPECTED_BROWSER_CLAIMS))

    def test_tampered_browser_report_section_fails_validation(self):
        artifact = load_browser_catalog(BROWSER_CATALOG_PATH)
        artifact["cases"][0]["report_section"] = "99. Nonsense heading"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "browser_regressions_v1.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "browser BR001 report_section does not match allowed claim"
            ):
                validate_catalog(browser_catalog_path=path)

    def test_tampered_browser_coverage_fails_validation(self):
        artifact = load_browser_catalog(BROWSER_CATALOG_PATH)
        artifact["cases"][1]["coverage"] = "Nonsense coverage claim."
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "browser_regressions_v1.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "browser BR002 coverage does not match allowed claim"
            ):
                validate_catalog(browser_catalog_path=path)

    def test_every_catalog_target_is_executed(self):
        validate_catalog()
        suite = unittest.TestSuite(
            unittest.defaultTestLoader.loadTestsFromName(case.test_id)
            for case in REGRESSION_CASES
        )
        result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
        self.assertTrue(
            result.wasSuccessful(),
            "catalog targets failed: " + "; ".join(
                [failure[1] for failure in result.failures]
                + [error[1] for error in result.errors]
            ),
        )
        self.assertEqual(result.testsRun, len(REGRESSION_CASES))


if __name__ == "__main__":
    unittest.main()
