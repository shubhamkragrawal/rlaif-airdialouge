"""Machine-readable catalog of local-app regression families and test targets."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import unittest
from typing import Iterable


@dataclass(frozen=True)
class RegressionCase:
    case_id: str
    family: str
    behavior: str
    test_id: str
    browser_ids: tuple[str, ...] = ()


REQUIRED_FAMILIES = (
    "grounded search",
    "no matching flights",
    "invalid selected flight",
    "combined search and passenger request",
    "combined selection, passenger, and booking request",
    "confirmation without a booking result",
    "booking-result mismatch",
    "repeated confirmation request",
    "raw JSON and empty output",
    "repetitive generation",
    "invented flight, airline, or price",
    "contradiction of non-empty results",
    "prompt injection and rendered HTML payloads",
    "stale search selection after criteria change",
    "adapter failure and timeout",
    "turn and generation budget exhaustion",
)


REQUIRED_BEHAVIORS = {
    "grounded search": ("grounded_search",),
    "no matching flights": ("no_matching_flights",),
    "invalid selected flight": ("invalid_selected_flight",),
    "combined search and passenger request": ("combined_search_passenger",),
    "combined selection, passenger, and booking request": (
        "combined_selection_passenger_booking",
    ),
    "confirmation without a booking result": ("confirmation_without_booking",),
    "booking-result mismatch": ("booking_result_mismatch",),
    "repeated confirmation request": ("repeated_confirmation",),
    "raw JSON and empty output": ("raw_json", "empty_output"),
    "repetitive generation": ("repetitive_generation",),
    "invented flight, airline, or price": (
        "invented_flight",
        "invented_airline",
        "invented_price",
    ),
    "contradiction of non-empty results": ("nonempty_result_contradiction",),
    "prompt injection and rendered HTML payloads": (
        "prompt_injection",
        "rendered_html_payload",
    ),
    "stale search selection after criteria change": ("stale_search_selection",),
    "adapter failure and timeout": ("adapter_failure", "adapter_timeout"),
    "turn and generation budget exhaustion": (
        "turn_budget_exhaustion",
        "generation_budget_exhaustion",
    ),
}


# Stable IDs are append-only. Multiple records may share a family so distinct
# behaviors remain independently traceable and executable.
REGRESSION_CASES = (
    RegressionCase("RLAIF-GROUNDED-SEARCH-001", REQUIRED_FAMILIES[0], "grounded_search",
                   "local_app.tests.test_browser_regressions.BrowserRegressionTests.test_BR005_grounded_search_names_only_retrieved_flight_and_fare",
                   ("BR005",)),
    RegressionCase("RLAIF-NO-MATCHING-FLIGHTS-001", REQUIRED_FAMILIES[1], "no_matching_flights",
                   "local_app.tests.test_browser_regressions.BrowserRegressionTests.test_BR002_impossible_search_returns_friendly_empty_result",
                   ("BR002",)),
    RegressionCase("RLAIF-INVALID-FLIGHT-SELECTION-001", REQUIRED_FAMILIES[2], "invalid_selected_flight",
                   "local_app.tests.test_browser_regressions.BrowserRegressionTests.test_BR001_invalid_selection_explains_flight_is_not_an_option",
                   ("BR001",)),
    RegressionCase("RLAIF-COMBINED-SEARCH-PASSENGER-001", REQUIRED_FAMILIES[3], "combined_search_passenger",
                   "local_app.tests.test_batch2_integration.BatchTwoIntegrationTests.test_passenger_forms_unicode_combined_and_rejection"),
    RegressionCase("RLAIF-COMBINED-SELECTION-BOOKING-001", REQUIRED_FAMILIES[4], "combined_selection_passenger_booking",
                   "local_app.tests.test_regression_catalog_scenarios.CatalogControllerScenarioTests.test_single_message_selection_passenger_and_booking"),
    RegressionCase("RLAIF-UNSUPPORTED-CONFIRMATION-001", REQUIRED_FAMILIES[5], "confirmation_without_booking",
                   "local_app.tests.test_browser_regressions.BrowserRegressionTests.test_BR003_confirmation_requires_simulator_booking_evidence",
                   ("BR003",)),
    RegressionCase("RLAIF-BOOKING-RESULT-MISMATCH-001", REQUIRED_FAMILIES[6], "booking_result_mismatch",
                   "local_app.tests.test_batch1_integration.BatchOneIntegrationTests.test_evidence_mismatches_reject_transaction_claim"),
    RegressionCase("RLAIF-REPEATED-CONFIRMATION-001", REQUIRED_FAMILIES[7], "repeated_confirmation",
                   "local_app.tests.test_batch1_integration.BatchOneIntegrationTests.test_simulated_confirmation_is_authoritative_and_idempotent"),
    RegressionCase("RLAIF-RAW-JSON-EMPTY-OUTPUT-001", REQUIRED_FAMILIES[8], "raw_json",
                   "local_app.tests.test_batch1_integration.BatchOneIntegrationTests.test_raw_state_and_empty_array_model_outputs_are_rejected"),
    RegressionCase("RLAIF-RAW-JSON-EMPTY-OUTPUT-002", REQUIRED_FAMILIES[8], "empty_output",
                   "local_app.tests.test_output_validation.OutputValidationTests.test_malformed_empty_json_fenced_json_state_control_and_limit"),
    RegressionCase("RLAIF-REPETITIVE-GENERATION-001", REQUIRED_FAMILIES[9], "repetitive_generation",
                   "local_app.tests.test_browser_regressions.BrowserRegressionTests.test_BR004_repetitive_generation_falls_back_to_concise_text",
                   ("BR004",)),
    RegressionCase("RLAIF-INVENTED-FLIGHT-001", REQUIRED_FAMILIES[10], "invented_flight",
                   "local_app.tests.test_runtime_integration.RuntimeIntegrationTests.test_ungrounded_model_output_falls_back_to_deterministic_response"),
    RegressionCase("RLAIF-INVENTED-AIRLINE-001", REQUIRED_FAMILIES[10], "invented_airline",
                   "local_app.tests.test_batch2_integration.BatchTwoIntegrationTests.test_controller_rejects_airline_after_flight_id"),
    RegressionCase("RLAIF-INVENTED-PRICE-001", REQUIRED_FAMILIES[10], "invented_price",
                   "local_app.tests.test_runtime_integration.RuntimeIntegrationTests.test_model_price_must_match_retrieved_context"),
    RegressionCase("RLAIF-AVAILABILITY-CONTRADICTION-001", REQUIRED_FAMILIES[11], "nonempty_result_contradiction",
                   "local_app.tests.test_runtime_integration.RuntimeIntegrationTests.test_model_cannot_deny_non_empty_search_results"),
    RegressionCase("RLAIF-PROMPT-INJECTION-HTML-001", REQUIRED_FAMILIES[12], "prompt_injection",
                   "local_app.tests.test_browser_regressions.BrowserRegressionTests.test_BR006_prompt_injection_and_xss_cannot_book_or_execute_markup",
                   ("BR006",)),
    RegressionCase("RLAIF-PROMPT-INJECTION-HTML-002", REQUIRED_FAMILIES[12], "rendered_html_payload",
                   "local_app.tests.test_browser_regressions.BrowserRegressionTests.test_BR007_accepted_text_is_rendered_as_text_and_cleared_from_input",
                   ("BR007",)),
    RegressionCase("RLAIF-STALE-SEARCH-SELECTION-001", REQUIRED_FAMILIES[13], "stale_search_selection",
                   "local_app.tests.test_batch1_integration.BatchOneIntegrationTests.test_route_change_and_book_checks_latest_search"),
    RegressionCase("RLAIF-ADAPTER-FAILURE-001", REQUIRED_FAMILIES[14], "adapter_failure",
                   "local_app.tests.test_runtime_integration.RuntimeIntegrationTests.test_model_failure_falls_back_without_breaking_the_session"),
    RegressionCase("RLAIF-ADAPTER-TIMEOUT-001", REQUIRED_FAMILIES[14], "adapter_timeout",
                   "local_app.tests.test_batch2_integration.BatchTwoIntegrationTests.test_cooperative_adapter_cancels_and_controller_recovers"),
    RegressionCase("RLAIF-TURN-BUDGET-001", REQUIRED_FAMILIES[15], "turn_budget_exhaustion",
                   "local_app.tests.test_regression_catalog_scenarios.CatalogControllerScenarioTests.test_turn_budget_exhaustion"),
    RegressionCase("RLAIF-GENERATION-BUDGET-001", REQUIRED_FAMILIES[15], "generation_budget_exhaustion",
                   "local_app.tests.test_regression_catalog_scenarios.CatalogControllerScenarioTests.test_generation_budget_exhaustion"),
)


BROWSER_CATALOG_PATH = (
    Path(__file__).resolve().parents[1] / "test_artifacts" / "browser_regressions_v1.json"
)
ARTIFACT_REPORT_PATH = (
    Path(__file__).resolve().parents[1] / "test_artifacts" / "test_artifact_report.md"
)


EXPECTED_BROWSER_CLAIMS = {
    "BR001": {
        "report_section": "2. Invalid flight selection — safe but misleading",
        "coverage": "Reject flight 9999 with explicit invalid-option wording; booking remains absent.",
        "report_evidence": ("Flight 9999 was not booked", "selected flight number was invalid"),
    },
    "BR002": {
        "report_section": "4. Impossible search — failed presentation check",
        "coverage": "An impossible IAD–LAS search under $1 yields friendly no-results text and structured raw results [].",
        "report_evidence": ("$1 budget", "raw model response `[]`"),
    },
    "BR003": {
        "report_section": "5. Booking flow — failed grounding and repetition checks",
        "coverage": "A model confirmation without simulator transaction evidence is rejected.",
        "report_evidence": ("reservation was confirmed", "not backed by a simulator result"),
    },
    "BR004": {
        "report_section": "5. Booking flow — failed grounding and repetition checks",
        "coverage": "Repeated model output is rejected and deterministic concise text is used.",
        "report_evidence": ("repeated closing phrases", "generation limit"),
    },
    "BR005": {
        "report_section": "1. Grounded search — passed",
        "coverage": "Positive control: model response is accepted when flight and fare match retrieved local data.",
        "report_evidence": ("flight 1027", "Southwest, $100", "local knowledge base"),
    },
    "BR006": {
        "report_section": "3. Prompt-injection and XSS attempt — passed",
        "coverage": "Executes the shipped page script in a JavaScript engine with a behavioral DOM and submits injection/XSS markup; verifies no payload elements, event attributes, or execution marker appear, invalid selection is rejected, and the input clears on submit and after response.",
        "report_evidence": ("HTML payload", "zero injected images", "nonexistent flight was rejected"),
    },
    "BR007": {
        "report_section": "3. Prompt-injection and XSS attempt — passed",
        "coverage": "Executes the shipped page script and verifies submitted accepted text appears literally in the transcript with no parsed markup, and the input is empty both on submit and after the assistant response.",
        "report_evidence": ("displayed as inert text", "no JavaScript marker"),
    },
}


def load_browser_catalog(path: Path = BROWSER_CATALOG_PATH) -> dict:
    """Read Task 13's browser catalog as the authoritative set of browser IDs."""
    with path.open(encoding="utf-8") as source:
        artifact = json.load(source)
    if artifact.get("artifact_version") != 1 or not isinstance(artifact.get("cases"), list):
        raise ValueError(f"unsupported browser regression catalog: {path}")
    return artifact


def _walk_tests(suite: unittest.TestSuite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from _walk_tests(test)
        else:
            yield test


def _report_sections(path: Path) -> dict[str, str]:
    """Return level-three report sections keyed by their exact heading text."""
    text = path.read_text(encoding="utf-8")
    matches = tuple(re.finditer(r"^### (.+)$", text, flags=re.MULTILINE))
    sections = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group(1)] = text[match.end():end]
    return sections


def validate_catalog(
    cases: Iterable[RegressionCase] = REGRESSION_CASES,
    required_families: Iterable[str] = REQUIRED_FAMILIES,
    browser_catalog_path: Path = BROWSER_CATALOG_PATH,
    artifact_report_path: Path = ARTIFACT_REPORT_PATH,
) -> None:
    """Raise ``ValueError`` when coverage, ID, browser, or target integrity fails."""
    case_list = tuple(cases)
    required = tuple(required_families)
    present_families = {case.family for case in case_list}
    missing = [family for family in required if family not in present_families]
    if missing:
        raise ValueError("missing required regression families: " + ", ".join(missing))

    ids = [case.case_id for case in case_list]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        raise ValueError("duplicate regression case IDs: " + ", ".join(duplicates))
    invalid = [case_id for case_id in ids
               if not re.fullmatch(r"RLAIF-[A-Z0-9]+(?:-[A-Z0-9]+)*-\d{3}", case_id)]
    if invalid:
        raise ValueError("invalid stable regression case IDs: " + ", ".join(invalid))

    required_behaviors = {
        family: REQUIRED_BEHAVIORS[family]
        for family in required
        if family in REQUIRED_BEHAVIORS
    }
    missing_behaviors = []
    unexpected_behaviors = []
    for family, expected in required_behaviors.items():
        present = {case.behavior for case in case_list if case.family == family}
        missing_behaviors.extend(
            f"{family}: {behavior}" for behavior in expected if behavior not in present
        )
        unexpected_behaviors.extend(
            f"{family}: {behavior}" for behavior in sorted(present - set(expected))
        )
    if missing_behaviors:
        raise ValueError("missing required regression behaviors: " + ", ".join(missing_behaviors))
    if unexpected_behaviors:
        raise ValueError("unexpected regression behaviors: " + ", ".join(unexpected_behaviors))

    behavior_keys = [(case.family, case.behavior) for case in case_list]
    duplicate_behaviors = sorted({
        f"{family}: {behavior}"
        for family, behavior in behavior_keys
        if behavior_keys.count((family, behavior)) > 1
    })
    if duplicate_behaviors:
        raise ValueError(
            "duplicate regression behavior mappings: " + ", ".join(duplicate_behaviors)
        )

    test_ids = [case.test_id for case in case_list]
    duplicate_targets = sorted({test_id for test_id in test_ids if test_ids.count(test_id) > 1})
    if duplicate_targets:
        raise ValueError("duplicate regression test targets: " + ", ".join(duplicate_targets))

    browser_artifact = load_browser_catalog(browser_catalog_path)
    artifact_ids = [item.get("case_id") for item in browser_artifact["cases"]]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError("duplicate browser regression IDs")
    artifact_cases = {item["case_id"]: item for item in browser_artifact["cases"]}
    expected_browser_ids = set(EXPECTED_BROWSER_CLAIMS)
    if set(artifact_cases) != expected_browser_ids:
        raise ValueError("browser artifact IDs do not match the allowed browser claims")

    report_sections = _report_sections(artifact_report_path)
    for browser_id, expected in EXPECTED_BROWSER_CLAIMS.items():
        item = artifact_cases[browser_id]
        if item.get("report_section") != expected["report_section"]:
            raise ValueError(f"browser {browser_id} report_section does not match allowed claim")
        if item.get("coverage") != expected["coverage"]:
            raise ValueError(f"browser {browser_id} coverage does not match allowed claim")
        section = report_sections.get(expected["report_section"])
        if section is None:
            raise ValueError(
                f"browser {browser_id} report_section is absent from artifact report"
            )
        missing_evidence = [
            evidence for evidence in expected["report_evidence"] if evidence not in section
        ]
        if missing_evidence:
            raise ValueError(
                f"browser {browser_id} report_section lacks expected content: "
                + ", ".join(missing_evidence)
            )

    mapped_browser_ids = {browser_id for case in case_list for browser_id in case.browser_ids}
    unknown = sorted(mapped_browser_ids - artifact_cases.keys())
    if unknown:
        raise ValueError("unknown browser regression IDs: " + ", ".join(unknown))
    unmapped = sorted(artifact_cases.keys() - mapped_browser_ids)
    if unmapped:
        raise ValueError("browser regression IDs lack catalog cases: " + ", ".join(unmapped))
    for case in case_list:
        for browser_id in case.browser_ids:
            if f"test_{browser_id}_" not in case.test_id:
                raise ValueError(
                    f"case {case.case_id} browser ID {browser_id} does not match its test target"
                )

    loader = unittest.defaultTestLoader
    for case in case_list:
        if not case.test_id:
            raise ValueError(f"case {case.case_id} has no executable unittest mapping")
        loaded = loader.loadTestsFromName(case.test_id)
        tests = tuple(_walk_tests(loaded))
        if not tests or any(type(test).__name__ == "_FailedTest" for test in tests):
            raise ValueError(
                f"case {case.case_id} has no executable unittest mapping: {case.test_id!r}"
            )


CASES_BY_ID = {case.case_id: case for case in REGRESSION_CASES}
CASES_BY_FAMILY = {
    family: tuple(case for case in REGRESSION_CASES if case.family == family)
    for family in REQUIRED_FAMILIES
}
