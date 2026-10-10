import unittest

from local_app.controller import ConversationController
from local_app.knowledge import FlightKnowledgeBase
from local_app.runtime import ModelAdapter


class RecordingAdapter(ModelAdapter):
    def __init__(self, name="dpo", response="Flight 1027 is the grounded choice at $100."):
        self.name = name
        self.response = response
        self.calls = []

    def generate(self, messages, retrieved_flights):
        self.calls.append((messages, retrieved_flights))
        return self.response


class FailingAdapter(ModelAdapter):
    name = "dpo"

    def generate(self, messages, retrieved_flights):
        raise RuntimeError("model backend failed")


class RuntimeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.kb = FlightKnowledgeBase(scenario_id=0)

    def test_selected_model_adapter_generates_with_retrieved_context(self):
        adapter = RecordingAdapter()
        controller = ConversationController(self.kb, model_adapters={"dpo": adapter})
        session = controller.new_session("dpo")

        result = controller.handle(session, "Find a flight from IAD to LAS under $100")

        self.assertTrue(result["model_used"])
        self.assertEqual(result["text"], adapter.response)
        self.assertEqual(len(adapter.calls), 1)
        messages, flights = adapter.calls[0]
        self.assertIn(1027, [flight["flight_number"] for flight in flights])
        self.assertEqual(messages[-1]["content"], "Find a flight from IAD to LAS under $100")
        self.assertIn("safe draft", messages[0]["content"])

    def test_ungrounded_model_output_falls_back_to_deterministic_response(self):
        adapter = RecordingAdapter(response="Flight 9999 costs $1 and is available.")
        controller = ConversationController(self.kb, model_adapters={"dpo": adapter})

        result = controller.handle(
            controller.new_session("dpo"), "Find a flight from IAD to LAS under $100"
        )

        self.assertFalse(result["model_used"])
        self.assertIn("1027", result["text"])
        self.assertNotIn("9999", result["text"])

    def test_model_price_must_match_retrieved_context(self):
        adapter = RecordingAdapter(response="Flight 1027 costs $99.")
        controller = ConversationController(self.kb, model_adapters={"dpo": adapter})

        result = controller.handle(
            controller.new_session("dpo"), "Find a flight from IAD to LAS under $100"
        )

        self.assertFalse(result["model_used"])
        self.assertNotIn("$99", result["text"])

    def test_model_cannot_deny_non_empty_search_results(self):
        adapter = RecordingAdapter(response="Sorry, there are no available flights.")
        controller = ConversationController(self.kb, model_adapters={"dpo": adapter})

        result = controller.handle(
            controller.new_session("dpo"), "Find a flight from IAD to LAS under $100"
        )

        self.assertFalse(result["model_used"])
        self.assertIn("1027", result["text"])

    def test_model_failure_falls_back_without_breaking_the_session(self):
        controller = ConversationController(
            self.kb, model_adapters={"dpo": FailingAdapter()}
        )

        result = controller.handle(
            controller.new_session("dpo"), "Find a flight from IAD to LAS under $100"
        )

        self.assertFalse(result["model_used"])
        self.assertIn("I found these simulated flights", result["text"])

    def test_unconfigured_adapter_keeps_deterministic_baseline(self):
        configured = RecordingAdapter()
        controller = ConversationController(self.kb, model_adapters={"dpo": configured})

        result = controller.handle(
            controller.new_session("sft"), "Find a flight from IAD to LAS under $100"
        )

        self.assertFalse(result["model_used"])
        self.assertEqual(configured.calls, [])
        self.assertIn("1027", result["text"])


if __name__ == "__main__":
    unittest.main()
