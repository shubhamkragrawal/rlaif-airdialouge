import unittest

from local_app.controller import ConversationController
from local_app.knowledge import FlightKnowledgeBase
from local_app.limits import ExecutionLimits


class CatalogControllerScenarioTests(unittest.TestCase):
    def test_single_message_selection_passenger_and_booking(self):
        controller = ConversationController(FlightKnowledgeBase(scenario_id=0))
        state = controller.new_session()

        result = controller.handle(
            state, "Book flight 1027 for Amy Gonzalez from IAD to LAS"
        )

        self.assertEqual(state.passenger, "Amy Gonzalez")
        self.assertEqual(state.selected_flight_id, 1027)
        self.assertIsNotNone(state.booking)
        self.assertEqual(state.booking["flight"]["flight_number"], 1027)
        self.assertIn("simulated booking is confirmed", result["text"])

    def test_turn_budget_exhaustion(self):
        controller = ConversationController(
            FlightKnowledgeBase(scenario_id=0),
            limits=ExecutionLimits(max_turns=1),
        )
        state = controller.new_session()

        controller.handle(state, "hello")
        result = controller.handle(state, "hello again")

        self.assertEqual(result["fallback_reason"], "turn_limit_exceeded")

    def test_generation_budget_exhaustion(self):
        class OverBudgetAdapter:
            last_usage = {"output_tokens": 9}

            def generate(self, *_args, **_kwargs):
                return "A concise response."

        controller = ConversationController(
            FlightKnowledgeBase(scenario_id=0),
            model_adapters={"dpo": OverBudgetAdapter()},
            limits=ExecutionLimits(max_generation_tokens=4),
        )

        result = controller.handle(controller.new_session("dpo"), "IAD to LAS")

        self.assertEqual(result["fallback_reason"], "generation_length_exceeded")
        self.assertFalse(result["model_used"])


if __name__ == "__main__":
    unittest.main()
