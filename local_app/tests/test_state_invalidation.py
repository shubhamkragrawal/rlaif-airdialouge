import unittest
from dataclasses import FrozenInstanceError

from local_app.controller import SessionState
from local_app.state_transitions import (
    ActiveStateInvalidation,
    ConfirmedBookingRecord,
    plan_passenger_change,
    plan_state_transition,
)
from local_app.tools import SearchCriteria


class StateInvalidationTests(unittest.TestCase):
    def setUp(self):
        self.criteria = SearchCriteria(
            origin="IAD", destination="LAS", departure_month="Jan",
            departure_day="1", return_month="Jan", return_day="3",
            max_price=300, max_connections=1,
            departure_time="morning", return_time="evening",
        )
        self.booking = {
            "booking_id": "booking_123", "hold_id": "hold_123",
            "flight": {
                "flight_number": 1027,
                "metadata": {"codeshares": ["XY42"], "tags": ["window"]},
            }, "passenger": "Amy Gonzalez",
            "simulated": True, "status": "confirmed",
            "audit": {"events": [{"kind": "confirmed"}]},
        }

    def test_material_criteria_changes_invalidate_active_state_and_preserve_record(self):
        changes = {
            "origin": "DCA", "destination": "PHX",
            "departure_day": "2", "return_day": "4",
            "max_price": 250, "max_connections": 0,
            "departure_time": "afternoon", "return_time": "night",
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                state = SessionState(
                    criteria=self.criteria, selected_flight_id=1027,
                    hold_id="hold_123", booking=self.booking,
                )
                next_criteria = SearchCriteria(**{
                    **self.criteria.__dict__, field: value,
                })
                plan = plan_state_transition(state, next_criteria)
                self.assertEqual(plan.changed_criteria, (field,))
                self.assertEqual(plan.active_invalidation, ActiveStateInvalidation(
                    search=True, selection=True, hold=True, booking_reference=True,
                ))
                record = plan.confirmed_records_to_preserve[0]
                self.assertIsInstance(record, ConfirmedBookingRecord)
                self.assertIsNot(record, state.booking)
                self.assertEqual(record.booking_id, "booking_123")
                self.assertEqual(record.passenger, "Amy Gonzalez")
                self.assertEqual(record.flight["flight_number"], 1027)
                self.assertEqual(state.booking["passenger"], "Amy Gonzalez")
                self.assertEqual(state.selected_flight_id, 1027)
                self.assertEqual(state.hold_id, "hold_123")

    def test_equivalent_criteria_is_noop(self):
        state = SessionState(criteria=self.criteria)
        plan = plan_state_transition(state, SearchCriteria(**self.criteria.__dict__))
        self.assertTrue(plan.is_noop)
        self.assertFalse(plan.criteria_changed)
        self.assertEqual(plan.changed_criteria, ())
        self.assertEqual(plan.active_invalidation, ActiveStateInvalidation())
        self.assertEqual(plan.confirmed_records_to_preserve, ())

    def test_failed_search_preserves_prior_active_state_for_atomic_rollback(self):
        state = SessionState(
            criteria=self.criteria, selected_flight_id=1027,
            hold_id="hold_123", booking=self.booking,
        )
        next_criteria = SearchCriteria(**{
            **self.criteria.__dict__, "max_price": 200,
        })
        plan = plan_state_transition(state, next_criteria, search_succeeded=False)
        self.assertTrue(plan.criteria_changed)
        self.assertFalse(plan.is_noop)
        self.assertFalse(plan.should_apply)
        self.assertEqual(plan.active_invalidation, ActiveStateInvalidation())
        self.assertEqual(plan.confirmed_records_to_preserve, ())
        self.assertEqual(state.criteria, self.criteria)
        self.assertEqual(state.selected_flight_id, 1027)
        self.assertEqual(state.hold_id, "hold_123")
        self.assertIs(state.booking, self.booking)

    def test_passenger_change_preserves_immutable_past_booking_evidence(self):
        state = SessionState(
            criteria=self.criteria, passenger="Amy Gonzalez",
            selected_flight_id=1027, hold_id="hold_123", booking=self.booking,
        )
        plan = plan_passenger_change(state, "Jordan Lee")
        self.assertEqual(plan.changed_criteria, ("passenger",))
        self.assertEqual(plan.active_invalidation, ActiveStateInvalidation(
            selection=True, hold=True, booking_reference=True,
        ))
        self.assertEqual(len(plan.confirmed_records_to_preserve), 1)
        record = plan.confirmed_records_to_preserve[0]
        self.assertIsInstance(record, ConfirmedBookingRecord)
        self.assertEqual(record.passenger, "Amy Gonzalez")
        self.assertEqual(state.booking["passenger"], "Amy Gonzalez")
        self.assertEqual(state.passenger, "Amy Gonzalez")

    def test_confirmed_record_is_a_deeply_immutable_value_snapshot(self):
        state = SessionState(criteria=self.criteria, booking=self.booking)
        record = plan_state_transition(
            state,
            SearchCriteria(**{**self.criteria.__dict__, "max_price": 200}),
        ).confirmed_records_to_preserve[0]

        # Mutating every nested source layer after planning cannot alter history.
        self.booking["flight"]["metadata"]["codeshares"][0] = "MUTATED"
        self.booking["flight"]["metadata"]["tags"].append("aisle")
        self.booking["audit"]["events"][0]["kind"] = "rewritten"
        self.booking["audit"]["events"].append({"kind": "added"})
        self.assertEqual(record.flight["metadata"]["codeshares"], ("XY42",))
        self.assertEqual(record.flight["metadata"]["tags"], ("window",))
        self.assertEqual(record.additional_fields["audit"]["events"][0]["kind"], "confirmed")
        self.assertEqual(len(record.additional_fields["audit"]["events"]), 1)

        # Each container layer and the record object itself reject writes.
        with self.assertRaises(TypeError):
            record.flight["flight_number"] = 1
        with self.assertRaises(TypeError):
            record.flight["metadata"]["codeshares"][0] = "MUTATED"
        with self.assertRaises(TypeError):
            record.additional_fields["audit"]["events"][0]["kind"] = "MUTATED"
        with self.assertRaises(FrozenInstanceError):
            record.passenger = "Jordan Lee"

    def test_equivalent_passenger_is_noop(self):
        state = SessionState(passenger="Amy Gonzalez", booking=self.booking)
        plan = plan_passenger_change(state, "Amy Gonzalez")
        self.assertTrue(plan.is_noop)
        self.assertEqual(plan.active_invalidation, ActiveStateInvalidation())
        self.assertEqual(plan.confirmed_records_to_preserve, ())


if __name__ == "__main__":
    unittest.main()
