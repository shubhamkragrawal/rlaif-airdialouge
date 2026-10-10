import unittest

from local_app.parsing import (MAX_NAME_LENGTH, normalize_name_text,
                               parse_passenger_name, passenger_name_candidates)


class PassengerNameParsingTests(unittest.TestCase):
    def test_explicit_and_embedded_name_phrases(self):
        cases = (
            ("My name is Amy Gonzalez", "Amy Gonzalez"),
            ("Please book for José O’Neil", "José O'Neil"),
            ("passenger: Anne-Marie Smith", "Anne-Marie Smith"),
            ("Search IAD to LAS for Amy Gonzalez", "Amy Gonzalez"),
            ("Book flight 1027 for Amy Gonzalez, please", "Amy Gonzalez"),
            ("Amy Gonzalez is traveling tomorrow", "Amy Gonzalez"),
            ("José García is travelling", "José García"),
            ("José García will be traveling", "José García"),
        )
        for message, expected in cases:
            with self.subTest(message=message):
                result = parse_passenger_name(message)
                self.assertEqual(result.status, "found")
                self.assertEqual(result.name, expected)
                self.assertTrue(result.matched_phrase)

    def test_collects_every_passenger_in_mixed_contexts(self):
        cases = (
            ("My name is Amy Gonzalez, José García is traveling",
             {"Amy Gonzalez", "José García"}),
            ("Search IAD to LAS for Amy Gonzalez. Book for José García",
             {"Amy Gonzalez", "José García"}),
            ("Search IAD to LAS for Amy Gonzalez and for José García",
             {"Amy Gonzalez", "José García"}),
            ("My name is Amy Gonzalez. José García will be traveling",
             {"Amy Gonzalez", "José García"}),
            ("José García is travelling", {"José García"}),
            ("Passengers Amy Gonzalez and José García are traveling",
             {"Amy Gonzalez", "José García"}),
            ("Passengers Amy Gonzalez and Jean‑Luc Picard are traveling",
             {"Amy Gonzalez", "Jean-Luc Picard"}),
            ("Jose\u0301 Garci\u0301a is travelling after another name",
             {"José García"}),
            ("Passengers अमित कुमार and Amy Gonzalez are traveling",
             {"अमित कुमार", "Amy Gonzalez"}),
        )
        for message, expected in cases:
            with self.subTest(message=message):
                found = {candidate.candidate for candidate in passenger_name_candidates(message)
                         if candidate.candidate}
                self.assertEqual(found, expected)

    def test_travel_clause_does_not_treat_airport_codes_as_passengers(self):
        outcomes = passenger_name_candidates("Passengers JFK and LAX are traveling")
        self.assertFalse(any(candidate.found for candidate in outcomes))

    def test_passenger_name_stops_before_all_following_flight_criteria(self):
        for message in (
            "Book flight 1027 for Amy Gonzalez from IAD to LAS departing 1/1 returning 1/3 under $500",
            "Find from IAD to LAS for Amy Gonzalez departing in the morning and returning in the evening",
        ):
            with self.subTest(message=message):
                result = parse_passenger_name(message)
                self.assertEqual(result.status, "found")
                self.assertEqual(result.name, "Amy Gonzalez")

    def test_unicode_normalization_and_whitespace(self):
        self.assertEqual(normalize_name_text("  Ａｎｎｅ Marie—Jones. "), "Anne Marie-Jones")

    def test_rejects_commands_codes_and_unsafe_values(self):
        for message in (
            "book for JFK", "passenger book", "my name is search flights",
            "book for <script>alert(1)</script>", "passenger A",
        ):
            with self.subTest(message=message):
                self.assertEqual(parse_passenger_name(message).status, "rejected")

    def test_rejected_non_latin_candidate_is_preserved_for_privacy_redaction(self):
        result = parse_passenger_name("My name is अनन्या   शर्मा")
        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.candidate, "अनन्या शर्मा")

    def test_no_unanchored_name_guess(self):
        result = parse_passenger_name("Search flights from IAD to LAS for next week")
        self.assertEqual(result.status, "rejected")

    def test_length_limit(self):
        result = parse_passenger_name("passenger " + "A" * 110)
        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.reason, "name_too_long")
        self.assertEqual(len(result.candidate or ""), 110)


if __name__ == "__main__":
    unittest.main()
