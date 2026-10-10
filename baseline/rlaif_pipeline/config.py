# used by main pipeline
# CHECKPOINT_EVERY = 50
# N_CANDIDATES     = 1
# CHECKPOINT_LOCAL = "/tmp/rlaif_checkpoint.json"
# CHECKPOINT_BLOB  = "rlaif/checkpoints/rlaif_checkpoint.json"
# OUTPUT_LOCAL     = "/tmp/rlaif_best_of_2.jsonl"
# OUTPUT_BLOB      = "rlaif/output/rlaif_best_of_2.jsonl"




# # used by both simulator and judge
# intent_field_map = {
#         "departure_airport":  "Departing from",
#         "return_airport":     "Destination",
#         "departure_month":    "Departure month",
#         "departure_day":      "Departure day",
#         "departure_time_num": "Preferred departure time (0=early, 1=morning, 2=afternoon, 3=evening)",
#         "return_month":       "Return month",
#         "return_day":         "Return day",
#         "return_time_num":    "Preferred return time",
#         "budget":             "Maximum budget (USD)",
#         "num_people":         "Number of passengers",
#         "class":              "Seat class",
#         "max_connections":    "Maximum connections/stops allowed",
#         "goal":               "Booking goal",
#     }

# judge_retry_fallback_response ={
#         "score": -1.0,
#         "outcome": "critical_failure",
#         "justification": "Agent output was entirely incoherent, babbling, or unresponsive, causing evaluation failure.",
#         "reasoning": {
#             "booked_flight": "none",
#             "exists_in_db": False,
#             "hallucination_detected": True,
#             "hallucination_detail": "Agent completely hallucinated the conversation flow or failed to respond logically.",
#             "hard_constraints": {
#                 "budget": "not_applicable",
#                 "destination": "not_applicable",
#                 "dates": "not_applicable",
#                 "max_stops": "not_applicable"
#             },
#             "soft_constraints": "Failed to conduct a meaningful conversation.",
#             "conversation_quality": "Terrible. The agent was utterly confused and failed to assist the customer."
#         },
#         "error": True,
#     }

CHECKPOINT_EVERY = 50
N_CANDIDATES     = 1 # changed here!!! pivoted to refining of 1 response!!
CHECKPOINT_LOCAL = "/tmp/rlaif_checkpoint.json"
CHECKPOINT_BLOB  = "rlaif/checkpoints/rlaif_checkpoint.json"
OUTPUT_LOCAL     = "/tmp/rlaif_best_of_2.jsonl"
OUTPUT_BLOB      = "rlaif/output/rlaif_best_of_2.jsonl"
 
 
# used by both simulator and judge
intent_field_map = {
    "departure_airport":  "Departing from",
    "return_airport":     "Destination",
    "departure_month":    "Departure month",
    "departure_day":      "Departure day",
    "departure_time_num": "Preferred departure time (0=early, 1=morning, 2=afternoon, 3=evening)",
    "return_month":       "Return month",
    "return_day":         "Return day",
    "return_time_num":    "Preferred return time",
    "budget":             "Maximum budget (USD)",
    "num_people":         "Number of passengers",
    "class":              "Seat class",
    "max_connections":    "Maximum connections/stops allowed",
    "goal":               "Booking goal",
}
 
# Fallback returned when all judge retries fail.
# Score is now 1-10. Using 1 (worst) because a judge failure on a response
# that crashed evaluation is genuinely a bad signal — suppress that behaviour.
judge_retry_fallback_response = {
    "score": 1,
    "outcome": "critical_failure",
    "justification": "Agent output was entirely incoherent, babbling, or unresponsive, causing evaluation failure.",
    "reasoning": {
        "booked_flight": "none",
        "exists_in_db": False,
        "hallucination_detected": True,
        "hallucination_detail": "Agent completely hallucinated the conversation flow or failed to respond logically.",
        "hard_constraints": {
            "budget": "not_applicable",
            "destination": "not_applicable",
            "dates": "not_applicable",
            "max_stops": "not_applicable",
        },
        "soft_constraints": "Failed to conduct a meaningful conversation.",
        "conversation_quality": "Terrible. The agent was utterly confused and failed to assist the customer.",
    },
    "error": True,
}