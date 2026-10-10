# Rebuild — decisions and deviations

Layout note: scripts live in `rebuild/`, generated artifacts in `rebuild/artifacts/`.
The Day 1 brief lists the artifact filenames flat; splitting code from outputs is the only
change, and the filenames themselves are unchanged.

## 1 · We render the KB ourselves; `parsed_kb` is not usable

The corpus ships a pre-rendered `parsed_kb` string from the original repo. It is lossy: it
re-indexes flights as "Flight 1..30" instead of their real flight numbers (which are what
the dialogues actually reference — "flight number 1027"), drops both airports, both dates
and the cabin class, and prints every departure time as `N/A`.

A judge asked "does the flight the agent named exist in the KB, and are its airline, times,
airports, stop count and price accurate?" cannot answer any of that from `parsed_kb`, and a
policy trained against it can only hallucinate the missing fields. `prompts.render_kb`
builds the table from the raw `kb` list instead, one line per flight with every field.

## 2 · Outcome markers are appended to the final gold agent turn

The system prompt requires every conversation to close with `BOOKING_CONFIRMED <flight>` or
`BOOKING_FAILED`. The corpus's agent turns never contain either — they close on a
pleasantry ("Thank you for choosing us.").

Training on the raw turns would therefore teach arms B, C and D to ignore that instruction,
while arm A (zero-shot, which does follow it) would be the only arm ever scoring a task
success. The headline metric would measure whether an arm had been fine-tuned, not whether
it booked the right flight.

So `rlaif_data.outcome_marker` derives the marker from the dialogue's labelled
`expected_action` and `build_sft_data.py` appends it to the final agent turn:

| `expected_action.status` | marker |
|---|---|
| `book`, `change` with a flight | `BOOKING_CONFIRMED <flight_number>` |
| `cancel` | `BOOKING_CONFIRMED` |
| `no_flight`, `no_reservation` | `BOOKING_FAILED` |

Putting the flight number in the marker also makes Day 3's task-success check a plain
string comparison against `expected_action.flight`, with no transcript parsing and no judge
in the loop. Note `expected_action.flight` is a *set* of acceptable flights (1–6 entries),
so success is membership, not equality.

Marker-bearing final turns are capped at 30% of the 10k budget, so the model learns the
marker without learning to close every message with a pleasantry.

## 3 · `cancel` and `change` dialogues are only partly answerable — a data limit

26% of the corpus asks the agent to cancel or change an existing reservation
(`intent.goal`), and 46% of labelled outcomes are `no_flight` / `no_reservation`. But the
record carries no reservation field: the `kb` list holds available flights only. There is
no honest way for the agent to know whether a customer has a booking, so on those dialogues
it can only guess, and the gold turn it is trained toward is unpredictable from its input.

Nothing is dropped — the splits cover all 5,000 dialogues as the brief specifies, and
`splits.json` records each dialogue's `goal` and `status` so any script can filter.
**Recommendation for Day 3: draw the 50 evaluation scenarios from `held_out` where
`goal == "book"`** (187 of the 250 qualify). Report it in the README as a known limit of
the corpus rather than a result.

## 4 · Split sizes consume the corpus exactly

250 + 700 + 4,050 = 5,000. There is no spare pool. If a split needs to grow, another must
shrink, and `splits.json` has to be rewritten and everything downstream re-run.

## 5 · Still to confirm on the GPU box

The token-length histogram and truncation count are character-estimated locally (no
tokenizer installed here; Python 3.9, no torch/transformers). Before training, run:

    python3 build_sft_data.py --tokenizer Qwen/Qwen2.5-3B-Instruct --max-seq-len 2048

Character lengths (p50 4,308, max 5,463) suggest ~1,200 tokens typical and well under
2,048, but that is an estimate, not the count.
