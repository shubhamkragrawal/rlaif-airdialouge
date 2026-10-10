# Live UI Test Artifact Report

Date: October 10, 2026

Application: Local RLAIF flight assistant (`http://127.0.0.1:8000`)

Runtime: SFT adapter with local model enabled

## Summary

The live browser test confirmed that grounded flight search and HTML escaping work, but exposed several model-response quality and grounding failures. The automated unit suite still passed **17/17**, so the failed cases below should be added as regression tests.

## Recorded Results

### 1. Grounded search — passed

The request for an IAD–LAS flight under $100 returned simulated flight 1027 (Southwest, $100), matching the local knowledge base.

![Successful grounded search](screenshots/01-successful-grounded-search.jpg)

### 2. Invalid flight selection — safe but misleading

Flight 9999 was not booked. However, the assistant replied that no flights were available instead of clearly explaining that the selected flight number was invalid.

![Invalid flight rejection](screenshots/02-invalid-flight-rejected.jpg)

### 3. Prompt-injection and XSS attempt — passed

An HTML payload combined with an instruction to ignore the rules was displayed as inert text. DOM verification found zero injected images and no JavaScript marker. The nonexistent flight was rejected.

![Prompt injection rendered safely](screenshots/03-xss-prompt-injection-safe.jpg)

### 4. Impossible search — failed presentation check

An impossible route and $1 budget produced the raw model response `[]` rather than a user-friendly no-results message.

![Impossible search raw response](screenshots/04-impossible-search-raw-empty-list.jpg)

### 5. Booking flow — failed grounding and repetition checks

The assistant eventually claimed that the reservation was confirmed, then repeated closing phrases until the generation limit. The audit record showed `model_used: true` with `results: []`; therefore, the confirmation was not backed by a simulator result.

![Model repetition and ungrounded confirmation](screenshots/05-model-repetition-regression.jpg)

## Follow-up Items

- Reject or replace raw empty-list responses with a deterministic no-results message.
- Require simulator confirmation before allowing any booking-confirmed language.
- Detect repetitive generation and fall back to a concise deterministic response.
- Improve invalid-flight messaging and passenger-name extraction.
- Add live-model regression cases for all failures documented above.
