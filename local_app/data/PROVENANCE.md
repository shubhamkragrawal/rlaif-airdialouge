# Release fixture provenance

`held_out_50.jsonl` is a reduced fixture for the deterministic local 50-scenario
cohort. It contains the first 75 source records, reduced to their `intent`,
`expected_action`, `dialogue`, and `kb` fields. Keeping the intervening
ineligible records preserves source line numbers as scenario IDs. Selecting the
first 50 records whose `expected_action` (or fallback `action`) status is `book`
or `no_flight` therefore matches
`local_app.evaluation.load_scenarios(count=50, start=0)` and yields the original
scenario IDs through 74.

Source: repository file `airdialogue_sampled_5k.jsonl`, which derives from the
AirDialogue dataset. Full source SHA-256:
`2ee4061571077a338e3802317ce562f0cd8c5f2ed47fe01ffa479e1a5a8c33b3`.

Fixture SHA-256:
`dda7898144f9a326f558088353d3fcc16da6f4865daec7e8c4324f17496959e4`.

The upstream dataset's redistribution rights and license provenance are not
established in this repository. This small derived fixture is included only to
make the exact local regression cohort reproducible; its presence is not a
representation that the upstream dataset is openly licensed. Review applicable
AirDialogue terms before redistributing this package. The 55 MB source corpus
and private model weights are not included.
