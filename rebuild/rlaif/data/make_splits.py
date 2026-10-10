"""Step zero: write the three-way dialogue split, once.

Run this before any other script. Everything downstream reads artifacts/splits.json;
nothing else is allowed to derive a split of its own.

    python3 make_splits.py

Sizes follow the Day 1 brief:
    held_out  250   evaluation (the 50 eval scenarios come from here, 200 spare)
    pair_gen  700   preference-pair generation
    sft      4050   SFT training examples

pair_gen is disjoint from sft for yield, not hygiene: sampling four continuations for a
prefix the SFT policy just trained on reproduces something near the gold turn four times,
the judge ranks them flat, and the gap filter throws the pair away.
"""

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rlaif.shared.data as D

SEED = 20260922
SIZES = [("held_out", 250), ("pair_gen", 700), ("sft", 4050)]


def main():
    records = D.load_dialogues()
    n = len(records)
    total = sum(size for _, size in SIZES)
    if total > n:
        raise SystemExit(
            "split sizes total %d but only %d dialogues are available" % (total, n)
        )

    ids = [r["dialogue_id"] for r in records]
    rng = random.Random(SEED)
    shuffled = list(ids)
    rng.shuffle(shuffled)

    splits = {}
    cursor = 0
    for name, size in SIZES:
        splits[name] = sorted(shuffled[cursor:cursor + size])
        cursor += size

    # Per-dialogue goal and outcome, so downstream scripts can filter without reopening
    # the 55MB corpus. Needed because cancel/change dialogues are only partly answerable
    # from the data we have - see NOTES.md.
    goals = {}
    statuses = {}
    for r in records:
        goals[r["dialogue_id"]] = (r.get("intent") or {}).get("goal")
        statuses[r["dialogue_id"]] = (r.get("expected_action") or {}).get("status")

    payload = {
        "seed": SEED,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_file": os.path.basename(D.SOURCE_FILE),
        "source_sha256": D.file_sha256(D.SOURCE_FILE),
        "n_dialogues": n,
        "id_scheme": "0-based line index of source_file",
        "sizes": dict(SIZES),
        "unassigned": n - total,
        "goal": {str(k): v for k, v in goals.items()},
        "status": {str(k): v for k, v in statuses.items()},
        "line_hash": [r["line_hash"] for r in records],
    }
    for name, _ in SIZES:
        payload[name] = splits[name]

    if not os.path.isdir(D.ARTIFACTS):
        os.makedirs(D.ARTIFACTS)
    with open(D.SPLITS_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    # Read it straight back through the loader, so the disjointness assertion that every
    # downstream script relies on is exercised here first.
    check = D.load_splits()
    print("wrote %s" % D.SPLITS_FILE)
    print("  seed          %d" % check["seed"])
    print("  source sha256 %s" % check["source_sha256"][:16])
    for name, size in SIZES:
        got = check[name]
        by_goal = {}
        for i in got:
            g = check["goal"].get(str(i))
            by_goal[g] = by_goal.get(g, 0) + 1
        print("  %-9s %5d  goals %s" % (name, len(got), by_goal))
    print("  unassigned    %5d" % check["unassigned"])
    print("  pairwise disjointness: OK")


if __name__ == "__main__":
    main()
