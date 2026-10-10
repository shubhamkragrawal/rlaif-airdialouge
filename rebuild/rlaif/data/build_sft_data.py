"""Build the turn-level SFT dataset from the `sft` split.

    python3 build_sft_data.py                      # structure + character stats
    python3 build_sft_data.py --tokenizer Qwen/Qwen2.5-3B-Instruct   # real token histogram

One example per agent turn: prompt = system prompt + KB + dialogue history up to turn t,
completion = the agent's turn t. This matches the pair-generation setup exactly (prefix in,
single reply out), so the training and sampling distributions agree. The KB is re-tokenized
once per turn, which is wasteful and irrelevant at this scale.

The opening agent turn is always skipped - it is a fixed greeting, it teaches nothing, and
it would dilute the budget.

The final agent turn of each dialogue gets its outcome marker appended (see
rlaif_data.outcome_marker for why). Those examples are capped at a share of the budget so
the model learns the marker without learning to close every message with a pleasantry.
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rlaif.shared.data as D

SEED = 20260922
MAX_EXAMPLES = 10000
FINAL_TURN_SHARE = 0.30  # at most this fraction of the budget spent on marker-bearing turns
OUT_FILE = os.path.join(D.ARTIFACTS, "sft_examples.jsonl")


def build_candidates(records_by_id, sft_ids):
    """All eligible (dialogue, turn) examples, grouped by dialogue."""
    per_dialogue = []
    skipped_greeting = 0
    for did in sft_ids:
        rec = records_by_id.get(did)
        if rec is None:
            continue
        turns = D.parse_turns(rec["dialogue"])
        agent_idx = D.agent_turn_indices(turns)
        if len(agent_idx) < 2:
            continue
        skipped_greeting += 1
        marker = D.outcome_marker(rec)
        items = []
        for ordinal, turn_index in enumerate(agent_idx):
            if ordinal == 0:
                continue  # the greeting
            is_final = turn_index == agent_idx[-1]
            completion = turns[turn_index][1]
            if is_final:
                completion = completion + "\n" + marker
            items.append({
                "dialogue_id": did,
                "turn_index": turn_index,
                "agent_turn_ordinal": ordinal,
                "is_final": is_final,
                "messages": D.build_messages(rec, turns, turn_index),
                "completion": completion,
            })
        if items:
            per_dialogue.append(items)
    return per_dialogue, skipped_greeting


def select(per_dialogue, budget, rng):
    """Spend the budget across dialogues so no single dialogue dominates.

    Pass 1 takes final turns (capped), pass 2 round-robins the rest by turn ordinal, so
    every dialogue contributes its second agent turn before any dialogue contributes its
    fifth.
    """
    finals = [items[-1] for items in per_dialogue if items[-1]["is_final"]]
    rng.shuffle(finals)
    final_budget = min(len(finals), int(budget * FINAL_TURN_SHARE))
    chosen = finals[:final_budget]
    kept_final = set((e["dialogue_id"], e["turn_index"]) for e in chosen)

    rounds = []
    depth = max(len(items) for items in per_dialogue) if per_dialogue else 0
    order = list(range(len(per_dialogue)))
    for d in range(depth):
        rng.shuffle(order)
        for di in order:
            items = per_dialogue[di]
            if d < len(items):
                e = items[d]
                if (e["dialogue_id"], e["turn_index"]) not in kept_final:
                    rounds.append(e)

    chosen.extend(rounds[:max(0, budget - len(chosen))])
    chosen.sort(key=lambda e: (e["dialogue_id"], e["turn_index"]))
    return chosen


def histogram(values, buckets, label):
    print("  %s" % label)
    prev = 0
    total = len(values)
    for edge in buckets:
        n = sum(1 for v in values if prev <= v < edge)
        bar = "#" * int(60.0 * n / total) if total else ""
        print("    %6d-%-6d %6d  %s" % (prev, edge, n, bar))
        prev = edge
    n = sum(1 for v in values if v >= prev)
    print("    %6d+       %6d  %s" % (prev, n, "#" * int(60.0 * n / total) if total else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=None,
                    help="HF tokenizer id; when given, prints a real token histogram")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--budget", type=int, default=MAX_EXAMPLES)
    args = ap.parse_args()

    splits = D.load_splits()
    records = D.load_dialogues()
    records_by_id = dict((r["dialogue_id"], r) for r in records)
    sft_ids = splits["sft"]

    rng = random.Random(SEED)
    per_dialogue, n_dialogues = build_candidates(records_by_id, sft_ids)
    n_candidates = sum(len(items) for items in per_dialogue)
    chosen = select(per_dialogue, args.budget, rng)

    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        for e in chosen:
            fh.write(json.dumps(e) + "\n")

    used_dialogues = len(set(e["dialogue_id"] for e in chosen))
    n_final = sum(1 for e in chosen if e["is_final"])
    markers = {}
    for e in chosen:
        if e["is_final"]:
            m = e["completion"].rsplit("\n", 1)[-1].split(" ")[0]
            markers[m] = markers.get(m, 0) + 1

    print("wrote %s" % OUT_FILE)
    print("  sft split dialogues        %5d" % len(sft_ids))
    print("  dialogues with >=2 agent turns %5d" % n_dialogues)
    print("  candidate turns (greeting excluded) %6d" % n_candidates)
    print("  examples written           %5d" % len(chosen))
    print("  distinct dialogues used    %5d" % used_dialogues)
    print("  final-turn examples        %5d  markers %s" % (n_final, markers))
    per = {}
    for e in chosen:
        per[e["dialogue_id"]] = per.get(e["dialogue_id"], 0) + 1
    print("  examples per dialogue      min %d  max %d" % (min(per.values()), max(per.values())))

    texts = []
    for e in chosen:
        body = "".join(m["content"] for m in e["messages"]) + e["completion"]
        texts.append(body)
    chars = [len(t) for t in texts]
    chars.sort()
    print("  characters  p50 %d  p95 %d  max %d" % (
        chars[len(chars) // 2], chars[int(len(chars) * 0.95)], chars[-1]))

    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        lens = []
        for e in chosen:
            msgs = e["messages"] + [{"role": "assistant", "content": e["completion"]}]
            lens.append(len(tok.apply_chat_template(msgs, tokenize=True)))
        lens.sort()
        over = sum(1 for v in lens if v > args.max_seq_len)
        print("  tokens      p50 %d  p95 %d  p99 %d  max %d" % (
            lens[len(lens) // 2], lens[int(len(lens) * 0.95)],
            lens[int(len(lens) * 0.99)], lens[-1]))
        print("  would truncate at max_seq_len=%d: %d of %d (%.2f%%)" % (
            args.max_seq_len, over, len(lens), 100.0 * over / len(lens)))
        histogram(lens, [1024, 1536, 2048, 2560, 3072, 4096], "token length histogram")
        if over > 0.02 * len(lens):
            print("  WARNING: truncation above 2%. Raise max_seq_len to 3072 rather than "
                  "clipping KBs - a clipped KB trains the model to hallucinate flights, "
                  "which is the exact failure the judge penalises.")
    else:
        print("  (no tokenizer available here - run with --tokenizer on the GPU box for "
              "the real token histogram and truncation count before training)")
        est = [int(c / 3.6) for c in chars]
        histogram(est, [1024, 1536, 2048, 2560, 3072, 4096],
                  "ROUGH token estimate (chars/3.6), not a substitute for the real count")


if __name__ == "__main__":
    main()
