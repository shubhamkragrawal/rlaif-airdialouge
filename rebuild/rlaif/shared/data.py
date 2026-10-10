"""Loading the dialogue corpus and the committed split. Standard library only.

Two rules this module exists to enforce:
  1. No script ever re-derives a split. `make_splits.py` writes `splits.json` once and
     everything else reads it through `load_splits`.
  2. `load_splits` asserts the three splits are pairwise disjoint and that the source file
     has not changed under them, and fails loudly if either is false. A silently
     contaminated held-out set produces a number that looks fine and means nothing.
"""

import hashlib
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SOURCE_FILE = os.path.join(REPO_ROOT, "airdialogue_sampled_5k.jsonl")
ARTIFACTS = os.path.join(REPO_ROOT, "rebuild", "artifacts")
SPLITS_FILE = os.path.join(ARTIFACTS, "splits.json")
DATASET_MANIFEST_FILE = os.path.join(ARTIFACTS, "dataset_manifest.json")

SPLIT_NAMES = ("held_out", "pair_gen", "sft")


def line_hash(raw_line):
    """Short, stable fingerprint of one raw JSONL line."""
    return hashlib.sha1(raw_line.encode("utf-8")).hexdigest()[:12]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_dataset_manifest(path=DATASET_MANIFEST_FILE, require_sft_examples=False):
    """Verify immutable corpus/split/example hashes before a training or eval run.

    The manifest is intentionally a checked-in artifact. A missing or changed file is
    a hard error, because line-index splits and derived examples would otherwise no
    longer describe the evidence being reproduced.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError("dataset manifest is required: %s" % path)
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported dataset manifest schema")
    files = manifest.get("files") or {}
    required = {"source", "splits"}
    if require_sft_examples:
        required.add("sft_examples")
    for name in required:
        entry = files.get(name) or {}
        target = os.path.join(REPO_ROOT, entry.get("path", ""))
        expected = entry.get("sha256")
        if not expected or not os.path.isfile(target):
            raise ValueError("dataset manifest entry is missing: %s" % name)
        actual = file_sha256(target)
        if actual != expected:
            raise ValueError("dataset %s changed (sha256 %s, expected %s)" %
                             (name, actual[:16], expected[:16]))
    return manifest


def load_dialogues(path=SOURCE_FILE):
    """Read the corpus. The dialogue id is the 0-based line index of this file."""
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, raw in enumerate(fh):
            raw = raw.rstrip("\n")
            if not raw:
                continue
            rec = json.loads(raw)
            rec["dialogue_id"] = i
            rec["line_hash"] = line_hash(raw)
            out.append(rec)
    return out


def load_splits(path=SPLITS_FILE, source_path=SOURCE_FILE, verify_source=True):
    """Load the committed split and refuse to return a contaminated one."""
    with open(path, "r", encoding="utf-8") as fh:
        splits = json.load(fh)

    sets = {}
    for name in SPLIT_NAMES:
        if name not in splits:
            raise ValueError("splits.json is missing the '%s' split" % name)
        ids = splits[name]
        sets[name] = set(ids)
        if len(sets[name]) != len(ids):
            raise ValueError("split '%s' contains duplicate dialogue ids" % name)

    for i, a in enumerate(SPLIT_NAMES):
        for b in SPLIT_NAMES[i + 1:]:
            overlap = sets[a] & sets[b]
            if overlap:
                raise ValueError(
                    "splits '%s' and '%s' overlap on %d dialogue ids (e.g. %s) - "
                    "the evaluation would be contaminated"
                    % (a, b, len(overlap), sorted(overlap)[:5])
                )

    if verify_source:
        actual = file_sha256(source_path)
        expected = splits.get("source_sha256")
        if expected and actual != expected:
            raise ValueError(
                "%s has changed since the split was written (sha256 %s, expected %s). "
                "Dialogue ids are line indices, so the split no longer means what it "
                "meant. Restore the file or re-run make_splits.py."
                % (os.path.basename(source_path), actual[:16], expected[:16])
            )

    # The source hash in splits.json protects line-index semantics. The immutable
    # manifest additionally protects the committed split file itself.
    if path == SPLITS_FILE and source_path == SOURCE_FILE and os.path.isfile(DATASET_MANIFEST_FILE):
        verify_dataset_manifest(require_sft_examples=False)

    return splits


def parse_turns(dialogue):
    """['customer: Hi.', 'agent: Hello.'] -> [('customer', 'Hi.'), ('agent', 'Hello.')].

    Turns whose speaker prefix is missing or unrecognised are dropped; the corpus has
    none, but a silent mis-parse here would poison every example built downstream.
    """
    turns = []
    for raw in dialogue:
        if ":" not in raw:
            continue
        speaker, text = raw.split(":", 1)
        speaker = speaker.strip().lower()
        if speaker not in ("customer", "agent"):
            continue
        turns.append((speaker, text.strip()))
    return turns


def agent_turn_indices(turns):
    """Positions of agent turns within `turns`, in order."""
    return [i for i, (speaker, _) in enumerate(turns) if speaker == "agent"]


def outcome_marker(record):
    """The closing marker the gold dialogue earns, from its labelled outcome.

    The corpus's agent turns end on a pleasantry and never carry a marker, but the system
    prompt requires one. Training on the raw turns would teach every SFT-derived arm to
    ignore that instruction, so arm A (zero-shot, which does follow it) would be the only
    arm ever scoring a task success. Appending the marker to the final agent turn is what
    keeps the headline metric measurable across all four arms.
    """
    from rlaif.shared.prompts import CONFIRM_MARKER, FAIL_MARKER

    action = record.get("expected_action") or record.get("action") or {}
    status = action.get("status")
    flights = action.get("flight") or []
    if status in ("book", "change") and flights:
        return "%s %s" % (CONFIRM_MARKER, flights[0])
    if status in ("book", "change", "cancel"):
        return CONFIRM_MARKER
    return FAIL_MARKER


def build_messages(record, turns, upto):
    """Chat messages for predicting the agent turn at index `upto`.

    system + KB, then the dialogue history strictly before `upto`. The agent is
    'assistant' and the customer is 'user'.
    """
    from rlaif.shared.prompts import system_message

    messages = [{"role": "system", "content": system_message(record.get("kb"))}]
    for speaker, text in turns[:upto]:
        messages.append(
            {"role": "assistant" if speaker == "agent" else "user", "content": text}
        )
    return messages
