"""Strict, four-layer judge parser. Failure returns None, never a preference."""
import json
import re

LABELS = ("A", "B", "C", "D")


def _validate(obj):
    if not isinstance(obj, dict):
        return None
    scores, ranking = obj.get("scores"), obj.get("ranking")
    if not isinstance(scores, dict) or set(scores) != set(LABELS):
        return None
    if any(type(v) is not int or not 1 <= v <= 10 for v in scores.values()):
        return None
    if not isinstance(ranking, list) or len(ranking) != 4:
        return None
    if any(not isinstance(v, str) for v in ranking) or set(ranking) != set(LABELS):
        return None
    if any(scores[a] < scores[b] for a, b in zip(ranking, ranking[1:])):
        return None
    result = {"reasoning": str(obj.get("reasoning", "")), "scores": scores, "ranking": ranking}
    if "checks" in obj:
        result["checks"] = obj["checks"]
    return result


def parse_ranking(text):
    """Strip fences; scrub controls; recover the last complete score/ranking keys.

    Recovery never fills missing labels, invents a score, or repairs contradictory ranks.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    clean = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)
    for candidate in (clean, re.sub(r"[\x00-\x1f\x7f]", " ", clean)):
        try:
            obj = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        # Valid JSON with invalid semantics is rejected, not salvaged from nested text.
        return _validate(obj)
    decoder = json.JSONDecoder()
    recovered = {}
    for key in ("scores", "ranking", "checks"):
        # rfind avoids a quoted example in the judge's preceding reasoning.
        pos = clean.rfind('"' + key + '"')
        if pos < 0:
            if key == "checks":
                continue
            return None
        tail = clean[pos + len(key) + 2:]
        match = re.match(r"\s*:\s*", tail)
        if not match:
            return None
        try:
            recovered[key], _ = decoder.raw_decode(tail[match.end():])
        except ValueError:
            return None
    return _validate(recovered)


safe_parse_ranking = parse_ranking
