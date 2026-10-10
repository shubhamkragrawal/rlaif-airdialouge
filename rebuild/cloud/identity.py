"""Guards for new cloud build and run identities.

Historical configurations remain readable as evidence, but preparation commands must
not silently recreate a historical image or reuse an old artifact prefix.  A fresh
preparation run therefore requires a date matching the image tag and a non-historical
suffix where a full run has a suffix.
"""

from datetime import date
import re


TAG_RE = re.compile(r"^(?P<date>\d{8})-v(?P<version>[1-9]\d*)$")
DATE_RE = re.compile(r"^\d{8}$")
SUFFIX_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")


def _date_value(value):
    if not DATE_RE.fullmatch(value):
        raise ValueError("date must be YYYYMMDD")
    try:
        return date(int(value[:4]), int(value[4:6]), int(value[6:]))
    except ValueError as exc:
        raise ValueError("date is not a real calendar date") from exc


def validate_identity(tag, run_date, suffix=None, *, allow_historical=False,
                      require_suffix=False, minimum_version=None):
    """Validate a new image/run identity and return its parsed fields.

    ``allow_historical`` is intentionally explicit for forensic regeneration.  Normal
    preparation rejects dates before today and rejects reusing a low image version.
    """
    match = TAG_RE.fullmatch(tag or "")
    if not match:
        raise ValueError("tag must have the immutable YYYYMMDD-vN form")
    tag_date = match.group("date")
    version = int(match.group("version"))
    requested = _date_value(run_date)
    if tag_date != run_date:
        raise ValueError("tag date %s does not match run date %s" % (tag_date, run_date))
    today = date.today()
    if not allow_historical and requested < today:
        raise ValueError("refusing historical date %s; use today's date or --allow-historical"
                         % run_date)
    if minimum_version is not None and version < minimum_version:
        raise ValueError("refusing historical image version v%d; require v%d or newer"
                         % (version, minimum_version))
    if suffix is not None:
        if not SUFFIX_RE.fullmatch(suffix):
            raise ValueError("suffix must be lowercase letters, digits, and hyphens")
        if suffix in {"v1", "v2", "v3", "v4", "adaptive-v1", "adaptive-v2",
                      "adaptive-v3", "adaptive-v4"} and not allow_historical:
            raise ValueError("refusing historical run suffix %r" % suffix)
    elif require_suffix:
        raise ValueError("a fresh full run requires an explicit non-historical suffix")
    return {"date": run_date, "version": version, "suffix": suffix}


def require_fresh_output(path):
    """Reject an output file that would overwrite a previously prepared artifact."""
    from pathlib import Path
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("refusing to overwrite existing preparation artifact: %s" % path)

