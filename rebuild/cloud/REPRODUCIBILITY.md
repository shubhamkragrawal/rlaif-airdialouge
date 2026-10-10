# Reproducibility limits and provenance

The supported package refuses to run against a changed dialogue corpus, split file, or
committed SFT example file. Their SHA256 values are recorded in
`rebuild/artifacts/dataset_manifest.json`; `rlaif.shared.data.verify_dataset_manifest`
checks them before training, and `load_splits` checks the source and split provenance
before preference generation, PPO, DPO, or evaluation.

The historical 8–9 October runs used the model identifier
`Qwen/Qwen2.5-3B-Instruct` without an explicit Hugging Face commit revision. Their
evaluation artifacts record the resolved revision after loading, but exact from-scratch
training reproduction cannot be claimed unless that original revision is recovered
from the cloud cache or artifact metadata.

The dated job JSON files retain the historical image tags, resource bounds,
arguments, and artifact prefixes, but their command fields were normalized to the
post-restructure `rlaif.*` module paths. They are therefore archival parameter
records, not byte-for-byte copies of the submitted manifests, and the module commands
must not be used with the historical flat-layout images. Fresh configurations are
written separately by the preparation commands below.

New closeout preparation requires `--model-revision` and rejects moving names such as
`main` or `latest`. Training and evaluation scripts accept the pinned revision and
record the requested/resolved value in run metadata. This preserves the historical
results while making the limitation explicit rather than silently claiming exact
reproduction.

The closeout preparation command writes a new, immutable source manifest and fresh job
configuration under a caller-selected empty directory. It requires a current image
date, a matching `YYYYMMDD-vN` tag (v14 or newer), and a fresh suffix; it never
overwrites the v13 image metadata or historical run prefixes.
