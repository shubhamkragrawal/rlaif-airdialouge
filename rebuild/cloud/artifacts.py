"""Upload complete files with a manifest last; checkpoint uploads run on the trainer thread."""
import hashlib
import json
from pathlib import Path


def download_tree(uri, local_dir):
    """Fetch a GCS directory (or one file) without accepting unsafe object paths."""
    from google.cloud import storage
    if not uri.startswith("gs://"):
        raise ValueError("Expected gs:// input URI")
    bucket_name, _, prefix = uri[5:].partition("/")
    root = Path(local_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    client = storage.Client()
    exact = client.bucket(bucket_name).blob(prefix)
    if exact.exists():
        target = root / Path(prefix).name
        exact.download_to_filename(str(target), timeout=300)
        return str(target)
    prefix = prefix.rstrip("/") + "/"
    count = 0
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        if blob.name.endswith("/"):
            continue
        relative = blob.name[len(prefix):]
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Unsafe GCS object path")
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(target), timeout=300)
        count += 1
    if not count:
        raise FileNotFoundError(uri)
    return str(root)


def sync_tree(root, uri):
    from google.cloud import storage
    if not uri.startswith("gs://"):
        raise ValueError("Expected gs:// output URI")
    bucket_name, _, prefix = uri[5:].partition("/")
    bucket = storage.Client().bucket(bucket_name)
    manifest = []
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        blob = bucket.blob(prefix.rstrip("/") + "/" + relative)
        blob.reload() if blob.exists() else None
        if not blob.metadata or blob.metadata.get("sha256") != digest:
            blob.metadata = {"sha256": digest}
            blob.upload_from_filename(str(path), timeout=300)
        manifest.append({"path": relative, "sha256": digest, "bytes": path.stat().st_size})
    bucket.blob(prefix.rstrip("/") + "/manifest.json").upload_from_string(
        json.dumps(manifest, indent=2), content_type="application/json")
    print("[upload] %d files -> %s" % (len(manifest), uri), flush=True)
