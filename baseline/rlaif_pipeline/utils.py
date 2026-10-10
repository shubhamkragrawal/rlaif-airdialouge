import os
import logging
from google.cloud import storage, secretmanager
from config import (
    CHECKPOINT_EVERY,
    N_CANDIDATES,
    CHECKPOINT_LOCAL,
    CHECKPOINT_BLOB,
    OUTPUT_LOCAL,
    OUTPUT_BLOB
)
import json
from datetime import datetime, timezone

# initializing logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
# Google cloud service functions for download and upload

def get_secret(project: str, secret_name: str) -> str:
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project}/secrets/{secret_name}/versions/latest"
    return client.access_secret_version(name=name).payload.data.decode("utf-8")

def download_dir_from_gcs(bucket_name: str, prefix: str, local_dir: str) -> None:
    client = storage.Client()
    blobs = list(client.bucket(bucket_name).list_blobs(prefix=prefix))
    if not blobs:
        raise FileNotFoundError(f"No files at gs://{bucket_name}/{prefix}")
    os.makedirs(local_dir, exist_ok=True)
    for blob in blobs:
        relative = blob.name[len(prefix):].lstrip("/")
        if not relative:
            continue
        local_path = os.path.join(local_dir, relative)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
    logger.info(f"Downloaded {len(blobs)} files from gs://{bucket_name}/{prefix}")


def upload_dir_to_gcs(bucket_name, local_dir, gcs_prefix):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            relative   = os.path.relpath(local_path, local_dir)
            blob_path  = os.path.join(gcs_prefix, relative).replace("\\", "/")
            bucket.blob(blob_path).upload_from_filename(local_path)
            logger.info(f"Uploaded {local_path} :: gs://{bucket_name}/{blob_path}")

def download_from_gcs(bucket_name: str, blob_path: str, local_path: str) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_path).download_to_filename(local_path)
    logger.info(f"Downloaded gs://{bucket_name}/{blob_path} :: {local_path}")


def upload_to_gcs(bucket_name: str, local_path: str, blob_path: str) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_path).upload_from_filename(local_path)
    logger.info(f"Uploaded {local_path} :: gs://{bucket_name}/{blob_path}")


def parse_gcs_uri(uri: str):
    """Parse gs://bucket/path into (bucket, path)."""
    assert uri.startswith("gs://"), f"Expected gs:// URI, got: {uri}"
    parts = uri[5:].split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""

# checkpoint snippets
def save_checkpoint(bucket: str, processed_indices: list, scores: list, count: int) -> None:
    data = {
        "processed_indices": processed_indices,
        "scores": scores,
        "example_count": count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(CHECKPOINT_LOCAL, "w") as f:
        json.dump(data, f)
    upload_to_gcs(bucket, CHECKPOINT_LOCAL, CHECKPOINT_BLOB)
    upload_to_gcs(bucket, OUTPUT_LOCAL, OUTPUT_BLOB)
    logger.info(f"Checkpoint saved — {count} examples done")


def load_checkpoint(bucket: str) -> dict | None:
    try:
        download_from_gcs(bucket, CHECKPOINT_BLOB, CHECKPOINT_LOCAL)
        with open(CHECKPOINT_LOCAL) as f:
            cp = json.load(f)
        logger.info(f"Resuming from checkpoint — {cp['example_count']} already done")
        try:
            download_from_gcs(bucket, OUTPUT_BLOB, OUTPUT_LOCAL)
        except Exception:
            logger.warning("Output file not in GCS — will create fresh")
        return cp
    except Exception:
        logger.info("No checkpoint found — starting fresh")
        return None