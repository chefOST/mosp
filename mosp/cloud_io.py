"""Cloud-only asset, R2 inventory and feature-cache I/O for MOSP."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

from .features import (CHECKPOINT_FILE_ID, CHECKPOINT_FILENAME, EGOPER_COMMIT,
                       SOURCE_GIT_BLOBS, file_sha256, load_feature_artifact)
from .runtime import write_json

BUCKET = "chefost-moscato-gt-videos"
ENDPOINT = "https://2aee25b6b08cc209d7cab66f1d0942fd.r2.cloudflarestorage.com"
FEATURE_PREFIX = "moscato/features/egoper-i3d-r50-k400-v1"
VIDEO_MANIFESTS = (
    "moscato/releases/564f350e536734db/videos/manifest.json",
    "moscato/non_ground_truth/manifest.json",
)


def s3_client():
    import boto3
    from botocore.config import Config
    return boto3.client("s3", endpoint_url=ENDPOINT, region_name="auto",
                        config=Config(retries={"max_attempts": 8, "mode": "adaptive"},
                                      request_checksum_calculation="when_required",
                                      response_checksum_validation="when_required"))


def read_json_object(client, key):
    response = client.get_object(Bucket=BUCKET, Key=key)
    try:
        return json.loads(response["Body"].read())
    finally:
        response["Body"].close()


def video_inventory(client):
    """Normalize the two existing collections without reading video bytes."""
    rows, seen = [], set()
    for manifest_key in VIDEO_MANIFESTS:
        document = read_json_object(client, manifest_key)
        prefix = manifest_key.rsplit("/", 1)[0]
        for row in document["entries"]:
            dataset = row.get("dataset")
            if dataset not in ("CMU", "EGTEA"):
                continue
            video_id = row.get("video_id", row.get("ground_truth_video_id"))
            key = row.get("object_key")
            if key is None and row.get("local_path"):
                key = prefix + "/" + row["local_path"]
            sha, size = row.get("sha256"), row.get("size_bytes", row.get("bytes"))
            if not video_id or not key or not sha or not size:
                raise ValueError(f"Incomplete R2 inventory entry: {dataset}/{video_id}")
            identity = (dataset, video_id)
            if identity in seen or Path(key).stem != video_id or ".." in Path(key).parts:
                raise ValueError(f"Duplicate or unsafe R2 inventory identity: {identity}")
            seen.add(identity)
            media = row.get("video", row.get("probe", row.get("media", {})))
            if "streams" in media:
                media = next(s for s in media["streams"] if s.get("codec_type", "video") == "video")
            rows.append({"dataset": dataset, "video_id": video_id,
                         "source": {"bucket": BUCKET, "key": key, "sha256": sha, "size_bytes": int(size)},
                         "video": media, "inventory_key": manifest_key})
    return {"schema_version": 1, "videos": rows}


def download_verified_video(client, row, destination):
    """Only called inside a cloud worker's temporary directory."""
    source, destination = row["source"], Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    head = client.head_object(Bucket=source["bucket"], Key=source["key"])
    if head["ContentLength"] != source["size_bytes"]:
        raise ValueError(f"R2 video size changed: {row['video_id']}")
    client.download_file(source["bucket"], source["key"], str(destination))
    if destination.stat().st_size != source["size_bytes"] or file_sha256(destination) != source["sha256"]:
        raise ValueError(f"R2 video checksum mismatch: {row['video_id']}")
    return destination


def put_verified_file(client, path, key, *, metadata=None, content_type="application/octet-stream"):
    """Create immutable feature/checksum objects; identical retries are allowed."""
    from botocore.exceptions import ClientError
    path = Path(path)
    sha = file_sha256(path)
    size = path.stat().st_size
    try:
        head = client.head_object(Bucket=BUCKET, Key=key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in {"404", "NoSuchKey", "NotFound"}:
            raise
        head = None
    if head is not None:
        if head["ContentLength"] != size or head.get("Metadata", {}).get("sha256") != sha:
            raise ValueError(f"Refusing to replace a different R2 artifact: {key}")
        return {"key": key, "sha256": sha, "size_bytes": size, "etag": head["ETag"]}
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    with path.open("rb") as stream:
        client.put_object(Bucket=BUCKET, Key=key, Body=stream,
                          ContentLength=size, ContentMD5=base64.b64encode(digest.digest()).decode(),
                          ContentType=content_type, IfNoneMatch="*",
                          Metadata={**(metadata or {}), "sha256": sha})
    head = client.head_object(Bucket=BUCKET, Key=key)
    if (head["ContentLength"] != size or head.get("Metadata", {}).get("sha256") != sha
            or head["ETag"].strip('"') != digest.hexdigest()):
        raise ValueError(f"R2 artifact verification failed: {key}")
    return {"key": key, "sha256": sha, "size_bytes": size, "etag": head["ETag"]}


def publish_features(client, artifact_dir, recipe_id):
    artifact_dir = Path(artifact_dir)
    record = load_feature_artifact(artifact_dir)
    prefix = f"{FEATURE_PREFIX}/{recipe_id}/{record['dataset']}/{record['video_id']}"
    uploaded = []
    for name in ("features", "timestamps", "support_start", "support_end"):
        uploaded.append(put_verified_file(client, record[name + "_path"], prefix + "/" + name + ".npy"))
    # The manifest is the publication marker, written only after every array.
    uploaded.append(put_verified_file(client, artifact_dir / "metadata.json", prefix + "/metadata.json",
                                       content_type="application/json"))
    return {"dataset": record["dataset"], "video_id": record["video_id"],
            "metadata_path": str(artifact_dir / "metadata.json"), "r2_prefix": prefix,
            "frame_count": record["frame_count"], "artifacts": uploaded}


def restore_features(client, destination, recipe_id, dataset, video_id):
    """Restore a previously verified cloud cache; never download raw video locally."""
    from botocore.exceptions import ClientError
    destination = Path(destination)
    if destination.exists():
        return load_feature_artifact(destination)
    prefix = f"{FEATURE_PREFIX}/{recipe_id}/{dataset}/{video_id}"
    try:
        metadata = read_json_object(client, prefix + "/metadata.json")
    except ClientError as exc:
        if exc.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    if metadata.get("dataset") != dataset or metadata.get("video_id") != video_id:
        raise ValueError("Remote feature identity mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    temporary = Path(tempfile.mkdtemp(prefix=".restore-", dir=destination.parent))
    try:
        for name in ("features", "timestamps", "support_start", "support_end"):
            client.download_file(BUCKET, prefix + "/" + name + ".npy", str(temporary / (name + ".npy")))
        write_json(temporary / "metadata.json", metadata)
        load_feature_artifact(temporary)
        temporary.rename(destination)
        return load_feature_artifact(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def ensure_model_assets(cache_dir):
    """Fetch the user-selected author assets once on persistent cloud storage."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_dir = cache_dir / EGOPER_COMMIT
    source_dir.mkdir(exist_ok=True)
    for filename, expected_blob in SOURCE_GIT_BLOBS.items():
        path = source_dir / filename
        if not path.exists():
            url = (f"https://raw.githubusercontent.com/robert80203/EgoPER_official/{EGOPER_COMMIT}/"
                   f"I3D_extractor/src/feature_extractor/models/threed_models/{filename}")
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read()
            actual = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            if actual != expected_blob:
                raise ValueError(f"Pinned upstream source mismatch: {filename}")
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
    checkpoint = cache_dir / CHECKPOINT_FILENAME
    assets_path = cache_dir / "assets.json"
    if not checkpoint.exists():
        import gdown
        temporary = checkpoint.with_suffix(checkpoint.suffix + ".download")
        result = gdown.download(id=CHECKPOINT_FILE_ID, output=str(temporary), quiet=True)
        if result is None or not temporary.exists() or temporary.stat().st_size < 10_000_000:
            raise ValueError("The selected author checkpoint could not be downloaded")
        temporary.replace(checkpoint)
    sha = file_sha256(checkpoint)
    if assets_path.exists():
        previous = json.loads(assets_path.read_text())
        if previous["checkpoint_sha256"] != sha:
            raise ValueError("Cached extractor checkpoint changed")
    assets = {"source_dir": str(source_dir), "upstream_commit": EGOPER_COMMIT,
              "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha,
              "checkpoint_bytes": checkpoint.stat().st_size,
              "recipe_id": EGOPER_COMMIT[:12] + "-" + sha[:12] + "-fp16-v1"}
    write_json(assets_path, assets)
    return assets
