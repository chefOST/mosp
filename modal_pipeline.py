"""Detached L4 pipeline: R2 videos -> versioned features -> joint training -> scoring.

All media and extractor weights stay on Modal/R2. Local execution only packages
code and existing annotation files and submits the pinned run configuration.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import modal

APP_NAME = "chefost-mosp-training"
VOLUME_NAME = "chefost-mosp-training"
SECRET_NAME = "chefost-mosp-training-20261116"
WANDB_ENTITY = "chefOST"
WANDB_PROJECT = "MOSP - full evals"
WORK = Path("/work")
ANNOTATION_MOUNT = Path("/opt/chefost-annotations")
DATASETS = ("CMU", "EGTEA")
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME)
secrets = [modal.Secret.from_name(SECRET_NAME)]


def annotation_relative_paths(dataset):
    stem = dataset.lower()
    return [
        f"annotations/pseudo_labels/{dataset}/pseudo_annotations_poc_new_{stem}.json",
        f"annotations/ground_truth/{dataset}/gt_annotations_{stem}.json",
        f"annotations/ground_truth/{dataset}/gt_annotations_{stem}_idx.json",
        f"vocabulary/{dataset}/files_dict.json", f"vocabulary/{dataset}/state_dict.json",
        f"vocabulary/{dataset}/action_dict.json",
        f"vocabulary/{dataset}/{'obj_dict.json' if dataset == 'EGTEA' else 'object_dict.json'}",
    ]


image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
         .pip_install("torch==2.6.0", "numpy==2.2.6", "Pillow==11.3.0",
                      "opencv-python-headless==4.11.0.86", "gdown==5.2.0",
                      "boto3==1.43.126", "wandb==0.30.0")
         .env({"PYTHONUNBUFFERED": "1", "WANDB_SILENT": "true"})
         .add_local_python_source("mosp", copy=True))
coordinator_image = image
if modal.is_local():
    local_annotations = Path(__file__).resolve().parent.parent / "chefOST-data/MOSCATO"
    for dataset in DATASETS:
        for relative in annotation_relative_paths(dataset):
            coordinator_image = coordinator_image.add_local_file(
                local_annotations / relative, str(ANNOTATION_MOUNT / relative))


def _safe_group(group):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,100}", group):
        raise ValueError("Run group must be a short path-safe identifier")
    return group


def _run_root(config):
    return WORK / "runs" / _safe_group(config["group"])


def _wandb_run(config, stage, dataset=None):
    import wandb
    identity = f"{config['group']}:{stage}:{dataset or 'all'}"
    run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT,
                     group=config["group"], job_type=stage,
                     name=f"{dataset or 'Pipeline'} / {stage}",
                     id=hashlib.sha256(identity.encode()).hexdigest()[:12], resume="allow",
                     config={**config, "dataset": dataset},
                     tags=["MOSCATO", "L4", "curated-split", "no-validation"],
                     settings=wandb.Settings(x_disable_stats=True, x_disable_meta=True,
                                             x_disable_machine_info=True))
    print(json.dumps({"event": "wandb_run", "stage": stage, "dataset": dataset, "url": run.url}), flush=True)
    return run


def _stage_annotations(config):
    from mosp.features import file_sha256
    from mosp.runtime import write_json
    destination = WORK / "annotations" / config["annotation_revision"]
    for relative, expected_sha in config["annotation_sha256"].items():
        source, target = ANNOTATION_MOUNT / relative, destination / relative
        if target.exists():
            if file_sha256(target) != expected_sha:
                raise ValueError(f"Cloud annotation cache changed: {relative}")
            continue
        if file_sha256(source) != expected_sha:
            raise ValueError(f"Packaged annotation mismatch: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(target)
    write_json(destination / "snapshot.json", {
        "revision": config["annotation_revision"], "sha256": config["annotation_sha256"]})
    return destination


_MODEL_CACHE = None


@app.function(image=image, gpu="L4", cpu=4, memory=16384,
              volumes={"/work": volume}, secrets=secrets, max_containers=4,
              scaledown_window=60, timeout=86400, startup_timeout=1200, retries=2)
def extract_one(row, assets, config):
    global _MODEL_CACHE
    from mosp.cloud_io import download_verified_video, publish_features, restore_features, s3_client
    from mosp.features import FeatureConfig, extract_video_features, load_egoper_model

    volume.reload()
    client = s3_client()
    destination = WORK / "features" / assets["recipe_id"] / row["dataset"] / row["video_id"]
    restored = restore_features(client, destination, assets["recipe_id"], row["dataset"], row["video_id"])
    if restored is not None:
        if (restored.get("source") != row["source"]
                or restored.get("annotation_revision") != config["annotation_revision"]
                or restored["feature_provenance"]["model_assets"]["checkpoint_sha256"] != assets["checkpoint_sha256"]):
            raise ValueError("Cached feature artifact identity/provenance mismatch")
        result = publish_features(client, destination, assets["recipe_id"])
        volume.commit()
        return result
    if _MODEL_CACHE is None:
        _MODEL_CACHE = load_egoper_model(assets["checkpoint_path"], assets["source_dir"],
                                         device="cuda", mixed_precision=True,
                                         expected_checkpoint_sha256=assets["checkpoint_sha256"])
        print(json.dumps({"event": "extractor_loaded", "checkpoint_sha256": assets["checkpoint_sha256"],
                          "gpu": "L4", "model": "EgoPER I3D-ResNet-50 Kinetics-400"}), flush=True)
    if _MODEL_CACHE.provenance["checkpoint_sha256"] != assets["checkpoint_sha256"]:
        raise ValueError("Warm worker loaded a different checkpoint")
    print(json.dumps({"event": "extracting", "dataset": row["dataset"], "video_id": row["video_id"]}), flush=True)
    with tempfile.TemporaryDirectory(prefix="mosp-video-") as temporary:
        video = download_verified_video(client, row, Path(temporary) / (row["video_id"] + ".mp4"))
        extract_video_features(video, destination, model=_MODEL_CACHE,
                               config=FeatureConfig(batch_size=config["feature_batch_size"]),
                               provenance={"dataset": row["dataset"], "video_id": row["video_id"],
                                           "source": row["source"],
                                           "annotation_revision": config["annotation_revision"]})
        # Persist the complete feature cache before publishing immutable R2 keys.
        volume.commit()
        result = publish_features(client, destination, assets["recipe_id"])
    volume.commit()
    print(json.dumps({"event": "features_complete", "dataset": row["dataset"],
                      "video_id": row["video_id"], "samples": result["frame_count"]}), flush=True)
    return result


@app.function(image=image, gpu="L4", cpu=4, memory=32768, volumes={"/work": volume},
              secrets=secrets, max_containers=2, timeout=86400, startup_timeout=1200, retries=2)
def train_dataset(dataset, manifest_path, config):
    from mosp import train as training
    from mosp.runtime import write_json
    volume.reload()
    # A durable claim also covers a coordinator crash between spawn and saving
    # its call ID. Never run two optimizers against one checkpoint directory.
    claims = modal.Dict.from_name("chefost-mosp-training-claims", create_if_missing=True)
    call_id = modal.current_function_call_id()
    if not call_id:
        raise RuntimeError("Training must execute as a remote Modal call")
    signature = hashlib.sha256(json.dumps({"dataset": dataset, "config": config,
                                           "manifest": Path(manifest_path).read_text()},
                                          sort_keys=True).encode()).hexdigest()
    claim_key = config["group"] + "/" + dataset
    claim = {"call_id": call_id, "signature": signature}
    if not claims.put(claim_key, claim, skip_if_exists=True):
        previous = claims[claim_key]
        if previous["signature"] != signature:
            raise ValueError("Training group is already claimed by different inputs")
        if previous["call_id"] != call_id:
            return modal.FunctionCall.from_id(previous["call_id"]).get()
    output = _run_root(config) / dataset / "training"
    args = training.parser().parse_args([
        "--manifest", manifest_path, "--out", str(output), "--epochs", str(config["epochs"]),
        "--beta", str(config["beta"][dataset]), "--device", "cuda", "--threads", "4",
        "--checkpoint-every", "10", "--seed", str(config["seed"]),
        "--wandb-project", WANDB_PROJECT, "--wandb-entity", WANDB_ENTITY,
        "--wandb-group", config["group"], "--wandb-job-type", "train",
        "--wandb-name", f"{dataset} / train",
    ] + (["--resume"] if (output / "last.pt").exists() else []))
    print(json.dumps({"event": "training_start", "dataset": dataset, "epochs": config["epochs"],
                      "resume": args.resume, "gpu": "L4"}), flush=True)
    training.train(args, checkpoint_callback=lambda _: volume.commit())
    result = {"dataset": dataset, "checkpoint_path": str(output / "last.pt"),
              "run_metadata": json.loads((output / "run.json").read_text())}
    write_json(_run_root(config) / dataset / "training-complete.json", result)
    volume.commit()
    return {"dataset": dataset, "checkpoint_path": result["checkpoint_path"],
            "wandb_url": result["run_metadata"]["wandb"]["url"]}


@app.function(image=image, gpu="L4", cpu=4, memory=32768, volumes={"/work": volume},
              secrets=secrets, max_containers=2, timeout=86400, startup_timeout=1200, retries=1)
def evaluate_dataset(dataset, checkpoint, inventory, feature_manifest, config):
    import wandb
    from mosp.evaluate import evaluate_checkpoint
    from mosp.runtime import write_json
    volume.reload()
    output = _run_root(config) / dataset / "evaluation"
    run = _wandb_run(config, "evaluate", dataset)
    succeeded = False
    try:
        run.summary["phase"] = "scoring held-out predictions"
        if (output / "metrics.json").exists():
            metrics = json.loads((output / "metrics.json").read_text())
        else:
            metrics = evaluate_checkpoint(checkpoint, WORK / "annotations" / config["annotation_revision"],
                                          dataset, inventory, feature_manifest, output, device="cuda")
        run.summary["metrics"] = metrics
        artifact = wandb.Artifact(f"{config['group']}-{dataset.lower()}-predictions", type="predictions",
                                  metadata={"dataset": dataset, "scope": "current curated held-out GT; sampled feature anchors"})
        artifact.add_dir(str(output))
        run.log_artifact(artifact, aliases=["final"]).wait()
        result = {"dataset": dataset, "metrics": metrics, "wandb_url": run.url}
        write_json(_run_root(config) / dataset / "evaluation-complete.json", result)
        volume.commit()
        run.summary["phase"] = "complete"
        succeeded = True
        return result
    except Exception as exc:
        run.summary["phase"] = "failed"
        run.summary["error"] = str(exc)
        raise
    finally:
        run.finish(exit_code=0 if succeeded else 1)


@app.function(image=image, cpu=4, memory=16384, volumes={"/work": volume}, secrets=secrets,
              max_containers=2, timeout=86400, retries=1)
def dataset_pipeline(dataset, inventory, assets, config):
    import wandb
    from mosp.cloud_data import prepare_training_manifest
    from mosp.data import validate_manifest
    from mosp.pipeline_state import attach_saved_call, completed_feature_index
    from mosp.runtime import write_json

    volume.reload()
    annotation_root = WORK / "annotations" / config["annotation_revision"]
    gt_ids = set(json.loads((annotation_root / f"annotations/ground_truth/{dataset}/gt_annotations_{dataset.lower()}.json").read_text()))
    meta = json.loads((annotation_root / f"vocabulary/{dataset}/files_dict.json").read_text())
    records = {r["video_id"]: r for r in inventory["videos"] if r["dataset"] == dataset}
    if set(records) != set(meta):
        raise ValueError(f"R2 and annotation video inventory differ for {dataset}")
    train_ids, test_ids = sorted(set(meta) - gt_ids), sorted(gt_ids)
    expected_count = {"CMU": 134, "EGTEA": 74}[dataset]
    if len(train_ids) != expected_count:
        raise ValueError(f"Curated split changed unexpectedly for {dataset}")
    root = _run_root(config) / dataset
    root.mkdir(parents=True, exist_ok=True)
    run = _wandb_run(config, "features", dataset)
    feature_rows = []
    training_call = None
    succeeded = False
    try:
        previous_error = run.summary.get("error")
        if previous_error:
            run.summary["recovery/previous_error"] = previous_error
        run.summary["error"] = None
        run.summary.update({"phase": "extracting training features", "training_videos": len(train_ids),
                            "test_videos": len(test_ids), "validation_videos": 0,
                            "paper_split_verified": False, "gpu_workers_shared_limit": 4})
        prepared = root / "prepared"
        manifest = prepared / "manifest.json"
        feature_path = root / "feature-index.json"
        cached_features = completed_feature_index(
            feature_path, manifest, root / "training/run.json", dataset=dataset,
            source_records=records, feature_root=WORK / "features" / assets["recipe_id"] / dataset,
            annotation_revision=config["annotation_revision"], checkpoint_sha256=assets["checkpoint_sha256"])
        if cached_features is not None:
            all_features = cached_features
            run.summary["phase"] = "reusing verified features; reattaching training"
            run.log({"features/train_complete": len(train_ids), "features/train_total": len(train_ids),
                     "features/test_complete": len(test_ids), "features/test_total": len(test_ids)})
        else:
            for result in extract_one.map([records[v] for v in train_ids],
                                          kwargs={"assets": assets, "config": config}, order_outputs=False):
                feature_rows.append(result)
                run.log({"features/train_complete": len(feature_rows), "features/train_total": len(train_ids)})
            volume.reload()
            train_features = {"videos": sorted(feature_rows, key=lambda x: x["video_id"])}
            write_json(root / "train-feature-index.json", train_features)
            if manifest.exists():
                validate_manifest(manifest, check_features=True)
            else:
                if prepared.exists():
                    shutil.rmtree(prepared)  # Only incomplete, generated preparation for this run.
                manifest = prepare_training_manifest(annotation_root, dataset, config["annotation_revision"],
                                                      inventory, train_features, prepared,
                                                      annotation_sha256=config["annotation_sha256"],
                                                      action_segment_end="exclusive")
            volume.commit()
        run.summary["phase"] = "training started; extracting held-out features"
        call_path = root / "training-call.json"
        if call_path.exists():
            training_call, training_call_id = attach_saved_call(call_path, modal.FunctionCall.from_id)
        else:
            if cached_features is not None:
                raise ValueError("Completed features lack their training handoff; refusing duplicate training")
            training_call = train_dataset.spawn(dataset, str(manifest), config)
            training_call_id = training_call.object_id
            write_json(call_path, {"call_id": training_call_id})
            volume.commit()
        run.summary["training_call_id"] = training_call_id
        # Held-out extraction does not consult labels and overlaps GPU training.
        if cached_features is None:
            test_rows = []
            for result in extract_one.map([records[v] for v in test_ids],
                                          kwargs={"assets": assets, "config": config}, order_outputs=False):
                test_rows.append(result)
                run.log({"features/test_complete": len(test_rows), "features/test_total": len(test_ids)})
            volume.reload()
            all_features = {"videos": sorted(feature_rows + test_rows, key=lambda x: x["video_id"])}
            write_json(feature_path, all_features)
            volume.commit()
        artifact = wandb.Artifact(f"{config['group']}-{dataset.lower()}-inputs", type="dataset",
                                  metadata={"features_store": "Cloudflare R2", "recipe_id": assets["recipe_id"],
                                            "annotation_revision": config["annotation_revision"]})
        artifact.add_file(str(feature_path), name="feature-index.json")
        artifact.add_file(str(manifest), name="training-manifest.json")
        run.log_artifact(artifact, aliases=["verified"]).wait()
        run.summary["phase"] = "features complete"
        succeeded = True
    except Exception as exc:
        run.summary["phase"] = "failed"
        run.summary["error"] = str(exc)
        raise
    finally:
        run.finish(exit_code=0 if succeeded else 1)
    result = training_call.get()
    volume.reload()
    completion = root / "evaluation-complete.json"
    if completion.exists():
        from mosp.runtime import file_sha256
        evaluation = json.loads(completion.read_text())
        if (evaluation.get("dataset") != dataset
                or evaluation["metrics"]["checkpoint_sha256"] != file_sha256(result["checkpoint_path"])):
            raise ValueError("Completed evaluation belongs to a different checkpoint")
    else:
        evaluation = evaluate_dataset.remote(dataset, result["checkpoint_path"], inventory, all_features, config)
    return {"training": result, "evaluation": evaluation}


@app.function(image=coordinator_image, cpu=2, memory=8192, volumes={"/work": volume},
              secrets=secrets, max_containers=1, timeout=86400, retries=0)
def orchestrate(config, recovery_revision=None):
    from mosp.cloud_io import ensure_model_assets, s3_client, video_inventory
    from mosp.features import load_egoper_model
    from mosp.runtime import write_json
    _safe_group(config["group"])
    if config["wandb_project"] != WANDB_PROJECT or config["gpu"] != "L4":
        raise ValueError("Unexpected W&B destination or GPU")
    volume.reload()
    root = _run_root(config)
    root.mkdir(parents=True, exist_ok=True)
    run = _wandb_run(config, "pipeline")
    succeeded = False
    try:
        previous_error = run.summary.get("error")
        if previous_error:
            run.summary["recovery/previous_error"] = previous_error
        run.summary["error"] = None
        if recovery_revision:
            run.summary["recovery/controller_revision"] = recovery_revision
            write_json(root / f"recovery-{recovery_revision[:12]}.json", {
                "controller_revision": recovery_revision,
                "original_configuration_preserved": True,
                "function_call_id": modal.current_function_call_id()})
        run.summary["phase"] = "preparing pinned annotations and extractor"
        configuration_path = root / "configuration.json"
        if configuration_path.exists() and json.loads(configuration_path.read_text()) != config:
            raise ValueError("Refusing to reuse a run group with different configuration")
        write_json(configuration_path, config)
        _stage_annotations(config)
        assets = ensure_model_assets(WORK / "assets" / "egoper-i3d-r50")
        # Validate serialization and exact source/checkpoint architecture before
        # spending GPU time. This is a correctness check, not a GPU benchmark.
        probe_model = load_egoper_model(assets["checkpoint_path"], assets["source_dir"],
                                         device="cpu", mixed_precision=False,
                                         expected_checkpoint_sha256=assets["checkpoint_sha256"])
        del probe_model
        assets["recipe_id"] += "-data-" + config["annotation_revision"][:8]
        inventory = video_inventory(s3_client())
        counts = {ds: sum(r["dataset"] == ds for r in inventory["videos"]) for ds in DATASETS}
        if counts != {"CMU": 163, "EGTEA": 86}:
            raise ValueError(f"Unexpected source inventory: {counts}")
        write_json(root / "source-inventory.json", inventory)
        write_json(root / "extractor-assets.json", assets)
        volume.commit()
        run.config.update({"resolved_extractor": assets, "video_counts": counts})
        run.summary["phase"] = "parallel feature extraction and per-dataset training"
        calls = {ds: dataset_pipeline.spawn(ds, inventory, assets, config) for ds in DATASETS}
        write_json(root / "dataset-calls.json", {ds: call.object_id for ds, call in calls.items()})
        volume.commit()
        result = {ds: call.get() for ds, call in calls.items()}
        volume.reload()
        write_json(root / "complete.json", result)
        volume.commit()
        run.summary["phase"] = "complete"
        run.summary["results"] = result
        succeeded = True
        return result
    except Exception as exc:
        run.summary["phase"] = "failed"
        run.summary["error"] = str(exc)
        write_json(root / "failure.json", {"error": str(exc)})
        volume.commit()
        raise
    finally:
        run.finish(exit_code=0 if succeeded else 1)


def build_run_config(group=None, epochs=100):
    """Local metadata-only launch preparation; never opens a video/weight file."""
    from mosp.features import file_sha256
    repository = Path(__file__).resolve().parent
    annotations = repository.parent / "chefOST-data/MOSCATO"
    annotation_revision = subprocess.check_output(
        ["git", "-C", str(annotations.parent), "rev-parse", "HEAD"], text=True).strip()
    code_revision = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    hashes = {}
    for dataset in DATASETS:
        for relative in annotation_relative_paths(dataset):
            path = annotations / relative
            with path.open("rb") as stream:
                if stream.read(50).startswith(b"version https://git-lfs"):
                    raise ValueError(f"Annotation LFS object missing: {relative}")
            hashes[relative] = file_sha256(path)
    if group is None:
        group = "moscato-i3d-r50-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M")
    _safe_group(group)
    if epochs < 1:
        raise ValueError("Epoch budget must be positive")
    return {"group": group, "epochs": epochs, "seed": 0, "gpu": "L4",
            "feature_batch_size": 4, "annotation_revision": annotation_revision,
            "annotation_sha256": hashes, "code_revision": code_revision,
            "wandb_entity": WANDB_ENTITY, "wandb_project": WANDB_PROJECT,
            "beta": {"CMU": 0.15, "EGTEA": 0.5},
            "beta_provenance": {"CMU": "maximum F1@0.50 shown in published supplementary ablation; not a new sweep",
                                "EGTEA": "fixed reproduction assumption; paper does not specify its value"},
            "epochs_provenance": f"fixed {epochs}-epoch reproduction budget; paper does not state an epoch count",
            "validation_policy": "none; inference from authors' documented train/test-only protocol",
            "split_scope": "current curated pseudo-label pool excluding all manual-GT videos; official paper split IDs unverified",
            "extractor_scope": "user-selected EgoPER I3D checkpoint/recipe; exact MOSCATO extractor unspecified"}


@app.local_entrypoint()
def main(group: str = "", epochs: int = 100):
    config = build_run_config(group or None, epochs)
    call = orchestrate.spawn(config)
    print(json.dumps({"group": config["group"], "function_call_id": call.object_id,
                      "wandb_project": WANDB_PROJECT}), flush=True)
