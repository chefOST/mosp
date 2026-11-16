"""Prediction-only EGTEA job. No training, target labels, or score calculation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import modal

APP_NAME = "chefost-mosp-predictions"
GROUP = "CMU-EGTEA-I3D-R50-20261116"
ENTITY = "chefOST"
PROJECT = "MOSP - full evals"
WORK = Path("/work")
app = modal.App(APP_NAME)
volume = modal.Volume.from_name("chefost-mosp-training")
secret = modal.Secret.from_name("chefost-mosp-training-20261116")

# Reuse the established environment; no media or checkpoint is bundled locally.
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
         .pip_install("torch==2.6.0", "numpy==2.2.6", "Pillow==11.3.0",
                      "opencv-python-headless==4.11.0.86", "gdown==5.2.0",
                      "boto3==1.43.126", "wandb==0.30.0")
         .env({"PYTHONUNBUFFERED": "1", "WANDB_SILENT": "true"})
         .add_local_python_source("mosp", copy=True))


def _output_files(index):
    """Explicit membership prevents accidental metric-file directory uploads."""
    files = []
    for video_id, row in sorted(index["per_video"].items()):
        files.append((row["prediction_path"], f"predictions/{video_id}.npz"))
        files.append((row["receipt_path"], f"receipts/{video_id}.json"))
    return files


def _assert_no_scores(value):
    """Reject score data before it can enter the output run or artifact."""
    forbidden = {"metrics", "frame_micro", "precision", "recall", "f1", "f1_max", "f1_score", "tp", "fp", "fn"}
    if isinstance(value, dict):
        if any(str(key).lower() in forbidden for key in value):
            raise ValueError("Prediction output metadata contains score fields")
        for item in value.values():
            _assert_no_scores(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_scores(item)


@app.function(image=image, gpu="L4", cpu=4, memory=32768,
              volumes={"/work": volume}, secrets=[secret], max_containers=1,
              scaledown_window=2, timeout=7200, startup_timeout=1200, retries=1)
def predict_egtea(group=GROUP, code_revision=None):
    import wandb
    from mosp.batch_predict import batch_predict
    from mosp.runtime import file_sha256, write_json

    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,100}", group):
        raise ValueError("Invalid source run group")
    volume.reload()
    root = WORK / "runs" / group
    completed = json.loads((root / "EGTEA/training-complete.json").read_text())
    if completed["run_metadata"]["status"] != "completed" or completed["run_metadata"]["last_completed_epoch"] != 100:
        raise ValueError("The requested final EGTEA model is not complete")
    checkpoint = Path(completed["checkpoint_path"])
    checkpoint_sha = file_sha256(checkpoint)
    output = root / "EGTEA" / "inference_outputs"
    identity = f"{group}:predictions:EGTEA"
    run = wandb.init(entity=ENTITY, project=PROJECT, group=group, job_type="predictions",
                     name="EGTEA / saved outputs",
                     id=hashlib.sha256(identity.encode()).hexdigest()[:12], resume="allow",
                     config={"dataset": "EGTEA", "mode": "prediction_only", "checkpoint_epoch": 100,
                             "checkpoint_sha256": checkpoint_sha,
                             "model_training_run": completed["run_metadata"]["wandb"]["id"], "source_group": group,
                             "prediction_code_revision": code_revision,
                             "input_selection": "checkpoint's reserved test IDs; no label values loaded"},
                     settings=wandb.Settings(x_disable_stats=True, x_disable_meta=True,
                                             x_disable_machine_info=True))
    succeeded = False
    try:
        run.summary["phase"] = "predicting from cached features"
        run.summary["error"] = None
        print(json.dumps({"event": "prediction_run", "url": run.url,
                          "checkpoint_sha256": checkpoint_sha}), flush=True)

        def progress(record, count, total):
            volume.commit()
            run.log({"predictions/videos_complete": count, "predictions/videos_total": total})
            print(json.dumps({"event": "prediction_complete", "video_id": record["video_id"],
                              "complete": count, "total": total}), flush=True)

        index = batch_predict(checkpoint, root / "EGTEA/feature-index.json",
                              root / "source-inventory.json", output, device="cuda",
                              progress_callback=progress)
        if index["dataset"] != "EGTEA" or len(index["per_video"]) != 12:
            raise ValueError("Expected exactly the twelve reserved EGTEA videos")
        _assert_no_scores(index)
        volume.commit()
        artifact = wandb.Artifact(f"{group}-egtea-predictions", type="predictions",
                                  metadata={"dataset": "EGTEA", "checkpoint_sha256": checkpoint_sha,
                                            "checkpoint_epoch": 100, "prediction_videos": 12,
                                            "scope": "saved prediction probabilities and timestamps only"})
        for path, name in _output_files(index):
            if name.endswith('.json'):
                _assert_no_scores(json.loads(Path(path).read_text()))
            artifact.add_file(path, name=name)
        artifact.add_file(str(output / "index.json"), name="output_index.json")
        saved = run.log_artifact(artifact, aliases=["final", "latest", "outputs-only"]).wait()
        result = {"dataset": "EGTEA", "prediction_videos": 12,
                  "checkpoint_sha256": checkpoint_sha, "checkpoint_epoch": 100,
                  "output_index": str(output / "index.json"), "wandb_url": run.url,
                  "artifact": saved.source_name, "status": "complete"}
        write_json(root / "EGTEA/predictions-complete.json", result)
        volume.commit()
        run.summary.update({"phase": "complete", "prediction_videos": 12,
                            "outputs_artifact": saved.source_name})
        succeeded = True
        return result
    except Exception as exc:
        run.summary["phase"] = "failed"
        run.summary["error"] = str(exc)
        raise
    finally:
        run.finish(exit_code=0 if succeeded else 1)
