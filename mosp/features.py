"""Timestamped RGB features using the explicitly selected EgoPER I3D recipe.

This is an adaptation assumption, not evidence of the unpublished MOSCATO
extractor. Native video files are read without modification. Downloads belong
to the caller's cloud orchestration; this module never downloads models/media.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import types
from typing import Any, Callable, Iterable, Iterator

import numpy as np


EGOPER_COMMIT = "7740723201f8698cb888605eefe53bb29d28bb7a"
CHECKPOINT_FILE_ID = "1SF4NduQ7w08wP00IgftZjnRqRYRdppd6"
CHECKPOINT_FILENAME = "kinetics400-rgb-i3d-resnet-50-f32-s2-precise_bn-warmupcosine-bs1024-e196.pth.tar"
SOURCE_GIT_BLOBS = {
    "i3d_resnet.py": "d3c7531573a3c53ea18e6f8bec997f7b19544154",
    "inflate_from_2d_model.py": "3eb3afea1ab0cde057ee19e4570d6f247f925fd8",
}
FEATURE_DIM = 2048
INPUT_SIZE = 256
CONTEXT_SIZE = 64
CONTEXT_INDICES = np.arange(0, CONTEXT_SIZE, 2, dtype=np.int64)
RGB_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
RGB_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


@dataclass(frozen=True)
class FeatureConfig:
    sample_fps: float = 10.0
    batch_size: int = 4

    def validate(self) -> None:
        if not math.isfinite(self.sample_fps) or self.sample_fps != 10.0:
            raise ValueError("The selected EgoPER recipe requires an explicit 10 Hz frame grid")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_egoper_model(
    checkpoint_path: str | Path,
    source_dir: str | Path,
    *,
    device: str = "cuda",
    mixed_precision: bool = True,
    expected_checkpoint_sha256: str | None = None,
) -> "TorchFeatureModel":
    """Load caller-cached assets, checking both upstream source blob identities.

    ``source_dir`` can be the pinned repository root or its threed_models folder.
    Constructing the class directly avoids its factory's ImageNet download.
    """
    import torch

    checkpoint_path = Path(checkpoint_path).resolve(strict=True)
    source_dir = Path(source_dir).resolve(strict=True)
    model_dir = source_dir / "I3D_extractor/src/feature_extractor/models/threed_models"
    if not model_dir.is_dir():
        model_dir = source_dir
    source_hashes = {}
    for filename, expected_blob in SOURCE_GIT_BLOBS.items():
        content = (model_dir / filename).read_bytes()
        git_blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if git_blob != expected_blob:
            raise ValueError(f"Upstream source is not the pinned EgoPER revision: {filename}")
        source_hashes[filename] = hashlib.sha256(content).hexdigest()
    checkpoint_sha = file_sha256(checkpoint_path)
    if expected_checkpoint_sha256 is not None and checkpoint_sha != expected_checkpoint_sha256:
        raise ValueError("Checkpoint SHA-256 does not match the caller's pinned asset")

    package_name = "_mosp_egoper_" + EGOPER_COMMIT
    module_name = package_name + ".i3d_resnet"
    if module_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(model_dir)]
        sys.modules[package_name] = package
        spec = importlib.util.spec_from_file_location(module_name, model_dir / "i3d_resnet.py")
        if spec is None or spec.loader is None:
            raise ImportError("Unable to load pinned I3D implementation")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    model_type = sys.modules[module_name].I3D_ResNet
    model = model_type(50, num_classes=400, dropout=0.5, without_t_stride=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("Expected the EgoPER checkpoint's state_dict")
    state = checkpoint["state_dict"]
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    provenance = {
        "upstream_repository": "https://github.com/robert80203/EgoPER_official",
        "upstream_commit": EGOPER_COMMIT,
        "source_sha256": source_hashes,
        "checkpoint_filename": CHECKPOINT_FILENAME,
        "checkpoint_file_id": CHECKPOINT_FILE_ID,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "architecture": "RGB I3D-ResNet-50, Kinetics-400, without_t_stride",
        "pooling": "spatial mean, then temporal maximum; classifier bypassed",
        "mixed_precision": bool(mixed_precision and torch.device(device).type == "cuda"),
        "torch_version": torch.__version__,
    }
    return TorchFeatureModel(model, device, mixed_precision, provenance)


class TorchFeatureModel:
    def __init__(self, model: Any, device: str, mixed_precision: bool, provenance: dict[str, Any]):
        self.model = model
        self.device = device
        self.mixed_precision = mixed_precision
        self.provenance = provenance

    def __call__(self, clips: np.ndarray) -> np.ndarray:
        import torch

        device = torch.device(self.device)
        tensor = torch.from_numpy(np.ascontiguousarray(clips)).to(device=device, dtype=torch.float32)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16,
            enabled=self.mixed_precision and device.type == "cuda",
        ):
            output = self.model(tensor)
        return output.float().cpu().numpy()


def preprocess_rgb(frame: np.ndarray) -> np.ndarray:
    """Match EgoPER's OpenCV short-side resize and torchvision center crop."""
    import cv2

    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("Expected a uint8 H×W×3 RGB frame")
    height, width = frame.shape[:2]
    if min(height, width) < 1:
        raise ValueError("Empty RGB frame")
    if height <= width:
        resized_h, resized_w = INPUT_SIZE, int(INPUT_SIZE * width / height)
    else:
        resized_h, resized_w = int(INPUT_SIZE * height / width), INPUT_SIZE
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    top = int(round((resized_h - INPUT_SIZE) / 2.0))
    left = int(round((resized_w - INPUT_SIZE) / 2.0))
    crop = resized[top:top + INPUT_SIZE, left:left + INPUT_SIZE]
    tensor = crop.transpose(2, 0, 1).astype(np.float32) / np.float32(255)
    return np.ascontiguousarray((tensor - RGB_MEAN) / RGB_STD)


class TemporalWindow:
    """An equivalent ring-buffer implementation of upstream roll(-1)/[::2]."""
    def __init__(self) -> None:
        self.buffer: np.ndarray | None = None
        self.index = 0

    def append(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32)
        if frame.ndim != 3 or frame.shape[0] != 3:
            raise ValueError("Expected a C×H×W frame with three channels")
        if self.buffer is None:
            self.buffer = np.zeros((CONTEXT_SIZE, *frame.shape), dtype=np.float32)
        if frame.shape != self.buffer.shape[1:]:
            raise ValueError("Frame shape changed within a video")
        slot = self.index % CONTEXT_SIZE
        self.buffer[slot] = frame
        selected = self.buffer[(slot + 1 + CONTEXT_INDICES) % CONTEXT_SIZE]
        self.index += 1
        return selected.transpose(1, 0, 2, 3)  # C,T,H,W; independent copy


def feature_batches(
    frames: Iterable[np.ndarray],
    model: Callable[[np.ndarray], np.ndarray],
    config: FeatureConfig,
) -> Iterator[np.ndarray]:
    """Stream native-size sampled RGB frames through the exact temporal recipe."""
    config.validate()
    window, pending = TemporalWindow(), []
    for frame in frames:
        pending.append(window.append(preprocess_rgb(frame)))
        if len(pending) == config.batch_size:
            yield _predict(model, pending)
            pending = []
    if pending:
        yield _predict(model, pending)


def _predict(model: Callable, clips: list[np.ndarray]) -> np.ndarray:
    result = np.asarray(model(np.stack(clips)), dtype=np.float32)
    if result.shape != (len(clips), FEATURE_DIM) or not np.isfinite(result).all():
        raise ValueError("I3D must return finite B×2048 features")
    return result


def probe_video(path: str | Path) -> dict[str, Any]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams",
         "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True, timeout=120,
    )
    probe = json.loads(result.stdout)
    if not probe.get("streams"):
        raise ValueError("Video has no readable video stream")
    return probe


def sampled_rgb_frames(path: str | Path, config: FeatureConfig, probe: dict[str, Any]) -> Iterator[np.ndarray]:
    """Decode a timestamp grid to a pipe; do not re-encode or save video/frames."""
    config.validate()
    stream = probe["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    rotation = float(stream.get("tags", {}).get("rotate", 0))
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            rotation = float(side_data["rotation"])
    quarter_turns = round(rotation / 90)
    if not math.isclose(rotation, quarter_turns * 90, abs_tol=0.01):
        raise ValueError("Unsupported non-right-angle video rotation")
    if quarter_turns % 2:
        width, height = height, width
    filter_spec = f"setpts=PTS-STARTPTS,fps=fps={config.sample_fps:g}:start_time=0:round=near:eof_action=round"
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-xerror", "-i", str(path),
               "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", filter_spec, "-vsync", "0",
               "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    errors: deque[str] = deque(maxlen=20)
    assert process.stdout is not None and process.stderr is not None

    def drain_errors() -> None:
        for line in iter(process.stderr.readline, b""):
            errors.append(line.decode(errors="replace").strip())

    reader = threading.Thread(target=drain_errors, daemon=True)
    reader.start()
    frame_bytes = width * height * 3
    try:
        while True:
            data = process.stdout.read(frame_bytes)
            if not data:
                break
            while len(data) < frame_bytes:
                remainder = process.stdout.read(frame_bytes - len(data))
                if not remainder:
                    raise ValueError("FFmpeg emitted a truncated raw frame")
                data += remainder
            yield np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        returncode = process.wait(timeout=120)
        reader.join(timeout=5)
        if returncode != 0:
            raise RuntimeError(f"FFmpeg failed ({returncode}): {'; '.join(errors)}")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        process.stdout.close()
        reader.join(timeout=5)
        process.stderr.close()


def support_timestamps(count: int, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Real input support only; output zero has no real frames and gets NaNs."""
    indices = np.arange(count, dtype=np.int64)
    first = indices - 63
    first = np.where(first < 0, np.mod(first, 2), first)
    starts, ends = first.astype(np.float64) / fps, (indices.astype(np.float64) - 1) / fps
    if count:
        starts[0] = ends[0] = np.nan
    return starts, ends


def load_feature_artifact(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve(strict=True)
    metadata_path = output_dir / "metadata.json"
    result = json.loads(metadata_path.read_text())
    if result.get("schema_version") != 1 or result.get("status") != "complete":
        raise ValueError("Incomplete or unsupported feature artifact")
    result = dict(result)
    for name in ("features", "timestamps", "support_start", "support_end"):
        path = (output_dir / result[name + "_path"]).resolve(strict=True)
        path.relative_to(output_dir)
        if file_sha256(path) != result["artifact_sha256"][name]:
            raise ValueError(f"Feature artifact checksum mismatch: {name}")
        result[name + "_path"] = str(path)
    result.update(metadata_path=str(metadata_path), artifact_dir=str(output_dir))
    return result


def extract_video_features(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    model: Callable[[np.ndarray], np.ndarray],
    config: FeatureConfig = FeatureConfig(),
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish a complete feature directory atomically; never replace a cache."""
    config.validate()
    video_path = Path(video_path).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace a feature artifact: {output_dir}")
    provenance = dict(provenance or {})
    forbidden = {"schema_version", "status", "artifact_sha256", "features_path", "timestamps_path",
                 "support_start_path", "support_end_path", "feature_provenance", "config"}
    if forbidden.intersection(provenance):
        raise ValueError("Caller provenance may not replace extractor-owned metadata")
    source_sha = file_sha256(video_path)
    source_size = video_path.stat().st_size
    declared_source = provenance.get("source", {})
    if declared_source.get("sha256", source_sha) != source_sha or declared_source.get("size_bytes", source_size) != source_size:
        raise ValueError("Input video does not match the declared source SHA-256/size")
    probe = probe_video(video_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.partial-", dir=output_dir.parent))
    try:
        raw_path = temporary / "features.raw"
        frames = sampled_rgb_frames(video_path, config, probe)
        count = 0
        try:
            with raw_path.open("wb") as stream:
                for batch in feature_batches(frames, model, config):
                    stream.write(batch.tobytes(order="C"))
                    count += len(batch)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            frames.close()
        if not count:
            raise ValueError("No sampled frames were decoded")
        raw = np.memmap(raw_path, dtype=np.float32, mode="r", shape=(count, FEATURE_DIM))
        features = np.lib.format.open_memmap(temporary / "features.npy", mode="w+", dtype=np.float32,
                                            shape=(count, FEATURE_DIM))
        for start in range(0, count, 2048):
            features[start:start + 2048] = raw[start:start + 2048]
        features.flush()
        del features, raw
        raw_path.unlink()
        timestamps = np.arange(count, dtype=np.float64) / config.sample_fps
        support_start, support_end = support_timestamps(count, config.sample_fps)
        np.save(temporary / "timestamps.npy", timestamps, allow_pickle=False)
        np.save(temporary / "support_start.npy", support_start, allow_pickle=False)
        np.save(temporary / "support_end.npy", support_end, allow_pickle=False)
        metadata = {
            **provenance, "schema_version": 1, "status": "complete", "config": asdict(config),
            "source": {**declared_source, "sha256": source_sha, "size_bytes": source_size},
            "frame_count": count,
            "feature_provenance": {
                "model": "EgoPER I3D-ResNet-50 Kinetics-400 RGB", "dimension": FEATURE_DIM, "layout": "TD",
                "timeline": "output anchors i/10 seconds relative to first video PTS; trailing support recorded separately",
                "upstream_commit": EGOPER_COMMIT,
                "recipe_is_moscato_author_confirmed": False,
                "context_grid_indices": "i-63, i-61, ..., i-1",
                "context_frames": 32, "context_buffer_slots": 64, "input_stride": 2, "output_stride": 1,
                "padding": "zeros after normalization; output zero has entirely padded input",
                "input_size": INPUT_SIZE, "resize": "shorter side 256, cv2.INTER_LINEAR",
                "crop": "center 256x256", "rgb_mean": RGB_MEAN[:, 0, 0].tolist(),
                "rgb_std": RGB_STD[:, 0, 0].tolist(),
                "model_assets": getattr(model, "provenance", {}),
            },
            "video_probe": probe,
            "native_video_modified": False,
            "decoder": {"name": "ffmpeg", "filter": "setpts=PTS-STARTPTS,fps=fps=10:start_time=0:round=near:eof_action=round",
                        "version": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]},
            "artifact_sha256": {},
        }
        for name in ("features", "timestamps", "support_start", "support_end"):
            metadata[name + "_path"] = name + ".npy"
            metadata["artifact_sha256"][name] = file_sha256(temporary / (name + ".npy"))
        with (temporary / "metadata.json").open("w") as stream:
            json.dump(metadata, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if output_dir.exists():
            raise FileExistsError(f"Feature output appeared while extracting: {output_dir}")
        temporary.rename(output_dir)
        return load_feature_artifact(output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
