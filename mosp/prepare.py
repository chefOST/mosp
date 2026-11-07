"""Read-only release audit and explicit training-manifest validation.

No inferred split or temporal mapping is written. JSON goes to stdout unless an
output path is explicitly supplied. Reserved split IDs are checked only for
separation; no reserved targets or scores are produced. Existing outputs are
never overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .data import SCHEMA_VERSION, validate_manifest


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected annotation JSON object: {path}")
    return value


def audit_release(root: str | Path) -> dict[str, Any]:
    """Inspect actual filenames, available IDs and leakage hazards, without writes."""
    root = Path(root).expanduser().resolve()
    if not (root / "annotations").is_dir():
        if (root / "MOSCATO" / "annotations").is_dir():
            root = root / "MOSCATO"
        else:
            raise FileNotFoundError(f"No annotations/ directory beneath {root}")
    report: dict[str, Any] = {
        "audit_schema_version": SCHEMA_VERSION,
        "root": str(root),
        "paper_split_verified": False,
        "note": "Inventory only. No train split or timeline alignment has been inferred.",
        "datasets": {},
    }
    for dataset in ("CMU", "EGTEA", "EgoPER/coffee", "EgoPER/oatmeal", "EgoPER/pinwheels", "EgoPER/quesadilla", "EgoPER/tea"):
        folder = root / "annotations" / "ground_truth" / dataset
        pseudo_folder = root / "annotations" / "pseudo_labels" / dataset
        if not folder.is_dir() and not pseudo_folder.is_dir():
            continue
        stem = dataset.split("/")[-1].lower()
        def choose(directory: Path, names: tuple[str, ...]) -> Path | None:
            found = [directory / name for name in names if (directory / name).is_file()]
            if len(found) > 1:
                raise ValueError(f"Ambiguous annotation sources; inspect explicitly: {found}")
            return found[0] if found else None
        gt_path = choose(folder, (f"gt_annotations_{stem}.json", "ground_truth_annotations.json"))
        pseudo_path = choose(pseudo_folder, (f"pseudo_annotations_poc_new_{stem}.json", "pseudo_annotations.json"))
        gt_ids = set(_read(gt_path)) if gt_path else set()
        pseudo_ids = set(_read(pseudo_path)) if pseudo_path else set()
        vocab = root / "vocabulary" / dataset
        entry: dict[str, Any] = {
            "ground_truth_path": str(gt_path) if gt_path else None,
            "pseudo_labels_path": str(pseudo_path) if pseudo_path else None,
            "ground_truth_videos": len(gt_ids),
            "pseudo_label_videos": len(pseudo_ids),
            "overlap_count": len(gt_ids & pseudo_ids),
            "overlap_ids": sorted(gt_ids & pseudo_ids),
            "ground_truth_ids": sorted(gt_ids),
            "pseudo_label_ids": sorted(pseudo_ids),
            "vocabulary_files": sorted(p.name for p in vocab.glob("*.json")),
            "warnings": [],
        }
        if gt_ids & pseudo_ids:
            entry["warnings"].append("Pseudo-label files contain manually annotated evaluation videos; they are not a training split.")
        metadata_path = vocab / "files_dict.json"
        if metadata_path.exists():
            records = _read(metadata_path)
            missing = []
            lengths = {}
            for video_id, record in records.items():
                feature = record.get("feature_path")
                resolved = Path(feature) if isinstance(feature, str) else None
                if resolved is not None and not resolved.is_absolute():
                    resolved = root / resolved
                if resolved is None or not resolved.is_file():
                    missing.append(video_id)
                lengths[video_id] = record.get("feature_len")
            entry["metadata_path"] = str(metadata_path)
            entry["metadata_videos"] = len(records)
            entry["missing_feature_path_count"] = len(missing)
            entry["missing_feature_path_ids"] = sorted(missing)
            entry["metadata_feature_lengths"] = lengths
            if missing:
                entry["warnings"].append("Metadata feature paths are missing locally; supply explicit verified feature paths and index mappings.")
        report["datasets"][dataset] = entry
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Read-only audit of a released MOSCATO directory")
    audit.add_argument("root", type=Path)
    audit.add_argument("--output", type=Path, help="Explicit new output file; default prints JSON")
    validate = commands.add_parser("validate", help="Check version-1 training inputs and reserved split IDs")
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--metadata-only", action="store_true", help="Skip training-array loading, not split or path checks")
    validate.add_argument("--output", type=Path, help="Explicit new output file; default prints JSON")
    args = parser.parse_args(argv)
    try:
        result = audit_release(args.root) if args.command == "audit" else validate_manifest(args.manifest, check_features=not args.metadata_only)
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
        else:
            print(rendered, end="")
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
