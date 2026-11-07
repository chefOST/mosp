import json

import pytest

from mosp.prepare import audit_release, main


def test_audit_reports_real_filename_overlap_and_is_read_only(tmp_path, capsys):
    gt = tmp_path / "annotations/ground_truth/CMU"
    pseudo = tmp_path / "annotations/pseudo_labels/CMU"
    vocabulary = tmp_path / "vocabulary/CMU"
    for path in (gt, pseudo, vocabulary):
        path.mkdir(parents=True)
    (gt / "gt_annotations_cmu.json").write_text(json.dumps({"test": {"state": {}}}))
    (pseudo / "pseudo_annotations_poc_new_cmu.json").write_text(json.dumps({"train": {}, "test": {}}))
    (vocabulary / "files_dict.json").write_text(json.dumps({"test": {"feature_path": "/missing/features.npy", "feature_len": 42}}))
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    main(["audit", str(tmp_path)])
    report = json.loads(capsys.readouterr().out)
    entry = report["datasets"]["CMU"]
    assert report["paper_split_verified"] is False
    assert entry["overlap_ids"] == ["test"]
    assert entry["missing_feature_path_count"] == 1
    assert {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_audit_explicit_output_refuses_overwrite(tmp_path):
    (tmp_path / "annotations").mkdir()
    output = tmp_path / "audit.json"
    main(["audit", str(tmp_path), "--output", str(output)])
    original = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        main(["audit", str(tmp_path), "--output", str(output)])
    assert error.value.code == 2
    assert output.read_bytes() == original


def test_audit_rejects_ambiguous_release_sources(tmp_path):
    folder = tmp_path / "annotations/ground_truth/EGTEA"
    folder.mkdir(parents=True)
    for name in ("gt_annotations_egtea.json", "ground_truth_annotations.json"):
        (folder / name).write_text("{}")
    with pytest.raises(ValueError, match="Ambiguous annotation"):
        audit_release(tmp_path)
