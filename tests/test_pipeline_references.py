"""Training references cannot publish score payloads or activate an evaluator."""
import ast
from pathlib import Path

import pytest

pytest.importorskip("modal")
import modal_pipeline as pipeline


def test_pipeline_returns_only_explicit_training_references():
    result = pipeline._pipeline_references({"CMU": {
        "training": {"dataset": "CMU", "checkpoint_path": "/work/last.pt",
                     "wandb_url": "https://wandb.example/train", "metrics": {"f1": .9}},
        "evaluation": {"metrics": {"precision": .9}},
    }})
    assert result == {"CMU": {"training": {
        "dataset": "CMU", "checkpoint_path": "/work/last.pt",
        "wandb_url": "https://wandb.example/train",
    }}}


def test_training_reference_fields_cannot_hide_nested_scores():
    with pytest.raises(ValueError, match="strings"):
        pipeline._pipeline_references({"CMU": {"training": {"wandb_url": {"f1": .9}}}})


def test_training_module_has_no_scoring_entrypoint():
    tree = ast.parse(Path(pipeline.__file__).read_text())
    functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "evaluate_dataset" not in functions
    assert not (Path(pipeline.__file__).parent / "mosp/evaluate.py").exists()
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert "mosp.evaluate" not in imports
