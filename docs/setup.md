# Setup and execution

Use `harrison/mosp-reproduction`. This is a new implementation of the published
MOSP/TAS/SAI method, not the authors' code or Ryan's earlier Transformer baseline.
Ryan's two original commits are preserved on `ryan/main` at
`003308d3747846580fabc646e887cdf4e662f94d`.

## Install and check

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python -m mosp.smoke --out runs/smoke
```

The last command requires a new output directory and creates synthetic features,
labels, a two-epoch training run, checkpoints, metrics and feature-only predictions.
These are software checks, not MOSCATO accuracy results. Run artifacts are ignored
by Git. The smoke command needs no API key, remote compute or dataset download.

## Prepare real inputs

The [shared data repository](https://github.com/chefOST/chefOST-data) supplies
annotations and vocabularies. Inspect a local release without changing it:

```sh
python -m mosp.prepare audit /path/to/MOSCATO
python -m mosp.prepare validate /path/to/manifest.json
```

Both commands print JSON. The audit inventories actual files and detects
pseudo-label/ground-truth overlap; it does not invent a training split.
Build the [explicit manifest](data-manifest.md) using verified feature provenance,
split IDs and time mappings. The model does not extract I3D features or regenerate
PLA pseudo-labels. The currently inspected local release lacks feature files and
has fewer ground-truth videos than the paper's benchmark.

## Train

Choose beta and the training budget under a documented validation protocol.
The following variables intentionally have no assumed paper defaults:

```sh
python -m mosp.train --manifest /path/to/manifest.json --out runs/cmu-reproduction \
  --epochs "$MOSP_EPOCHS" --beta "$MOSP_BETA"
```

Architecture defaults follow the paper: four stages, ten layers each, 64 channels,
Adam `1e-5`, positive state weight `5`. The feature dimension comes from the
manifest; paper-comparable features must be verified I3D/Kinetics 2048-D.
The default device is CPU; choose `--device cuda` explicitly for an available GPU.
Each optimizer step processes one full video. The current implementation does
not chunk long videos or make the temporal convolutions causal.

Training estimates correlations only from training videos and passes only
features to the model. Action annotations supervise TAS. All manually annotated
video IDs in the declared ground-truth source are barred from training.

Outputs: `run.json`, `manifest.snapshot.json`, `last.pt` and `selected.pt`.
With validation videos, `selected.pt` minimizes fused-state validation BCE;
otherwise it is the explicitly budgeted final epoch. Beta remains fixed during
a run. Test data is never used for selection. Checkpoints save model settings,
vocabulary/feature provenance, training and validation IDs, correlations,
optimizer state and software revision. The manifest snapshot preserves its
original relative paths and is a record, not a relocated data bundle.

For the no-SAI ablation, use `--without-sai --beta 0`. TAS remains supervised as
in the paper's reported model setup. See `--help` for deliberate architectural
overrides; changing them should be recorded as an experiment.

## Evaluate

```sh
python -m mosp.evaluate --manifest /path/to/manifest.json \
  --checkpoint runs/cmu-reproduction/selected.pt --evaluation-timeline annotation \
  --segment-threshold "$MOSP_SEGMENT_THRESHOLD" --gap-frames 10 \
  --map-empty-classes exclude --out runs/cmu-reproduction/test-metrics.json
```

Full annotation scoring requires `evaluation_indices` for each held-out video.
The ten-frame gap rule is then applied on that annotation timeline. Explicit
`--evaluation-timeline feature` is available as a sampled diagnostic and must
not be reported as full-frame benchmark evaluation.

Segment threshold and mAP empty-class policy are explicit because the paper
does not fully specify them. `exclude` averages AP over classes with positive
targets; `zero` includes empty classes with AP zero. Scores are percentages.
F1-max's test-threshold sweep is the paper's reported statistic, not a calibrated
inference threshold. For EgoPER, run separate task models; task-average reporting
remains a separate aggregation step. TAS benchmark scores and ChefOST-specific
metrics are not included in this state-evaluation command.

## Predict from features alone

```sh
python -m mosp.predict --checkpoint runs/cmu-reproduction/selected.pt \
  --features /path/to/video.npy --layout DT \
  --feature-provenance /path/to/feature_provenance.json --out runs/video-predictions.npz
```

The provenance JSON describes the actual input extractor and timeline and must
equal the training manifest's `feature_provenance` object. Do not copy expected
metadata onto unrelated features. No annotation path is accepted. Output arrays
are `[time, class]` state/action probabilities; the companion JSON records the
class order, feature/checkpoint hashes, model settings and provenance.
