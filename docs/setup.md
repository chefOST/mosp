# Setup and execution

## Install and check

With Python 3.10+:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
```

These are unit checks, not a training or benchmark run.

## Data status

The [shared data repository](https://github.com/chefOST/chefOST-data) supplies
annotations and vocabularies. No real training/validation split or complete
feature collection has been prepared.

One original I3D file, `S30_Eggs_7150991-1.npy`, is downloaded locally under
`features/`. Its `[2727,2048]` features and matching annotations passed a loader
check and a short model forward pass with random weights. This was not training
or an accuracy evaluation. The receipt and source hashes are in
`runs/real-data-diagnostic-20261107-s30/`; these local artifacts are not in Git.

If the shared `MOSCATO` directory is next to this checkout, inspect it with:

```sh
python -m mosp.prepare audit ../MOSCATO
```

The audit reports available files and overlapping video IDs. It does not select
training videos. See [feature sources](feature-sources.md) for downloads and the
[data manifest](data-manifest.md) for the required split and timeline mappings.
The code consumes features and released pseudo-labels; it does not extract I3D
features or regenerate pseudo-labels.

## CLI reference: requires prepared inputs

The following help commands work without a prepared dataset:

```sh
python -m mosp.prepare validate --help
python -m mosp.train --help
python -m mosp.evaluate --help
python -m mosp.predict --help
```

### Validate data

`mosp.prepare validate` takes a real manifest path and checks all declared splits,
feature files, vocabularies and index mappings. There is no ready training
manifest checked into this repository.

### Train

`mosp.train` requires `--manifest`, a new `--out` directory, `--epochs` and
`--beta`. Choose the epoch budget and beta before test evaluation.

The paper specifies four stages with ten layers each, Adam `1e-5`, and positive
state weight `5`. The 64 channels and dropout `0.5` are inherited from upstream
MS-TCN++ defaults. Feature dimension comes from the manifest; matching the paper
requires verified 2048-D I3D/Kinetics features.

Training processes one full video per optimizer step, on CPU by default.
`--device cuda` selects an available GPU. `--without-sai --beta 0` selects the
no-SAI ablation. Correlations use training labels only; test action annotations
are never model inputs.

Outputs are `run.json`, `manifest.snapshot.json`, `last.pt` and `selected.pt`.
With validation videos, the selected checkpoint minimizes validation fused-state
BCE. Otherwise it is the requested final epoch. The manifest snapshot retains
its original relative paths; it is a record, not a relocated data bundle.

### Evaluate

`mosp.evaluate` requires a manifest, checkpoint, `--evaluation-timeline`,
`--segment-threshold`, `--gap-frames` and `--map-empty-classes`.

Use `annotation` timing with explicit `evaluation_indices` for full-frame
scoring. `feature` timing is a sampled diagnostic. Gap length is measured on the
chosen evaluation timeline. The mAP policy is `exclude` (skip classes without
positives) or `zero` (include them with AP zero).

The command reports state F1-max, mAP and segment F1 as percentages. It does not
report TAS or ChefOST-specific metrics. EgoPER task-average reporting is a
separate aggregation step. See [reproduction decisions](reproduction-decisions.md)
for scoring choices and comparison limits.

### Predict

`mosp.predict` requires a checkpoint, a feature file, `--layout`, matching
`--feature-provenance` JSON and a new `.npz` output path. It accepts no annotation
files. Outputs are `[time,class]` state/action probabilities and a JSON record
of class order, model settings, input hashes and provenance.
