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
annotations and vocabularies. No real training split or complete feature
collection has been prepared.

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
python -m mosp.predict --help
```

### Validate data

`mosp.prepare validate` checks training feature files, vocabularies and index
mappings, plus split-ID exclusion rules. It does not load held-out targets or
score the model. There is no ready training manifest checked into this repository.

### Train

`mosp.train` requires `--manifest`, a new `--out` directory, `--epochs` and
`--beta`. The epoch budget is fixed for the run; beta is saved with the model.

The paper specifies four stages with ten layers each, Adam `1e-5`, and positive
state weight `5`. The 64 channels and dropout `0.5` are inherited from upstream
MS-TCN++ defaults. Feature dimension comes from the manifest; matching the paper
requires verified 2048-D I3D/Kinetics features.

Training processes one full video per optimizer step, on CPU by default.
`--device cuda` selects an available GPU. `--without-sai --beta 0` selects the
no-SAI ablation. Correlations use training labels only; test action annotations
are never model inputs.

Outputs are `run.json`, `manifest.snapshot.json` and `last.pt`, containing the
latest completed training epoch. Only training losses are recorded; there is no
held-out scoring, early stopping or best-checkpoint selection. Model selection
and benchmark scoring must run outside this repository. The manifest snapshot
retains its original relative paths; it is a record, not a relocated data bundle.

### Predict

`mosp.predict` requires a checkpoint, a feature file, `--layout`, matching
`--feature-provenance` JSON and a new `.npz` output path. It accepts no annotation
files. Outputs are `[time,class]` state/action probabilities and a JSON record
of class order, model settings, input hashes and provenance.

PyTorch's `model.eval()` disables training behavior such as dropout during
prediction. It does not compare predictions with labels or compute scores.
