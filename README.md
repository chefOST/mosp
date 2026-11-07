# MOSP / ChefOST

This branch recreates the learned model in
[MOSCATO (ICCV 2025)](https://www.khoury.northeastern.edu/home/eelhami/publications/iccv25_moscato.pdf):
MS-TCN++ object-state prediction (MOSP), temporal action segmentation (TAS), and
State–Action Interaction (SAI).

`harrison/mosp-reproduction` contains the new implementation. Ryan's original
code and history remain on `ryan/main`; `main` contains shared guidance.
The new package is `mosp`.

The model, joint losses, split checks, training, checkpoint loading, feature-only
inference and state evaluation run end to end on synthetic data. Real MOSCATO
training and published-result reproduction are not yet established. Several SAI
details are unspecified in the paper; see [reproduction decisions](docs/reproduction-decisions.md).

## Start

With Python 3.10+:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python -m mosp.smoke --out runs/smoke
```

The smoke test uses generated data and a small network. It checks execution,
not benchmark quality. Full training requires verified I3D features, explicit
train/validation/test IDs, and separate feature-to-state/action time mappings.
See [setup](docs/setup.md) and the [manifest contract](docs/data-manifest.md).

## Understand and contribute

Start with [how the model works](docs/model-explained.md) for a plain-language
explanation suitable for preparing a writeup. Use the
[shared data](https://github.com/chefOST/chefOST-data),
[project goals](https://github.com/chefOST/.github/blob/main/docs/PaperNotes.md#project-goals)
and [evaluation specification](https://github.com/chefOST/.github/blob/main/docs/Evaluation.md).
The implemented paper-style metrics are distinct from ChefOST's project metrics.

Details: [documentation](docs/README.md). Agents: [instructions](AGENTS.md).
Follow the [branch guide](https://github.com/chefOST/.github/blob/main/CONTRIBUTING.md).
