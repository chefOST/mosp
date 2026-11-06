# Setup and current limitations

## Source and branches

Imported on November 6, 2026 from
[`chefOST/oscar`, `ryan/main`](https://github.com/chefOST/oscar/tree/ryan/main).
The preserved branch points to `003308d3747846580fabc646e887cdf4e662f94d` and retains
both original commits. `main` contains shared guidance; `harrison/mosp-experiments`
contains the unchanged original implementation plus this guidance.

MOSP means Multiple Object State Prediction. SAI means State–Action Interaction.
Both belong to the [MOSCATO paper](https://github.com/eelhami/iccv25_moscato).
Ryan's original `NOTES.md` and package citation contain OSCAR references; they
are preserved as source history, not the authoritative paper identification.

## Synthetic smoke test

These commands follow Ryan's `moscato/make_synth.py` instructions. They were
inspected during migration, not executed. They generate synthetic data, train a
small model and evaluate its checkpoint; they do not reproduce the paper.
Use Python 3.10 or newer for the source's type syntax.

```sh
git switch harrison/mosp-experiments
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m moscato.make_synth --out ./synth_dataset
python -m moscato.train --data_root ./synth_dataset --dataset CMU --use_sai --epochs 15 --out runs/synth.pt
python -m moscato.evaluate --data_root ./synth_dataset --dataset CMU --ckpt runs/synth.pt
```

Expected outputs are synthetic files in `synth_dataset/`, a checkpoint at
`runs/synth.pt`, and printed evaluation metrics. These generated files are ignored.

## Before real-data runs

- Align the loader with the released dataset. `moscato/config.py` expects plural
  vocabulary filenames and generic annotation names; the shared data uses singular
  dictionary filenames and dataset-specific annotation names.
- Supply precomputed `.npy` features. This repository has no video feature extractor.
  Record feature model, dimensions, sampling and alignment with annotations.
- Review action inputs: the loader derives action IDs from annotations/metadata,
  and evaluation passes them into SAI. A predicted-action inference pipeline and
  its correspondence to the paper have not been verified.
- Validate architecture, splits, supervision and metrics against the paper before
  describing results as a reproduction. The current evaluator reports F1-max,
  mAP and segment F1, rather than all ChefOST evaluation metrics.

No completed real-data MOSP training/evaluation run was found in the inspected
repository and local records. The original branch ignores run artifacts, so this
does not establish whether an unrecorded run happened elsewhere.

Keep datasets and outputs outside ordinary Git history. Use the
[shared data repository](https://github.com/chefOST/chefOST-data) for access instructions.
