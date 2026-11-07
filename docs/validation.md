# Validation record

November 7, 2026. This records software validation, not reproduced benchmark
results. The implementation has not been trained on real MOSCATO videos.

## Executed

- 63 automated tests pass, covering action/state losses, stage shapes,
  SAI correlations and gradients, single-frame videos, split leakage,
  independent timelines, sparse vocabularies, complete annotation scoring,
  metric edge cases, checkpoint roundtrip and annotation-free inference.
- A two-epoch synthetic end-to-end run completed training, validation-based
  checkpoint selection, test evaluation and prediction without annotation
  inputs. All losses and gradients were finite. This run used a small network
  and generated numeric features, not I3D or MOSCATO videos.
- A separate forward/backward check used the paper architecture defaults:
  2048 input features, four stages and ten layers per stage, with six synthetic
  object-state classes, three action classes and 24 timepoints. Gradients were
  finite. This checks the configured architecture, not training quality.
- The actual local data audit found 29 CMU, 12 EGTEA and 46 EgoPER ground-truth
  videos. All 87 also occur in pseudo-label inventories. Source feature paths
  do not resolve locally. This is why explicit splits and feature paths are required.
- Independent implementation review checked the model/trainer interface,
  checkpoint selection, absence of test action inputs and metric aggregation.

Tests ran with the existing local VidOSC Python environment: PyTorch 2.15.0,
NumPy 2.5.4, pytest 9.2.0, CPU. No new environment was installed. Synthetic run
outputs live in ignored `runs/smoke-20261107-full-timeline/`; regenerate them with
`python -m mosp.smoke --out runs/a-new-smoke-directory`.

## Before a real reproduction claim

Obtain and validate source features, source annotation alignment and exact
train/validation/test membership. The local curated evaluation subset differs
from the paper's reported split. Fix hyperparameter selection before test scoring.
Validate the explicit SAI assumptions in [reproduction decisions](reproduction-decisions.md).

Promising existing feature releases are listed in [feature sources](feature-sources.md).
Their availability does not establish complete coverage or identical extraction.
No real-data result or synthetic score here should be cited as MOSCATO performance.
