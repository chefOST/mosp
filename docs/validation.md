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
- One original public I3D feature file, `S30_Eggs_7150991-1.npy`, was downloaded
  and validated against released CMU annotations: float32 `[2727,2048]`,
  39 objects, 77 states, 29 actions, and 11 source action segments. The loader
  produced `[2048,2727]` features and `[3003,2727]` state labels. All 87 available
  ground-truth IDs were retained in its training-exclusion inventory.
- The full default model produced finite outputs on an explicit 64-timepoint
  slice of that real feature sequence. This used random weights, no optimization
  and no evaluation scores. The SHA256 of the feature file is
  `25d2845505a3e4ce4ec881d0ce68fd4ae575750a37be4c4f66375e0b28845e54`.

Tests ran with the existing local VidOSC Python environment: PyTorch 2.15.0,
NumPy 2.5.4, pytest 9.2.0, CPU. No new environment was installed. Synthetic run
outputs live in ignored `runs/smoke-20261107-full-timeline/`; regenerate them with
`python -m mosp.smoke --out runs/a-new-smoke-directory`.
The ignored `runs/real-data-diagnostic-20261107-s30/` directory contains the
real-input manifest, source hashes, validation output and diagnostic receipt.

## Before a real reproduction claim

Obtain and validate source features, source annotation alignment and exact
train/validation/test membership. The local curated evaluation subset differs
from the paper's reported split. Fix hyperparameter selection before test scoring.
Validate the explicit SAI assumptions in [reproduction decisions](reproduction-decisions.md).

Promising existing feature releases are listed in [feature sources](feature-sources.md).
Their availability does not establish complete coverage or identical extraction.
No real-data result or synthetic score here should be cited as MOSCATO performance.
