# Data manifest contract

Version 1 describes training data and temporal alignment. Reserved validation/test
IDs are exclusion metadata only; this repository does not load their targets or
score predictions. Populate the manifest after verifying training feature files.

## Required fields

- `schema_version`: integer `1`.
- `dataset`: `CMU`, `EGTEA` or `EgoPER`. EgoPER requires `task`.
- `split_provenance`: a `description` and boolean `paper_split_verified`.
  A true claim also requires a `source` giving evidence for the original split.
- `vocabularies`: explicit `objects`, `states` and `actions` JSON paths.
- `annotations`: explicit `pseudo_labels` and `ground_truth` JSON paths.
  Only video IDs from `ground_truth` are used, to bar those videos from training.
  Optional `action_metadata` points to the released `files_dict.json`.
- `feature_provenance`: `model`, integer `dimension`, `layout` (`DT` or `TD`),
  and a descriptive `timeline`. Include actual extractor/checkpoint, stride,
  preprocessing and timestamp details when available; dimensions alone do not
  establish compatibility with the paper.
- `splits`: `train`, `validation`, `test` arrays of unique, disjoint video IDs.
  The last two lists only reserve videos from training; they can be empty.
- `videos`: exactly one record for every training ID. No records or feature
  files are needed for reserved IDs.

Paths are absolute or relative to the manifest's directory. They are never
guessed from the dataset name. This supports CMU `object_dict.json`, EGTEA
`obj_dict.json`, singular action/state dictionary names and the release's
dataset-specific annotation filenames. Sparse source vocabulary IDs are
mapped to a contiguous internal order; that mapping is saved in checkpoints.

## Per-video record

- `feature_path`: numeric `.npy` feature matrix, with the declared dimension/layout.
- `state_source`: must be `pseudo_labels`.
- `state_indices`: one source state-timeline index per feature timepoint.
- `action_source`: `pseudo_labels` or `metadata` for TAS supervision.
- `action_indices` gives one source action-timeline
  index per feature timepoint, and `action_segment_end` explicitly states
  `inclusive` or `exclusive` boundaries.

Each index map is an integer list or an explicit path to a 1-D integer `.npy`
array. Maps must be chronological, in bounds and the required length. Repeated
indices are permitted where justified by sampling. No array-length stretching
or guessed frame-rate conversion is performed. State/action index maps refer
to separate timelines; do not assume they have the same units.

## Failures caught before meaningful training

Split overlaps, duplicate IDs, manually annotated videos assigned to training,
missing features, unknown object/state names, inconsistent vocabularies, invalid
feature shapes and missing time maps raise errors. Unknown action labels are
ignored (`-100`) rather than silently reassigned to a known class; a training
video with no known action targets fails.

Algorithm 1's correlation counts use only mapped training samples and their
explicit source action midpoints. A different sampling grid can change these
counts and must be disclosed. Adjacent annotated actions retain their boundaries
even when their class names match. No validation/test states contribute.

Checkpoints retain training IDs, reserved IDs, vocabulary and feature provenance
for external consumers. Prediction accepts a feature file directly and requires
no annotation manifest or target labels.

This contract detects mistakes; it does not independently prove that supplied
features, mappings or split-provenance claims match the authors' unreleased setup.
