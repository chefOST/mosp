# Data manifest contract

Version 1 describes real video split membership and temporal alignment. It is an
input contract, not an existing prepared training or validation dataset. Populate
it only after selecting actual video IDs and verifying their feature files.

## Required fields

- `schema_version`: integer `1`.
- `dataset`: `CMU`, `EGTEA` or `EgoPER`. EgoPER requires `task`.
- `split_provenance`: a `description` and boolean `paper_split_verified`.
  A true claim also requires a `source` giving evidence for the original split.
- `vocabularies`: explicit `objects`, `states` and `actions` JSON paths.
- `annotations`: explicit `pseudo_labels` and `ground_truth` JSON paths;
  optional `action_metadata` points to the released `files_dict.json`.
- `feature_provenance`: `model`, integer `dimension`, `layout` (`DT` or `TD`),
  and a descriptive `timeline`. Include actual extractor/checkpoint, stride,
  preprocessing and timestamp details when available; dimensions alone do not
  establish compatibility with the paper.
- `splits`: `train`, `validation`, `test` arrays of unique video IDs. Empty
  validation is allowed only with fixed, disclosed model-selection settings.
- `videos`: exactly one record for every selected video ID.

Paths are absolute or relative to the manifest's directory. They are never
guessed from the dataset name. This supports CMU `object_dict.json`, EGTEA
`obj_dict.json`, singular action/state dictionary names and the release's
dataset-specific annotation filenames. Sparse source vocabulary IDs are
mapped to a contiguous internal order; that mapping is saved in checkpoints.

## Per-video record

- `feature_path`: numeric `.npy` feature matrix, with the declared dimension/layout.
- `state_source`: `pseudo_labels` for training; `ground_truth` for test.
  Validation may use either, but must describe its supervision in the split plan.
- `state_indices`: one source state-timeline index per feature timepoint.
- `action_source`: `pseudo_labels`, `metadata` or `none`. Training requires
  an action source; held-out inference needs none.
- When actions are present: `action_indices` gives one source action-timeline
  index per feature timepoint, and `action_segment_end` explicitly states
  `inclusive` or `exclusive` boundaries.
- For full-frame evaluation: `evaluation_indices` gives one feature-time index
  for **every** timepoint in the native state-annotation timeline. It maps
  predictions back to the entire target timeline, rather than downsampling GT.

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

Evaluation also checks the checkpoint's original training/selection IDs,
dataset/task, vocabularies and feature provenance. A newly written manifest
cannot turn a checkpoint's training or validation videos into a held-out test.

This contract detects mistakes; it does not independently prove that supplied
features, mappings or split-provenance claims match the authors' unreleased setup.
