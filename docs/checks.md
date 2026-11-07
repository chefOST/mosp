# Implementation checks

The model has not been trained on real MOSCATO data. No real training/validation
split has been prepared, and no benchmark result has been produced.

Unit tests cover model shapes, losses and gradients, SAI evidence, metrics,
vocabulary handling, split exclusions and explicit temporal mappings. They check
individual functions; they do not run a training experiment or save checkpoints.

One original I3D feature file, `S30_Eggs_7150991-1.npy`, was checked against the
released CMU annotations. Its float32 array has shape `[2727,2048]`; the loader
produced `[3003,2727]` state targets for 39 objects and 77 states. An explicitly
selected short slice passed through the default model with random weights and
finite outputs. This establishes input compatibility, not prediction quality.
Source files and receipts are in ignored `features/` and
`runs/real-data-diagnostic-20261107-s30/`.

Real training requires verified features and alignment, selected video IDs,
compute resources and a training/model-selection plan. The current local test
collection differs from the paper's reported split. See
[feature sources](feature-sources.md) and
[reproduction decisions](reproduction-decisions.md) for the remaining questions.
