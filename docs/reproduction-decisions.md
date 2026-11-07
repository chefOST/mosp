# Reproduction differences and unknowns

This document lists known differences and anything whose match to MOSCATO is
unresolved. Confirmed matches are omitted. The authors' MOSCATO implementation
and checkpoint were unavailable for comparison; differences from upstream
MS-TCN++ are identified separately.

Sources: [MOSCATO paper](https://www.khoury.northeastern.edu/home/eelhami/publications/iccv25_moscato.pdf),
[supplement](https://www.khoury.northeastern.edu/home/eelhami/publications/iccv25_moscato_suppmat.pdf),
[pinned MS-TCN++ code](https://github.com/sj-li/MS-TCN2/tree/f423a9e65f4ccb1cd7322eb9f94946a19e787993).

## Known differences and missing components

- Available test data: our local ground truth contains 29 CMU, 12 EGTEA and
  46 EgoPER videos, versus the paper's 36, 25 and 50. Scores on this subset
  would not use the paper's complete test set. No verified original split
  manifest is available locally.
- TAS smoothing differs from upstream: we discard adjacent pairs if either
  action label is ignored or either frame is padded, then average over included
  elements. Upstream averages zero-masked entries over all elements. Whether
  MOSCATO changed the upstream behavior is unknown.
- Pipeline coverage: we consume supplied features and pseudo-labels. I3D
  extraction and PLA's LLM/VLM prompting, state refinement, point-of-change
  estimation and visibility detection are not implemented here.
- Repository boundary: this package implements training and prediction, with
  no benchmark scoring or validation-based checkpoint selection. Those parts
  of a reproduction must be implemented outside this repository.

## Unknown whether our implementation matches

### Losses and branch training

- Loss reduction: we average MOSP BCE over valid frames and object–state pairs
  within each stage, then sum the stage losses. SAI contributes one mean BCE
  term. The paper does not establish these exact reductions and their relative
  weighting.
- SAI-to-MOSP supervision: our SAI loss updates TAS and the fusion layer, but
  does not directly supervise the MOSP branch. We add no teacher/consistency loss
  from SAI predictions to MOSP. The paper's text describes complementary
  pseudo-labels for MOSP, while its equations do not clearly define an additional
  connection; the intended coupling is unresolved.
- Initialization and sharing: our branches have separate randomly initialized
  weights. The authors' exact initialization and weight sharing are unverified.

### SAI

- Action segments and rounding: we group consecutive final-TAS argmax labels
  into `[start,end)` segments and place state boundaries at
  `floor((start + end) / 2)`. Exact decoding and discrete boundary conventions
  are unverified.
- Action confidence: we average the selected action's predicted probability
  over the relevant half-segment. An odd-length segment shares its middle
  sample between the two halves. The paper does not specify this confidence
  aggregation or integer-rounding convention.
- Video endpoints: the missing previous/next action contributes zero evidence
  before the first midpoint or after the last. The paper does not define those
  endpoint inputs.
- Fusion: our default is a classwise `Conv1d(2*C,C,1,groups=C)` with bias and
  sigmoid, using interleaved after/before evidence for each object–state pair.
  Dense `Conv1d(2*C,C,1)` is also selectable. The paper specifies a 1D
  convolution but not its kernel, channel sharing, bias or output activation.
- Gradient path: predicted boundaries and action identities are detached;
  confidence values retain gradients into TAS. The authors' gradient handling
  through segment construction is unverified.
- Zero counts: an action/direction with no positive counts gets a zero
  correlation row. We reject SAI training if all correlation evidence is zero.
  Algorithm 1 does not specify these cases.

### Data and time alignment

- Feature equivalence: exact I3D checkpoint, RGB/flow composition, preprocessing,
  clip length, stride and timestamps remain unverified. Matching a filename,
  feature length and 2048 dimensions does not establish identical features.
- Annotation boundaries: source action end-inclusivity and feature-to-state/action
  mappings must be specified in our manifest. Their equivalence to the authors'
  conventions is unresolved.
- Correlation sampling: we count labels on the mapped feature grid, retaining
  the source action midpoints, with no duration weighting. Skipping or repeating
  source frames can change the counts. We pool counts across training videos
  before normalization. The authors' exact counting grid and cross-video
  weighting are not explicit.
- Action gaps: unannotated spans and unknown action labels are ignored by TAS
  loss (`-100`); no background category is added. In the inspected CMU example,
  only 714 of 2727 feature timepoints have action supervision. The authors'
  gap/background treatment is unknown.
- Split equivalence: our required training and reserved-ID lists have not been
  prepared and matched to the original paper splits. The loader excludes
  declared manually annotated IDs from training, but that local inventory is
  incomplete relative to the paper.

### Training schedule

- Beta and epochs: callers choose both before a run; beta is fixed during that
  run. The supplement reports a beta sweep and its best result without a clear
  selection split/criterion. The authors' epoch budget is also unspecified.
- Checkpoints: training saves the latest completed epoch without scoring it.
  The authors' checkpoint-selection procedure remains unspecified; choosing a
  checkpoint is an external responsibility, not logic implemented here.
