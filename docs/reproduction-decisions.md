# Reproduction decisions

Status: November 7, 2026. This document separates published requirements from
choices needed to make the available description executable. It does not report
a completed real-data run or reproduced benchmark score.

## Sources and implementation boundary

- [MOSCATO paper](https://www.khoury.northeastern.edu/home/eelhami/publications/iccv25_moscato.pdf):
  §§3.2–3.4, Algorithm 1, and §5.1 define the learned model and evaluation.
- [MOSCATO supplement](https://www.khoury.northeastern.edu/home/eelhami/publications/iccv25_moscato_suppmat.pdf):
  §3 describes the beta sweep, segment merging and additional experiments.
- [MS-TCN++ source](https://github.com/sj-li/MS-TCN2/tree/f423a9e65f4ccb1cd7322eb9f94946a19e787993):
  architecture and training defaults, pinned to
  `f423a9e65f4ccb1cd7322eb9f94946a19e787993`.
  The upstream implementation is by Shi-Jie Li and collaborators and is
  [MIT licensed](https://github.com/sj-li/MS-TCN2/blob/f423a9e65f4ccb1cd7322eb9f94946a19e787993/LICENSE).
  Code adapted from it must retain its copyright and permission notice.

The authors' MOSCATO repository currently exposes documentation and data links,
not a runnable model or checkpoint. This implementation is a ChefOST recreation;
compatibility with unavailable official weights has not been established.

## Unverified loss conventions

- Average MOSP binary cross-entropy over valid frames and object–state pairs
  within each stage, then sum the stage losses. SAI contributes one mean BCE
  term. MOSCATO does not specify these exact reductions or relative stage weights.
- Exclude TAS smoothing pairs if either frame is padded or its action label is
  ignored, and average over the remaining elements. This differs from upstream's
  global mean of zero-masked entries; MOSCATO's handling is unverified.

## SAI details not fixed by the paper

The following are explicit engineering choices, not verified author settings:

- **Zero-count rows:** keep an action's before/after correlation row zero when
  its total count is zero. Algorithm 1 does not specify this zero-denominator
  case. Training rejects SAI if all correlation evidence is zero.
- **Segments:** take consecutive runs of the final TAS argmax predictions.
  Represent predicted action intervals as `[start, end)` and put a state boundary
  at `floor((start + end) / 2)`.
- **Confidence:** use the mean predicted probability of the segment's selected
  action class over the relevant half. The before half is
  `[start, ceil((start + end) / 2))`; the after half is
  `[floor((start + end) / 2), end)`. They share the middle sample for odd-length
  segments, so a one-sample action has two valid half estimates. The paper does
  not specify this averaging or integer-rounding rule.
- **Video ends:** before the first action midpoint, supply zero evidence for
  the absent previous action; after the last midpoint, supply zero evidence for
  the absent next action. This zeros the missing fusion input, not the entire
  SAI prediction. The paper names `u0` and `uN` without defining these endpoint
  cases.
- **Fusion:** interleave the two evidence values for each of the `C = O*S`
  pairs. The default is a learned `Conv1d(2*C, C, kernel_size=1, groups=C)`
  followed by sigmoid. Each pair learns to combine its own before/after
  evidence. The paper states only “1D convolution”; its kernel, channel
  sharing, bias and output activation are unspecified.
- **Alternative fusion:** dense `Conv1d(2*C, C, 1)` permits cross-pair mixing,
  but grows quadratically in `C`. It is an optional, separately recorded
  configuration, not a more authoritative interpretation of the paper.
- **Gradient path:** segment boundaries and selected action identities are
  discrete; confidence averages retain a differentiable path into TAS. This
  distinction must be preserved when testing joint training.

Both fusion variants need real-data validation. Neither should be described as
the exact author implementation without additional evidence.

## Features and time alignment

The paper names the I3D family, Kinetics pretraining and dimensionality. It does
not establish the RGB versus optical-flow composition, checkpoint, input crop,
normalization, clip length, feature stride or timestamp convention. A
2048-dimensional array alone therefore does not establish feature equivalence.

The [manifest contract](data-manifest.md) defines the provenance and index maps.
Do not infer alignment by stretching sequences to matching lengths or assuming
the raw video rate equals the annotation rate. Timestamp-to-index conversion
must be justified by the actual extractor and source annotations.

Ground-truth actions are training targets and inputs to training-only
correlation estimation. They are not inference inputs. Held-out state labels
are evaluation targets only.

Unannotated action gaps and unknown action labels are currently ignored by the
TAS loss (`-100`); no background category is invented. The inspected CMU action
vocabulary has 29 named actions and no background entry. In the S30 real-input
diagnostic, source segments supervise 714 of 2727 feature timepoints. The authors'
gap/background policy needs confirmation before claiming identical TAS training.

## Split and model selection protocol

There is no independently verified paper split manifest in this checkout.
A pseudo-label file is not automatically
a training-only split: local pseudo-label bundles also contain ground-truth
evaluation video IDs, which must be excluded from training.

The local curated ground truth contains 29 CMU, 12 EGTEA and 46 EgoPER videos.
The paper reports 36, 25 and 50, respectively. Results on the local subset
must identify that subset and cannot directly establish reproduction of the
published test-set results. EgoPER uses one model per task with task-averaged
reporting; CMU and EGTEA use one model each (§5.1).

The supplement reports trying `beta` values
`0, 0.15, 0.30, 0.50, 0.70, 0.85, 1` and reporting the best, but does not identify
a held-out selection split. It also does not give a training epoch count or a
checkpoint-selection rule. The implementation requires an explicit beta;
do not present an arbitrary default as the paper's choice. The upstream
MS-TCN++ example's 100 epochs is not a verified MOSCATO schedule.

For ChefOST experiments, define the validation videos and selection metric
before training. Fit correlations and model parameters on training videos only;
select beta and checkpoint using validation data, then freeze them for a single
test evaluation. Record the chosen beta, epoch, seed, manifests and code revision.
If no validation split is available, preregister fixed settings and disclose
that they were not tuned. Do not select settings using the test labels.

## Evaluation and the PLA boundary

The paper's F1-max sweeps one global threshold from `0.1` through `0.9`; it pools
F1 counts over the evaluation set. The resulting best test-set threshold is a
reported benchmark statistic, not evidence that a deployed decision threshold
was calibrated. Deployment thresholds need separate validation.

The supplement merges positive segments separated by fewer than ten background
frames. The unit is the evaluated annotation timeline; applying that rule to
features with a different stride would change its meaning. Record that timeline,
the segment extraction threshold and any postprocessing. Keep these paper
metrics distinct from the shared ChefOST metric definitions.

The current scorer applies gap merging only to thresholded segment predictions;
frame F1-max and mAP use the unmodified probabilities. The supplement does not
fully specify the interaction between postprocessing and frame metrics. The mAP
empty-class policy is required explicitly (`exclude` or `zero`), and AP evaluates
tied probabilities together. Full annotation scoring uses an explicit
annotation-to-feature prediction index map; sampled feature-timeline scoring is
separately labelled as a diagnostic. These choices must accompany reported scores.

This recreation consumes released training pseudo-labels. It does not recreate
PLA's GPT/VLM prompting, point-of-change estimation or GLIP visibility detection.
That is a deliberate scope boundary: learning the MOSP/TAS/SAI model from the
released labels is separate from independently reproducing label generation.
