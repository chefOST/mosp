# How the model works

This is ChefOST's implementation of the MOSCATO paper's learned prediction
pipeline. It is designed to follow the published method, with the unresolved
details recorded in [reproduction decisions](reproduction-decisions.md).
Passing software checks does not establish that we have reproduced the paper's
results.

## From video to a sequence of features

The model receives one visual feature vector per sampled timepoint, rather than
raw video pixels. Think of a vector as a numerical description of a short piece
of video. Feature extraction is a separate preprocessing step. The paper uses
2048-dimensional features from an I3D network pretrained on Kinetics; the exact
RGB/flow extraction recipe is not supplied. Our input manifest records where
features came from and how each timepoint aligns with the labels.

Two temporal networks read the same feature sequence. One predicts object
states; the other predicts actions. Both use the MS-TCN++ architecture: an initial
prediction stage followed by three stages that refine those predictions. Each
stage has ten dilated temporal convolution layers. Dilated layers examine
increasingly spaced timepoints, allowing nearby appearance and longer context to
influence a prediction. The refinements operate on the previous stage's
probabilities. See the [paper, §§3.2–3.4 and §5.1](https://www.khoury.northeastern.edu/home/eelhami/publications/iccv25_moscato.pdf).

## States and actions are different outputs

The state branch produces one score for every object-category/state pair.
For example, bread can be both “sliced” and “on plate.” Independent sigmoid
probabilities allow both to be true. A softmax over those pairs would force them
to compete unnecessarily.

The action branch, called temporal action segmentation (TAS), instead predicts
one action category at each timepoint. Its softmax probabilities express that
competition. Consecutive predictions of the same action form an action segment.
The action labels supplied during training teach this branch; test-time state
prediction uses its predicted actions.

## Actions provide a second source of state evidence

State–Action Interaction (SAI) connects the branches. Training labels establish
which object states tend to occur before and after each action. At inference,
the predicted action sequence and its confidence select this before/after
evidence. A learned fusion layer turns it into another set of state probabilities.
The final prediction blends the visual state probabilities with the SAI
probabilities using the explicitly chosen weight `beta`.

For example, visual evidence that bread has changed can be supported by a
predicted “slice bread” action and the learned association between that action
and sliced bread. An incorrect action prediction can also introduce incorrect
state evidence, so SAI is not a guarantee of correctness.

## Learning and scope

Training combines three objectives: state prediction against pseudo-labels,
SAI state prediction against those same labels, and action classification plus
temporal smoothing. Released pseudo-labels let us train without rerunning the
paper's LLM/VLM label-generation pipeline. Manually annotated test states are
reserved for evaluation.

This is offline sequence processing: temporal convolutions can use later frames,
so the implementation is not a causal live tracker. Its outputs describe object
categories and states; it does not assign persistent instance identities or
track separate bread slices through splitting and merging.
