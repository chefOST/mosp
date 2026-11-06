# MOSP / ChefOST

ChefOST evaluates object-state tracking in procedural cooking videos. This repository
is for Multiple Object State Prediction (MOSP), the network in
[MOSCATO (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Zameni_MOSCATO_Predicting_Multiple_Object_State_Change_Through_Actions_ICCV_2025_paper.html),
including its State–Action Interaction (SAI) module.

`main` holds shared project guidance. Ryan's original implementation and history
are preserved on [`ryan/main`](https://github.com/chefOST/mosp/tree/ryan/main), copied
from `chefOST/oscar` at `003308d3747846580fabc646e887cdf4e662f94d` (August 21, 2026).
Continue experiments on
[`harrison/mosp-experiments`](https://github.com/chefOST/mosp/tree/harrison/mosp-experiments),
which adds this guidance to Ryan's code. The Python package remains `moscato`.

The implementation includes training, evaluation and synthetic smoke-test utilities.
No completed real-data run or paper reproduction has been verified. Start with
[setup and current limitations](docs/setup.md) before running experiments.

## Data and evaluation

Use [shared annotations and vocabularies](https://github.com/chefOST/chefOST-data)
and [video access](https://github.com/chefOST/chefOST-data#videos).
The current loader also requires precomputed features and dataset-layout alignment.

ChefOST uses frame-averaged multilabel precision and recall (detail), cosine
similarity and state-change timing F1; follow the
[shared evaluation specification](https://github.com/chefOST/.github/blob/main/docs/Evaluation.md).
Ryan's evaluator reports F1-max, mAP and segment F1; it does not implement all
ChefOST metrics. Keep published results separate from ChefOST experiments.

## Context

Read the [project goals](https://github.com/chefOST/.github/blob/main/docs/PaperNotes.md#project-goals)
and [MOSCATO paper notes](https://github.com/chefOST/.github/blob/main/docs/PaperNotes.md#7-moscato).
Follow the [branch and documentation guide](https://github.com/chefOST/.github/blob/main/CONTRIBUTING.md).
See [W&B setup](weights_and_biases/README.md) for intended experiment destinations.

Details: [documentation](docs/README.md). Agents: [instructions](AGENTS.md).

