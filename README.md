# MOSP / ChefOST

Implementation of MOSP, TAS and SAI from the
[MOSCATO paper](https://www.khoury.northeastern.edu/home/eelhami/publications/iccv25_moscato.pdf).
The Python package is `mosp`.

This repository contains model training and prediction only. Benchmark scoring
and model selection belong outside this repository. No real training has been
completed; training video lists and the required feature collection still need
preparation.

## Install and check

With Python 3.10+:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
```

See [setup](docs/setup.md) for current data availability and command requirements,
[the model explanation](docs/model-explained.md) for the architecture, and
[reproduction decisions](docs/reproduction-decisions.md) for unresolved paper details.

Prediction exports probabilities and provenance for external consumers.
The [shared evaluation guide](https://github.com/chefOST/.github/blob/main/docs/Evaluation.md)
defines the separate project scoring protocol.

[Documentation](docs/README.md) · [Agent instructions](AGENTS.md)
