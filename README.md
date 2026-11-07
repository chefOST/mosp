# MOSP / ChefOST

Implementation of MOSP, TAS and SAI from the
[MOSCATO paper](https://www.khoury.northeastern.edu/home/eelhami/publications/iccv25_moscato.pdf).
The Python package is `mosp`.

The model, training code and state evaluator are implemented. No real training
or benchmark evaluation has been completed. Training/validation video lists
and the required feature collection still need preparation.

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

The evaluator reports paper-style state metrics. ChefOST's separate metrics are
specified in the [shared evaluation guide](https://github.com/chefOST/.github/blob/main/docs/Evaluation.md).

[Documentation](docs/README.md) · [Agent instructions](AGENTS.md)
