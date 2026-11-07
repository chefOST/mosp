# Weights & Biases

Intended entity: `chefOST`. Development project: `mosp-experiments`.
Full evaluations: `mosp - full evals`. Defaults are recorded in
[settings.json](settings.json).

The reproduction writes local run manifests, checkpoints and evaluation JSON.
It does not yet upload to W&B. These settings document the intended destinations;
they do not create projects or enable logging. Add an explicit logging integration
before expecting uploaded runs.

Authenticate using your own W&B account or environment credentials. Keep keys out
of Git and chat. Verify the destination in the actual run command: explicit
logger arguments can override environment variables and tracked defaults.
