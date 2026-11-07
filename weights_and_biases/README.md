# Weights & Biases

Intended entity: `chefOST`. Development project: `mosp-experiments`.
Defaults are recorded in [settings.json](settings.json).

The reproduction writes local training-loss logs, checkpoints and predictions.
It does not yet upload to W&B. These settings document the intended destinations;
they do not create projects or enable logging. Add an explicit logging integration
before expecting uploaded runs.

Authenticate using your own W&B account or environment credentials. Keep keys out
of Git and chat. Verify the destination in the actual run command: explicit
logger arguments can override environment variables and tracked defaults.
