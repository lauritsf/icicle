"""Train from a JSON configuration with optional command-line overrides.

The config contains model, optimizer, prior, data and run settings. Overrides
use the same field names, and the trainer saves the resolved configuration
alongside checkpoints.

    python -m experiments.train icicle --run.seed 7
    python -m experiments.train smoke
    python -m experiments.train ftpfn-icicle-prior
"""

import dataclasses
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import tyro  # noqa: E402

from icicle_jax.train import load_recipe, train  # noqa: E402


def main():
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    experiment = tyro.extras.overridable_config_cli({
        "icicle": ("Release training configuration", load_recipe(config_dir / "icicle.json")),
        "smoke": ("Small training check", load_recipe(config_dir / "smoke.json")),
        "ftpfn-icicle-prior": (
            "FT-PFN width 512, 10 HPs, ICICLE prior, 16,000 updates",
            load_recipe(config_dir / "ftpfn_icicle_prior.json"),
        ),
    })
    for field in dataclasses.fields(experiment):
        print(f"{field.name}: {getattr(experiment, field.name)}", flush=True)
    train(experiment.model, experiment.optim, experiment.prior, experiment.data, experiment.run)


if __name__ == "__main__":
    main()
