"""Tune a classifier with ICICLE and Ray Tune.

This example trains a classifier on synthetic data using one worker and two
CPU slots. ICICLE chooses which configuration to train next, and Ray Tune
saves and restores checkpoints so training can pause and resume.

    uv pip install '.[ray]' matplotlib
    python examples/raytune_freeze_thaw.py

Results and plots are saved under ray_results/. The script also prints the
best observed validation accuracy and its configuration.

To use your own training code, replace ExampleModel and update SPACE.
"""

import os
import pickle
from pathlib import Path
from time import time

import numpy as np
import ray
from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator

from icicle.hpo.ray_tune import FreezeThawJsonLogger, IcicleFreezeThawScheduler
from icicle.serving import OnnxSurrogate

# Parameter name, lower bound, upper bound, logarithmic sampling.
SPACE = (
    ("feat_scale", 0.7, 1.3, True),
    ("l2", 1e-6, 1e-2, True),
    ("lr", 0.003, 0.3, True),
    ("momentum", 0.0, 0.9, False),
)
BATCH_SIZE = 128
TRAINING_SEED = 1
METRIC = "val_acc"


class Trainable(tune.Trainable):
    """Connect training steps and checkpoints to Ray Tune."""

    def setup(self, config):
        self.model = ExampleModel(config)

    def step(self):
        started = time()
        accuracy = self.model.train_step()
        return {METRIC: accuracy, "step_started": started, "step_finished": time()}

    def save_checkpoint(self, checkpoint_dir):
        self.model.save(checkpoint_dir)
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir):
        self.model.load(checkpoint_dir)

    def reset_config(self, new_config):
        self.setup(new_config)
        return True


class StepBudget(tune.Stopper):
    """Stop the search after a shared number of reported training steps."""

    def __init__(self, limit):
        self.limit = limit
        self.spent = 0
        self.values = []

    def __call__(self, trial_id, result):
        self.spent += 1
        self.values.append((trial_id, result[METRIC]))
        return False

    def stop_all(self):
        return self.spent >= self.limit

def main(storage_path="ray_results", *, max_steps=40, total_steps=240, candidates=24,
         workers=1, train_samples=8000, updates_per_step=10, features=64):
    model = OnnxSurrogate(config_dim=len(SPACE), intra_op_num_threads=2)
    scheduler = IcicleFreezeThawScheduler(
        model, model.borders, SPACE,
        fid_grid=np.arange(1, max_steps + 1) / max_steps,
        metric=METRIC, advance_steps=4,
        fantasize=workers > 1,
    )
    param_space = {
        name: tune.loguniform(low, high) if log else tune.uniform(low, high)
        for name, low, high, log in SPACE
    }
    param_space.update(train_samples=train_samples, updates_per_step=updates_per_step,
                       features=features)
    budget = StepBudget(total_steps)
    os.environ["RAY_AUTH_MODE"] = "disabled"
    ray.init(num_cpus=2 * workers, include_dashboard=False, log_to_driver=False)
    try:
        results = tune.Tuner(
            tune.with_resources(Trainable, {"cpu": 2}),
            param_space=param_space,
            tune_config=tune.TuneConfig(
                scheduler=scheduler, search_alg=BasicVariantGenerator(random_state=0),
                num_samples=candidates, reuse_actors=True,
            ),
            run_config=tune.RunConfig(
                storage_path=str(Path(storage_path).resolve()), stop=budget, verbose=0,
                callbacks=[FreezeThawJsonLogger()],
                checkpoint_config=tune.CheckpointConfig(checkpoint_at_end=True),
            ),
        ).fit()
        if results.errors:
            raise results.errors[0]
        best = results.get_best_result(metric=METRIC, mode="max", scope="all")
        print(f"Recorded {budget.spent} training steps (budget target: {total_steps}); "
              f"{budget.spent * updates_per_step} optimizer updates")
        print(f"Best observed accuracy: {best.metrics_dataframe[METRIC].max():.3f}")
        print(f"Configuration: {best.config}")
        print(f"Results: {results.experiment_path}")
        print(f"Plot: {save_plot(results, budget.values, updates_per_step)}")
        return results
    finally:
        ray.shutdown()


def make_data(seed=0, n_train=800, n_validation=400, input_dim=8, n_features=64):
    """Generate a binary classification task with a fixed train/validation split."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_train + n_validation, input_dim)).astype(np.float64)
    w = rng.standard_normal((input_dim, n_features))
    b = rng.uniform(0.0, 2.0 * np.pi, n_features)
    teacher = rng.standard_normal(n_features)
    y = (np.cos(x @ w + b) @ teacher > 0).astype(np.float64)
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:], w, b


class ExampleModel:
    """NumPy logistic regression on Fourier features."""

    def __init__(self, config):
        xtr, self.ytr, xva, self.yva, w, b = make_data(
            n_train=config["train_samples"], n_features=config["features"],
        )
        self.phi_tr = np.cos(xtr @ (w * config["feat_scale"]) + b)
        self.phi_va = np.cos(xva @ (w * config["feat_scale"]) + b)
        self.lr, self.l2, self.mom = config["lr"], config["l2"], config["momentum"]
        self.rng = np.random.default_rng(TRAINING_SEED)
        self.w = self.rng.normal(0.0, 0.2, self.phi_tr.shape[1])
        self.v = np.zeros_like(self.w)
        self.updates_per_step = config["updates_per_step"]
        self.order = self.rng.permutation(len(self.phi_tr))
        self.position = 0

    def train_step(self):
        for _ in range(self.updates_per_step):
            if self.position == len(self.order):
                self.order = self.rng.permutation(len(self.phi_tr))
                self.position = 0
            indices = self.order[self.position : self.position + BATCH_SIZE]
            self.position += len(indices)
            phi, yb = self.phi_tr[indices], self.ytr[indices]
            p = 1.0 / (1.0 + np.exp(-(phi @ self.w)))
            grad = phi.T @ (p - yb) / len(yb) + self.l2 * self.w
            self.v = self.mom * self.v - self.lr * grad
            self.w = self.w + self.v
        acc = float((((self.phi_va @ self.w) > 0) == self.yva).mean())
        return acc

    def save(self, checkpoint_dir):
        (Path(checkpoint_dir) / "state.pkl").write_bytes(
            pickle.dumps((self.w, self.v, self.rng, self.order, self.position))
        )
        return checkpoint_dir

    def load(self, checkpoint_dir):
        self.w, self.v, self.rng, self.order, self.position = pickle.loads(
            (Path(checkpoint_dir) / "state.pkl").read_bytes()
        )


def save_plot(results, values, updates_per_step):
    """Show learning curves and search progress by configuration and tuning step."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    histories, leaders, progress = {}, [], []
    best, owner = -np.inf, None
    for step, (trial_id, value) in enumerate(values, 1):
        histories.setdefault(trial_id, []).append((step, value))
        x = step * updates_per_step
        if step > 1:
            progress.append(([(x - updates_per_step, best), (x, best)], step, owner))
        if value > best:
            if trial_id not in leaders:
                leaders.append(trial_id)
            if step > 1:
                progress.append(([(x, best), (x, value)], step, trial_id))
            best, owner = value, trial_id
    palette = plt.get_cmap("tab10" if len(leaders) <= 10 else "tab20")
    colors = {key: palette(i % palette.N) for i, key in enumerate(leaders)}
    norm, cmap = Normalize(0, len(values)), plt.get_cmap("viridis")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col", sharey=True,
                             layout="constrained")
    for trial_id, history in sorted(histories.items(), key=lambda item: item[0] in colors):
        steps, accuracy = np.asarray(history).T
        updates = np.arange(1, len(history) + 1) * updates_per_step
        leading = trial_id in colors
        axes[0, 0].plot(updates, accuracy, color=colors.get(trial_id, "0.65"),
                        lw=1.6 if leading else 1.1)
        points = np.column_stack([updates, accuracy])
        axes[1, 0].add_collection(LineCollection(
            np.stack([points[:-1], points[1:]], axis=1), colors=cmap(norm(steps[1:])),
            linewidths=1.6 if leading else 1.1, alpha=1 if leading else 0.5,
        ))
    for row, meaning in enumerate(["configuration colors", "tuning-step colors"]):
        curves, search = axes[row]
        segment_colors = [colors[key] if row == 0 else cmap(norm(step))
                          for _, step, key in progress]
        search.add_collection(LineCollection([segment for segment, _, _ in progress],
                                              colors=segment_colors, linewidths=2))
        curves.set(title=f"Learning curves — {meaning}", ylabel="Validation accuracy",
                   xlim=(0, max(map(len, histories.values())) * updates_per_step * 1.02))
        search.set(title=f"Search progress — {meaning}", xlim=(0, len(values) * updates_per_step))
        for ax in (curves, search):
            ax.set_ylim(0, 1.02)
            ax.grid(axis="y", alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_xlabel("Optimizer updates per configuration")
    axes[1, 1].set_xlabel("Total reported optimizer updates")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[1, :],
                 label=f"Tuning step ({updates_per_step} optimizer updates per report)",
                 ticks=np.linspace(0, len(values), 5), fraction=0.035, pad=0.025)
    path = Path(results.experiment_path) / "search.svg"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
