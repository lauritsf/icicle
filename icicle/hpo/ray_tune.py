"""Pause and resume Ray Tune trials using learning-curve predictions.

Each report advances one fidelity level. Trials pause after advance_steps
reports and stop at the end of the fidelity grid. A MetricMap converts their
metrics to the surrogate's input range before each decision.
"""

import numpy as np

try:
    from ray.tune.experiment import Trial
    from ray.tune.logger import JsonLoggerCallback
    from ray.tune.result import TRAINING_ITERATION
    from ray.tune.schedulers import TrialScheduler
except ImportError as e:  # pragma: no cover
    raise ImportError("ray is required for icicle.hpo.ray_tune -- pip install 'ray[tune]'") from e

from icicle.hpo import normalize_hp
from icicle.hpo.decider import FreezeThawDecider
from icicle.hpo.metric_map import MetricMap


class FreezeThawJsonLogger(JsonLoggerCallback):
    """Keep JSON histories intact across pauses, including trials that never start.

    Ray's default logger reopens the log on each resume and can overwrite it with
    an older storage copy. Keep the file open until the trial ends.
    """

    def log_trial_start(self, trial):
        if trial not in self._trial_files:
            super().log_trial_start(trial)

    def log_trial_end(self, trial, failed=False):
        super().log_trial_end(trial, failed=failed)
        if not trial.last_result.get(TRAINING_ITERATION, 0):
            # Unstarted trials have no artifact sync, but the result reader expects a file.
            storage = trial.storage
            fs = storage.storage_filesystem
            fs.create_dir(storage.trial_fs_path, recursive=True)
            with fs.open_output_stream(f"{storage.trial_fs_path}/result.json"):
                pass

    def on_experiment_end(self, trials, **info):
        # A global stopper bypasses the per-trial completion callback.
        for trial in trials:
            self.log_trial_end(trial)


def as_unit_vector(config: dict, space) -> np.ndarray:
    """A Tune config dict -> the unit-normalized HP vector the surrogate expects.

    ``space`` lists ``(name, lower, upper, log)`` specifications. Use the same
    order for every configuration; this order defines the model's input columns.
    """
    return np.array(
        [normalize_hp(config[name], lower, upper, log) for name, lower, upper, log in space],
        dtype=np.float32,
    )


class IcicleFreezeThawScheduler(TrialScheduler):
    """Freeze-thaw scheduling with configurable intervals between pauses.

    Configure loggers through Ray's ``RunConfig(callbacks=[...])``. Use
    ``FreezeThawJsonLogger()`` to preserve JSON history across pauses and resumes.
    Other Ray logging callbacks can be added there as usual.
    """

    def __init__(
        self,
        predict_fn,
        borders,
        space,
        fid_grid,
        *,
        metric: str,
        metric_map: MetricMap | None = None,
        seed: int = 0,
        vector_fn=None,
        advance_steps: int = 1,
        fantasize: bool = True,
    ):
        """Configure prediction, metric normalization and pause intervals.

        vector_fn(config) -> (H,) overrides the default search-space
        normalization. metric_map sets the metric direction and known bounds.
        """
        super().__init__()
        self._predict_fn = predict_fn
        space = tuple(space)
        self._vector_fn = vector_fn or (lambda config: as_unit_vector(config, space))
        self._fidelity_grid = np.asarray(fid_grid, dtype=np.float32)
        self._metric = metric
        self._map = metric_map if metric_map is not None else MetricMap(worst=0.0, best=1.0)
        self._decider = FreezeThawDecider(borders=np.asarray(borders), seed=seed)
        self._steps_per_resume = max(1, int(advance_steps))
        self._fantasize = bool(fantasize)
        borders = np.asarray(borders, dtype=np.float64)
        self._bucket_centers = (borders[:-1] + borders[1:]) / 2.0
        self._pause_at_step: dict[str, int] = {}
        self._completed_steps: dict[str, int] = {}
        self._config_vectors: dict[str, np.ndarray] = {}
        # Store raw observations and rebuild the normalized context when the map changes.
        self._observed_configs: list[np.ndarray] = []
        self._observed_fidelities: list[float] = []
        self._observed_values: list[float] = []

    def on_trial_add(self, tune_controller, trial):
        self._completed_steps.setdefault(trial.trial_id, 0)
        self._config_vectors[trial.trial_id] = np.asarray(self._vector_fn(trial.config), dtype=np.float32)

    def on_trial_result(self, tune_controller, trial, result) -> str:
        # Ray restores training_iteration with the trial checkpoint.
        steps = int(result.get(TRAINING_ITERATION, self._completed_steps[trial.trial_id] + 1))
        self._completed_steps[trial.trial_id] = steps
        raw = float(result[self._metric])
        fidelity = self._fidelity_grid[min(steps, len(self._fidelity_grid)) - 1]

        self._observed_configs.append(self._config_vectors[trial.trial_id])
        self._observed_fidelities.append(float(fidelity))
        self._observed_values.append(raw)

        if steps >= len(self._fidelity_grid):
            return TrialScheduler.STOP
        if steps < self._pause_at_step.get(trial.trial_id, steps):
            return TrialScheduler.CONTINUE
        return TrialScheduler.PAUSE

    def on_trial_complete(self, tune_controller, trial, result):
        self._completed_steps[trial.trial_id] = len(self._fidelity_grid)

    def on_trial_error(self, tune_controller, trial):
        self._completed_steps[trial.trial_id] = len(self._fidelity_grid)

    def on_trial_remove(self, tune_controller, trial):
        self._completed_steps[trial.trial_id] = len(self._fidelity_grid)

    def choose_trial_to_run(self, tune_controller):
        candidates = [
            t
            for t in tune_controller.get_trials()
            if t.status in (Trial.PENDING, Trial.PAUSED)
            and self._completed_steps.get(t.trial_id, 0) < len(self._fidelity_grid)
        ]
        if not candidates:
            return None

        context_hps = np.asarray(self._observed_configs, dtype=np.float32)
        context_fidelities = np.asarray(self._observed_fidelities, dtype=np.float32)
        self._map.fit(self._observed_values)
        context_values = self._map(self._observed_values)
        if self._fantasize and self._observed_values:
            # Use predicted means for pending reports when choosing another configuration.
            # Rebuild these estimates for each decision; keep them out of observed history.
            running = [
                t
                for t in tune_controller.get_trials()
                if t.status == Trial.RUNNING
                and self._completed_steps.get(t.trial_id, 0) < len(self._fidelity_grid)
                and t.trial_id in self._config_vectors
            ]
            if running:
                running_hps = np.stack([self._config_vectors[t.trial_id] for t in running])
                running_fidelities = np.array(
                    [
                        self._fidelity_grid[self._completed_steps[t.trial_id]]
                        for t in running
                    ],
                    dtype=np.float32,
                )
                probabilities = np.asarray(
                    self._predict_fn(
                        context_hps, context_fidelities, context_values,
                        running_hps, running_fidelities,
                    )
                )
                context_hps = np.concatenate([context_hps, running_hps])
                context_fidelities = np.concatenate([context_fidelities, running_fidelities])
                context_values = np.concatenate([context_values, probabilities @ self._bucket_centers])

        idx = self._decider.decide(
            np.stack([self._config_vectors[t.trial_id] for t in candidates]),
            np.array([self._completed_steps[t.trial_id] for t in candidates]),
            self._fidelity_grid,
            self._predict_fn,
            context_hps=context_hps,
            context_fidelities=context_fidelities,
            context_values=context_values,
        )
        chosen = candidates[idx]
        self._pause_at_step[chosen.trial_id] = min(
            self._completed_steps.get(chosen.trial_id, 0) + self._steps_per_resume, len(self._fidelity_grid)
        )
        return chosen

    def debug_string(self) -> str:
        done = sum(s >= len(self._fidelity_grid) for s in self._completed_steps.values())
        return (
            f"IcicleFreezeThaw: {len(self._completed_steps)} trials ({done} finished), "
            f"{self._decider.n_decisions} decisions, {len(self._observed_values)} observations"
        )
