"""Ray Tune scheduler glue, driven with fake trials (no ray cluster started)."""

import numpy as np
import pytest

ray = pytest.importorskip("ray")

from ray.tune.experiment import Trial  # noqa: E402
from ray.tune.schedulers import TrialScheduler  # noqa: E402

from icicle.hpo.metric_map import DEFAULT_BAND, MetricMap  # noqa: E402
from icicle.hpo.ray_tune import IcicleFreezeThawScheduler, as_unit_vector  # noqa: E402

SPACE = (("lr", 1e-4, 1e-1, True), ("width", 16, 512, True), ("dropout", 0.0, 1.0, False))
BORDERS = np.linspace(0.0, 1.0, 101)
FID_GRID = (np.arange(10) + 1) / 10


class FakeTrial:
    def __init__(self, trial_id, config, status=Trial.PENDING):
        self.trial_id = trial_id
        self.config = config
        self.status = status


class FakeController:
    def __init__(self, trials):
        self.trials = trials

    def get_trials(self):
        return self.trials


def greedy_predict(context_hps, context_fidelities, context_values, qry_hps, qry_fid):
    """Confidently predicts y = the config's lr coordinate: higher lr looks better."""
    probabilities = np.zeros((len(qry_hps), len(BORDERS) - 1))
    idx = np.clip(np.searchsorted(BORDERS, qry_hps[:, 0], side="right") - 1, 0, probabilities.shape[1] - 1)
    probabilities[np.arange(len(qry_hps)), idx] = 1.0
    return probabilities


def make_setup(n_trials=4, seed=0):
    rng = np.random.default_rng(seed)
    trials = [
        FakeTrial(
            f"t{i}",
            {
                "lr": 10.0 ** rng.uniform(-4, -1),
                "width": float(rng.integers(16, 512)),
                "dropout": float(rng.uniform()),
            },
        )
        for i in range(n_trials)
    ]
    sched = IcicleFreezeThawScheduler(
        greedy_predict, BORDERS, SPACE, FID_GRID, metric="score", seed=seed
    )
    ctrl = FakeController(trials)
    for t in trials:
        sched.on_trial_add(ctrl, t)
    return sched, ctrl, trials


def test_as_unit_vector_matches_space_order_and_bounds():
    v = as_unit_vector({"lr": 1e-1, "width": 16, "dropout": 0.5}, SPACE)
    np.testing.assert_allclose(v, [1.0, 0.0, 0.5], atol=1e-7)


def test_pause_always_stop_at_grid_end():
    sched, ctrl, trials = make_setup()
    t = trials[0]
    for step in range(1, len(FID_GRID) + 1):
        decision = sched.on_trial_result(ctrl, t, {"score": 0.5, "training_iteration": step})
        expected = TrialScheduler.STOP if step == len(FID_GRID) else TrialScheduler.PAUSE
        assert decision == expected


def test_pause_intervals_survive_scheduler_restore():
    from ray import cloudpickle

    _, controller, trials = make_setup(n_trials=1)
    scheduler = IcicleFreezeThawScheduler(
        greedy_predict, BORDERS, SPACE, FID_GRID, metric="score", advance_steps=4,
    )
    trial = trials[0]
    scheduler.on_trial_add(controller, trial)
    assert scheduler.choose_trial_to_run(controller) is trial
    for step in range(1, 11):
        decision = scheduler.on_trial_result(
            controller, trial, {"score": .5, "training_iteration": step},
        )
        if step == 10:
            assert decision == TrialScheduler.STOP
        elif step % 4 == 0:
            assert decision == TrialScheduler.PAUSE
            scheduler = cloudpickle.loads(cloudpickle.dumps(scheduler))
            assert scheduler.choose_trial_to_run(controller) is trial
        else:
            assert decision == TrialScheduler.CONTINUE
    assert len(scheduler._observed_values) == 10
    assert scheduler._decider.n_decisions == 3


def test_custom_configuration_encoding():
    trials = [FakeTrial(name, {"optimizer": name}) for name in ("adam", "sgd")]
    controller = FakeController(trials)
    scheduler = IcicleFreezeThawScheduler(
        greedy_predict, BORDERS, (), FID_GRID, metric="score",
        vector_fn=lambda config: [float(config["optimizer"] == "sgd")],
    )
    for trial in trials:
        scheduler.on_trial_add(controller, trial)
    scheduler.on_trial_result(controller, trials[0], {"score": .2, "training_iteration": 1})
    assert scheduler.choose_trial_to_run(controller) is trials[1]


def test_choose_trial_advances_the_argmax():
    sched, ctrl, _ = make_setup(n_trials=0)
    # unit lr-coordinates 0.2 / 0.4 / 0.95 (coord = (log10(lr) + 4) / 3)
    trials = [
        FakeTrial(f"t{i}", {"lr": 10.0 ** (3 * c - 4), "width": 64.0, "dropout": 0.1})
        for i, c in enumerate((0.2, 0.4, 0.95))
    ]
    ctrl.trials = trials
    for t in trials:
        sched.on_trial_add(ctrl, t)
    first = sched.choose_trial_to_run(ctrl)  # cold start: random, but a valid trial
    assert first in trials
    first.status = Trial.PAUSED
    # f_best = 0.6 -> thresholds land in (0.6, 0.64]: only the 0.95 delta clears them
    sched.on_trial_result(ctrl, first, {"score": 0.6, "training_iteration": 1})

    picks = {sched.choose_trial_to_run(ctrl).trial_id for _ in range(8)}
    assert picks == {"t2"}


def test_finished_and_running_trials_are_not_candidates():
    sched, ctrl, trials = make_setup()
    for t in trials[:-1]:
        t.status = Trial.RUNNING
    last = trials[-1]
    sched.on_trial_complete(ctrl, last, {"score": 0.9})
    assert sched.choose_trial_to_run(ctrl) is None


def test_loss_mode_remaps_the_whole_context_as_upper_drifts():
    _, ctrl, _ = make_setup(n_trials=0)
    contexts = []

    def predict(hps, fid, y, query_hps, query_fid):
        contexts.append(y.copy())
        return greedy_predict(hps, fid, y, query_hps, query_fid)

    sched = IcicleFreezeThawScheduler(
        predict, BORDERS, SPACE, FID_GRID, metric="loss", metric_map=MetricMap(best=0.0, maximize=False), seed=0
    )
    trials = [FakeTrial(f"t{i}", {"lr": 1e-2, "width": 64.0, "dropout": 0.1}) for i in range(2)]
    ctrl.trials = trials
    for t in trials:
        sched.on_trial_add(ctrl, t)
    lo, hi = DEFAULT_BAND

    def presented(loss, upper):
        return lo + (1.0 - np.asarray(loss) / upper) * (hi - lo)

    sched.on_trial_result(ctrl, trials[0], {"loss": 4.0, "training_iteration": 1})
    sched.on_trial_result(ctrl, trials[0], {"loss": 2.0, "training_iteration": 2})
    sched.choose_trial_to_run(ctrl)
    np.testing.assert_allclose(contexts[-1], presented([4., 2.], 3.9), atol=1e-7)

    sched.on_trial_result(ctrl, trials[1], {"loss": 8.0, "training_iteration": 1})
    sched.choose_trial_to_run(ctrl)
    np.testing.assert_allclose(contexts[-1], presented([4., 2., 8.], 7.6), atol=1e-7)
    assert contexts[-1].max() < 1.0

    sched.on_trial_result(ctrl, trials[1], {"loss": 10.0, "training_iteration": 2})
    sched.choose_trial_to_run(ctrl)
    np.testing.assert_allclose(contexts[-1], presented([4., 2., 8., 10.], 9.7), atol=1e-7)


@pytest.mark.parametrize("fail_prediction", [False, True])
def test_running_trial_estimates_do_not_enter_observed_history(fail_prediction):
    sched, ctrl, trials = make_setup(n_trials=3)
    sched.on_trial_result(ctrl, trials[0], {"score": .3, "training_iteration": 1})
    trials[0].status = Trial.RUNNING
    contexts = []

    def predict(hps, fid, y, query_hps, query_fid):
        contexts.append(y.copy())
        if fail_prediction and len(contexts) == 2:
            raise RuntimeError("prediction failed")
        return greedy_predict(hps, fid, y, query_hps, query_fid)

    sched._predict_fn = predict
    if fail_prediction:
        with pytest.raises(RuntimeError, match="prediction failed"):
            sched.choose_trial_to_run(ctrl)
    else:
        assert sched.choose_trial_to_run(ctrl) in trials[1:]

    assert [len(y) for y in contexts] == [1, 2]
    assert sched._observed_values == [.3]
    trials[0].status = Trial.PAUSED
    sched.choose_trial_to_run(ctrl)
    np.testing.assert_array_equal(contexts[-1], contexts[0])


def test_json_logger_keeps_resume_history_and_records_unstarted_trials(tmp_path):
    import json
    from types import SimpleNamespace

    import pyarrow.fs

    from icicle.hpo.ray_tune import FreezeThawJsonLogger

    def trial_at(name):
        trial = FakeTrial(name, {"lr": 0.1})
        directory = tmp_path / name
        directory.mkdir()
        trial.local_path = str(directory)
        trial.init_local_path = lambda: None
        trial.checkpoint = None
        trial.last_result = {}
        trial.storage = SimpleNamespace(
            storage_filesystem=pyarrow.fs.LocalFileSystem(),
            trial_fs_path=str(tmp_path / "storage" / name),
        )
        return trial

    logger = FreezeThawJsonLogger()
    trial = trial_at("trained")
    logger.log_trial_start(trial)
    logger.log_trial_result(0, trial, {"training_iteration": 1, "score": 0.5})
    # A resume must not fetch a stale storage copy over the current log.
    trial.checkpoint = object()
    logger.log_trial_start(trial)
    logger.log_trial_result(1, trial, {"training_iteration": 2, "score": 0.6})
    trial.last_result = {"training_iteration": 2, "score": 0.6}
    logger.log_trial_end(trial)
    rows = [json.loads(line) for line in (tmp_path / "trained/result.json").read_text().splitlines()]
    assert [row["training_iteration"] for row in rows] == [1, 2]
    unstarted = trial_at("unstarted")
    logger.on_experiment_end([trial, unstarted])
    assert (tmp_path / "storage/unstarted/result.json").read_text() == ""
