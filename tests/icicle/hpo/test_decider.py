"""The live-trial decider must reproduce the tabular loop's decisions exactly."""

import numpy as np

from icicle.hpo import FreezeThawDecider
from reproduction.baselines.scheduler import run_freeze_thaw
from tests.reproduction.baselines.test_scheduler import BORDERS, make_task, oracle_predict


def raw_oracle(task):
    """The tabular oracle restated over raw (config vector, fidelity value) queries."""
    tab = oracle_predict(task)

    def predict(ctx_hps, ctx_fid, ctx_y, qry_hps, qry_fid):
        cfg = np.array([np.flatnonzero((task.hps == h).all(1))[0] for h in qry_hps])
        lvl = np.searchsorted(task.fidelities, np.asarray(qry_fid) - 1e-6)
        e = np.zeros(0, dtype=np.int32)
        return tab(e, e, np.zeros(0), cfg.astype(np.int32), lvl.astype(np.int32))

    return predict


def drive_decider(task, predict, n_iters, seed):
    """Replay the freeze-thaw loop with an explicit observation history."""
    n_cfg, b_max = task.curves.shape
    dec = FreezeThawDecider(borders=BORDERS, seed=seed)
    n_obs = np.zeros(n_cfg, dtype=np.int64)
    chosen = []
    hps, fidelities, values = [], [], []
    for _ in range(n_iters):
        cand = np.flatnonzero(n_obs < b_max)
        if cand.size == 0:
            break
        idx = dec.decide(
            task.hps[cand], n_obs[cand], task.fidelities, predict,
            context_hps=hps, context_fidelities=fidelities, context_values=values,
        )
        c = int(cand[idx])
        y = float(task.curves[c, n_obs[c]])
        hps.append(task.hps[c])
        fidelities.append(float(task.fidelities[n_obs[c]]))
        values.append(y)
        n_obs[c] += 1
        chosen.append(c)
    return np.asarray(chosen), dec


def test_decision_parity_with_tabular_loop():
    task = make_task()
    for seed in (0, 3, 7):
        res = run_freeze_thaw(task, oracle_predict(task), BORDERS, n_iters=40, seed=seed)
        chosen, dec = drive_decider(task, raw_oracle(task), n_iters=40, seed=seed)
        np.testing.assert_array_equal(chosen, res.chosen)


def test_bookkeeping_and_reproducibility():
    task = make_task()
    chosen1, dec = drive_decider(task, raw_oracle(task), n_iters=25, seed=5)
    chosen2, _ = drive_decider(task, raw_oracle(task), n_iters=25, seed=5)
    np.testing.assert_array_equal(chosen1, chosen2)
    assert dec.n_decisions == 25


def test_cold_start_is_uniform_random_candidate():
    task = make_task()

    def exploding_predict(*args):
        raise AssertionError("predict_fn must not be called before any observation")

    dec = FreezeThawDecider(borders=BORDERS, seed=1)
    idx = dec.decide(
        task.hps, np.zeros(len(task.hps), int), task.fidelities, exploding_predict,
        context_hps=[], context_fidelities=[], context_values=[],
    )
    assert 0 <= idx < len(task.hps)
