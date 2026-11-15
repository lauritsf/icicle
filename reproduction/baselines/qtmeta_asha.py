"""Single-worker ASHA on Quick-Tune's recorded epochs and table-time budgets."""

import numpy as np

from reproduction.baselines.scheduler import rung_ladder
from reproduction.baselines.qtmeta_common import shared_initial_order, validate_table_task


def run_asha(objective, costs, *, cap, paired_index, task_id, seed, permutation_seed, eta=3):
    """Return accepted epochs and the first proposal exceeding the time budget.

    Scores and costs stay in float64. Promotions use the deepest eligible rung;
    after all configurations have started, finish the remaining promotions.
    """
    if eta < 2:
        raise ValueError("ASHA requires eta >= 2")
    lengths = validate_table_task(objective, costs)
    if not np.isfinite(cap) or cap < 0:
        raise ValueError("The time budget must be finite and nonnegative")

    order = shared_initial_order(
        n_configurations=len(objective), paired_index=paired_index,
        task_id=task_id, seed=seed, permutation_seed=permutation_seed,
    )
    budgets = rung_ladder(objective.shape[1], eta)
    rung_values = [[] for _ in budgets]
    promoted = [set() for _ in budgets]
    resources = np.zeros(len(objective), dtype=np.int64)
    position = 0
    cumulative = 0.0
    incumbent = float(np.nanmin(objective))
    rows = []

    def promotion(quota):
        for rung in range(len(budgets) - 2, -1, -1):
            ranked = sorted(rung_values[rung], key=lambda item: (-item[0], item[1]))
            limit = len(ranked) // eta if quota else len(ranked)
            for _, config in ranked[:limit]:
                if config not in promoted[rung] and resources[config] < lengths[config]:
                    promoted[rung].add(config)
                    return config, rung + 1
        return None, 0

    while True:
        config, rung = promotion(quota=True)
        if config is None:
            if position < len(order):
                config = int(order[position])
                position += 1
                rung = 0
            else:
                config, rung = promotion(quota=False)
        if config is None:
            return rows

        stop = min(budgets[rung], lengths[config])
        for level in range(resources[config], stop):
            proposed = cumulative + float(costs[config, level])
            accepted = proposed <= cap
            observed = None
            if accepted:
                observed = float(objective[config, level])
                incumbent = max(incumbent, observed)
                cumulative = proposed
                resources[config] = level + 1
            rows.append(dict(chosen=config, budget=level + 1, observed=observed,
                             accepted=accepted, cost=cumulative, f_best=incumbent))
            if not accepted:
                return rows
        rung_values[rung].append((float(objective[config, stop - 1]), config))
