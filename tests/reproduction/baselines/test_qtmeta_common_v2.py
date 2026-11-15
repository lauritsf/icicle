"""Domain-separated seeding, structural ceilings, and v2 conventions."""

import numpy as np
import pytest

from reproduction.baselines.qtmeta_common import (
    _domain_key,
    domain_digest,
    initial_index,
    random_search_events,
    structural_ceiling,
)
from reproduction.baselines.qtmeta_asha import shared_initial_order

TASK = "qtmeta/micro/set0-BCT"
PAIRED_LABEL = "icicle/qtmeta-common-v2/paired-start/v1"
ORDER_LABEL = "icicle/qtmeta-common-v2/candidate-order/v1"


def v2_protocol(n_candidates: int = 6) -> dict:
    curves = np.full((n_candidates, 4), np.nan)
    costs = np.full((n_candidates, 4), np.nan)
    for config in range(n_candidates):
        length = 2 + config % 3
        curves[config, :length] = np.linspace(0.2, 0.8, length) + 0.01 * config
        costs[config, :length] = 10.0 + config
    return (
        {
            "initialization": {"domain_label": PAIRED_LABEL},
            "random_search": {"domain_label": ORDER_LABEL},
            "budget": {"caps_seconds": {"micro": 60.0}},
        },
        curves,
        costs,
    )


class TestDomainKeys:

    def test_frozen_byte_encoding(self):
        key = _domain_key({"domain_label": PAIRED_LABEL}, TASK, 0)
        assert key == f"{PAIRED_LABEL}|{TASK}|0".encode()
        assert domain_digest(key) == 6844561416268996976

    def test_domain_change_changes_digest(self):
        paired = _domain_key({"domain_label": PAIRED_LABEL}, TASK, 0)
        order = _domain_key({"domain_label": ORDER_LABEL}, TASK, 0)
        assert domain_digest(paired) != domain_digest(order)

    def test_pinned_initial_index(self):
        v2 = {"initialization": {"domain_label": PAIRED_LABEL}}
        assert initial_index(v2, TASK, 0, 100) == 76

    def test_all_methods_resolve_the_same_start(self):
        protocol, _curves, _costs = v2_protocol()
        start = initial_index(protocol, TASK, 3, 6)
        order = shared_initial_order(
            n_configurations=6,
            paired_index=start,
            task_id=TASK,
            seed=3,
            permutation_seed=ORDER_LABEL,
        )
        assert order[0] == start
        assert sorted(order) == list(range(6))


    def test_five_bo_seeds_give_five_distinct_permutations(self):
        orders = [
            tuple(
                shared_initial_order(
                    n_configurations=32,
                    paired_index=0,
                    task_id=TASK,
                    seed=seed,
                    permutation_seed=ORDER_LABEL,
                )
            )
            for seed in range(5)
        ]
        assert len(set(orders)) == 5


class TestStructuralCeiling:
    def test_cap_bound_dominates(self):
        curves = np.tile(np.linspace(0.1, 0.9, 5), (4, 1))
        costs = np.full((4, 5), 10.0)
        # 20 finite epochs, but only 3 steps of cost 10 fit under cap 35.
        assert structural_ceiling(curves, costs, 35.0) == 4

    def test_finite_bound_dominates(self):
        curves = np.tile(np.linspace(0.1, 0.9, 3), (2, 1))
        costs = np.full((2, 3), 1.0)
        assert structural_ceiling(curves, costs, 1e9) == 7

    def test_any_accepted_sequence_fits_below_ceiling(self):
        rng = np.random.default_rng(0)
        curves = np.full((5, 6), np.nan)
        costs = np.full((5, 6), np.nan)
        for config in range(5):
            length = int(rng.integers(2, 7))
            curves[config, :length] = rng.uniform(0, 1, length)
            costs[config, :length] = rng.uniform(0.5, 3.0, length)
        cap = 12.0
        ceiling = structural_ceiling(curves, costs, cap)
        finite = [(c, e) for c in range(5) for e in range(6) if np.isfinite(costs[c, e])]
        for _trial in range(50):
            order = rng.permutation(len(finite))
            cumulative = 0.0
            events = 0
            for position in order:
                config, epoch = finite[position]
                step = float(costs[config, epoch])
                if cumulative + step > cap:
                    events += 1  # the single terminal rejection
                    break
                cumulative += step
                events += 1
            assert events <= ceiling


class TestRandomSearchV2:
    def test_paired_start_terminal_rejection_and_determinism(self):
        protocol, curves, costs = v2_protocol()
        ids = [f"cfg{i}" for i in range(6)]
        kwargs = dict(
            protocol=protocol,
            task=TASK,
            bo_seed=2,
            candidate_ids_ordered=ids,
            curves=curves,
            costs=costs,
            cap_seconds=42.0,
            max_iterations=structural_ceiling(curves, costs, 42.0),
        )
        events = random_search_events(**kwargs)
        again = random_search_events(**kwargs)
        assert events == again
        start = initial_index(protocol, TASK, 2, 6)
        assert events[0]["proposed_config_index"] == start
        assert events[0]["proposed_epoch"] == 1
        assert all(event["accepted"] for event in events[:-1])
        assert not events[-1]["accepted"]
        accepted_cost = sum(
            event["table_step_cost_seconds"] for event in events if event["accepted"]
        )
        assert accepted_cost <= 42.0
        assert accepted_cost + events[-1]["table_step_cost_seconds"] > 42.0

