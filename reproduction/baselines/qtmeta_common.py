"""Shared protocol, seeding and cost-accounting utilities for Quick-Tune comparisons."""

import hashlib
from pathlib import Path

import numpy as np

from reproduction.bench.task import valid_lengths


def _domain_key(block: dict, task: str, bo_seed: int) -> bytes:
    return f"{block['domain_label']}|{task}|{bo_seed}".encode()


def domain_digest(key: bytes) -> int:
    """First 8 bytes of SHA-256 over the frozen key encoding, big-endian."""
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def shared_initial_order(
    *,
    n_configurations: int,
    paired_index: int,
    task_id: str,
    seed: int,
    permutation_seed: str,
) -> np.ndarray:
    """Put the paired start first, then shuffle the remaining configurations."""
    if not 0 <= paired_index < n_configurations:
        raise ValueError("paired initial index is out of range")
    key = f"{permutation_seed}|{task_id}|{seed}".encode()
    rng_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    remaining = np.delete(np.arange(n_configurations, dtype=np.int64), paired_index)
    return np.concatenate(
        [
            np.asarray([paired_index], dtype=np.int64),
            np.random.default_rng(rng_seed).permutation(remaining),
        ]
    )


def initial_index(protocol: dict, task: str, bo_seed: int, n_candidates: int) -> int:
    """Resolve the task/seed start without consulting any objective value."""
    key = _domain_key(protocol["initialization"], task, bo_seed)
    return domain_digest(key) % n_candidates


def task_bounds(curves: np.ndarray) -> tuple[float, float]:
    """Whole-table normalization anchors in higher-is-better objective units."""
    return float(np.nanmin(curves)), float(np.nanmax(curves))


def normalized_regret(incumbent: float, worst: float, best: float) -> float:
    """Whole-table normalized regret, clipped only for numerical residue."""
    denominator = best - worst
    if not denominator > 0:
        raise ValueError(f"nonpositive normalization denominator {denominator}")
    return float(np.clip((best - incumbent) / denominator, 0.0, 1.0))


def validate_table_task(curves: np.ndarray, costs: np.ndarray) -> np.ndarray:
    """Validate ragged-prefix objectives/costs and return finite prefix lengths."""
    if curves.shape != costs.shape:
        raise ValueError(f"curve/cost shape mismatch: {curves.shape} != {costs.shape}")
    lengths = valid_lengths(curves)
    valid = np.isfinite(curves)
    if not np.array_equal(valid, np.isfinite(costs)) or np.isinf(costs).any():
        raise ValueError("objective and cost masks differ")
    if (costs[valid] < 0).any():
        raise ValueError("table costs must be nonnegative")
    return lengths


def random_search_events(
    *,
    protocol: dict,
    task: str,
    bo_seed: int,
    candidate_ids_ordered: list[str],
    curves: np.ndarray,
    costs: np.ndarray,
    cap_seconds: float,
    max_iterations: int,
    allow_truncation: bool = False,
) -> list[dict]:
    """Paired random search: shared start, then a seeded permutation of remaining configs."""
    lengths = validate_table_task(curves, costs)
    first = initial_index(protocol, task, bo_seed, len(candidate_ids_ordered))
    order = shared_initial_order(
        n_configurations=len(candidate_ids_ordered), paired_index=first,
        task_id=task, seed=bo_seed,
        permutation_seed=protocol["random_search"]["domain_label"],
    )
    worst, best = task_bounds(curves)
    incumbent = worst
    cumulative = 0.0
    events = []
    iteration = 0
    for config in order:
        for level in range(int(lengths[config])):
            if iteration >= max_iterations:
                if allow_truncation:
                    return events
                raise RuntimeError(f"{task}: random search exhausted max_iterations")
            step_cost = float(costs[config, level])
            accepted = cumulative + step_cost <= cap_seconds
            objective = float(curves[config, level])
            if accepted:
                cumulative += step_cost
                incumbent = max(incumbent, objective)
            events.append(
                {
                    "iteration": iteration,
                    "proposed_config_index": int(config),
                    "proposed_config_id": candidate_ids_ordered[config],
                    "proposed_epoch": level + 1,
                    "proposed_objective": objective,
                    "table_step_cost_seconds": step_cost,
                    "cumulative_table_cost_seconds": cumulative,
                    "accepted": accepted,
                    "incumbent": incumbent,
                    "normalized_regret": normalized_regret(incumbent, worst, best),
                    "failure": False,
                    "failure_reason": None,
                }
            )
            iteration += 1
            if not accepted:
                return events
    return events


def structural_ceiling(curves: np.ndarray, costs: np.ndarray, cap_seconds: float) -> int:
    """Bound the number of accepted transitions plus one rejected proposal.

    Each transition consumes a distinct finite epoch. The count is bounded by
    both the table size and the number of cheapest steps fitting within the cap.
    """
    lengths = validate_table_task(curves, costs)
    finite = int(lengths.sum())
    step_costs = np.sort(costs[np.isfinite(costs)])
    k_max = int(np.searchsorted(np.cumsum(step_costs), cap_seconds, side="right"))
    return 1 + min(finite, k_max)


def content_sha256(path: str | Path) -> str:
    """SHA-256 of one artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
