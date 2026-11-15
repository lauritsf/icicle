"""Pack referenced configurations before encoding them.

Each task may use only part of its configuration table. Packing those rows into
a fixed-size buffer avoids encoding unused rows while preserving the mapping
from points to configurations.

n_pad is a static upper bound on the referenced-row count. Callers must choose
a large enough buffer; training uses a fixed set of size buckets to limit
recompilation.
"""

import jax
import jax.numpy as jnp


def next_pow2(x: int) -> int:
    """Smallest power of two >= x (>= 1)."""
    return 1 if x <= 1 else 1 << (int(x) - 1).bit_length()


def count_referenced_configs(point_config_idx: jax.Array, num_configs: int, groups: int = 1):
    """Largest referenced-configuration count across equal device groups."""
    b = point_config_idx.shape[0]
    rows = jnp.arange(b)[:, None]
    ref = jnp.zeros((b, num_configs), bool).at[rows, point_config_idx].set(True)
    per_group = ref.sum(axis=1).reshape(groups, -1).sum(axis=1)
    return per_group.max()


def pack_referenced_configs(config_hps, point_config_idx, n_pad):
    """Pack the distinct referenced configs across the batch into one flat buffer.

    Args:
        config_hps: ``(B, C, H)`` config table.
        point_config_idx: ``(B, P)`` config id per point, in ``[0, C)``.
        n_pad: Static flat-buffer size bounding ``sum_b C_distinct``.

    Returns:
        ``flat_hps (n_pad, H)``, ``flat_task (n_pad,)`` (which task owns each slot, for the
        per-task feature identity), ``point_slot (B, P)`` (each point's slot in the flat buffer),
        ``valid (n_pad,)`` (real slots vs trailing padding).
    """
    b, c, h = config_hps.shape
    rows = jnp.arange(b)[:, None]

    ref = jnp.zeros((b, c), bool).at[rows, point_config_idx].set(True)
    relabel = jnp.cumsum(ref, axis=1) - 1  # rank among a task's referenced (valid where ref)
    n_b = ref.sum(axis=1)  # (B,) distinct per task
    offset = jnp.cumsum(n_b) - n_b  # (B,) exclusive prefix sum -> task's block start

    slot = offset[:, None] + relabel  # (B, C) flat slot for referenced (b, c)
    # Scatter unused rows to an extra slot that is then discarded. The index
    # n_pad must fit all referenced rows.
    keep = ref & (slot < n_pad)
    slot_full = jnp.where(keep, slot, n_pad).reshape(-1)  # (B*C,)
    task_grid = jnp.broadcast_to(rows, (b, c)).reshape(-1)

    flat_hps = (
        jnp.zeros((n_pad + 1, h), config_hps.dtype)
        .at[slot_full]
        .set(config_hps.reshape(-1, h))[:n_pad]
    )
    flat_task = (
        jnp.zeros((n_pad + 1,), jnp.int32).at[slot_full].set(task_grid.astype(jnp.int32))[:n_pad]
    )

    point_slot = offset[:, None] + jnp.take_along_axis(relabel, point_config_idx, axis=1)
    point_slot = jnp.minimum(point_slot, n_pad - 1)  # points always reference a kept config
    valid = jnp.arange(n_pad) < n_b.sum()
    return flat_hps, flat_task, point_slot, valid
