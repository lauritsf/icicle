"""Save and restore model and optimizer state with Orbax.

Checkpoints use step-numbered directories. Task sampling derives its key from
(seed, step), so no separate RNG state is needed. Checkpoint frequency is
independent of validation frequency.
"""

from pathlib import Path

import orbax.checkpoint as ocp
from flax import nnx


def make_manager(directory: str, *, max_to_keep: int | None = None) -> ocp.CheckpointManager:
    """Create the checkpoint directory and manager; retain all steps by default."""
    path = Path(directory).absolute()
    path.mkdir(parents=True, exist_ok=True)
    options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep)
    return ocp.CheckpointManager(path, options=options)


def _payload(model: nnx.Module, optimizer: nnx.Optimizer) -> dict:
    return {"model": nnx.state(model), "optimizer": nnx.state(optimizer)}


def save(manager: ocp.CheckpointManager, step: int, model, optimizer) -> None:
    """Save model + optimizer state at ``step`` (async; the manager flushes on close/wait)."""
    manager.save(step, args=ocp.args.StandardSave(_payload(model, optimizer)))


def restore_latest(manager: ocp.CheckpointManager, model, optimizer) -> int:
    """Restore the latest training state and return the next step, or zero if absent."""
    step = manager.latest_step()
    if step is None:
        return 0
    restored = manager.restore(step, args=ocp.args.StandardRestore(_payload(model, optimizer)))
    nnx.update(model, restored["model"])
    nnx.update(optimizer, restored["optimizer"])
    return step + 1


def restore_model(directory: str, model, *, step=None) -> int | None:
    """Load model state and return its saved step, or None if no checkpoint exists."""
    path = Path(directory).absolute()
    if not path.exists():
        return None
    options = ocp.CheckpointManagerOptions(read_only=True)
    with ocp.CheckpointManager(
        path, item_handlers=ocp.PyTreeCheckpointHandler(), options=options
    ) as manager:
        if step is not None and step not in manager.all_steps():
            raise ValueError(f"no checkpoint at step {step}; saved: {sorted(manager.all_steps())}")
        step = manager.latest_step() if step is None else step
        if step is None:
            return None
        target = {"model": nnx.state(model)}
        restored = manager.restore(
            step, args=ocp.args.PyTreeRestore(
                item=target, partial_restore=True,
                restore_args=ocp.checkpoint_utils.construct_restore_args(target),
            )
        )
        nnx.update(model, restored["model"])
        return step
