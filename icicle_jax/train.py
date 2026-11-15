"""Training recipes, prior sampling and optimization."""

import dataclasses
import json
import time
from contextlib import ExitStack
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax.sharding import AxisType, NamedSharding, PartitionSpec

from icicle_jax import checkpoint
from icicle_jax.bar_distribution import BarDistribution
from icicle_jax.loading import ModelConfig, build_model
from icicle_jax.nn import ragged
from icicle_jax.prior.bnn import build_output_sorted
from icicle_jax.prior.observations import parse_observations
from icicle_jax.prior.task import sample_task, truncated_poisson

# Validation tasks use a separate random stream and a fixed chunk size.
VAL_CHUNK = 32  # validation batch size; divisible by the device count

# Use four coarse ragged-buffer bounds plus a 2,048-aligned minimum.
# Each bound requires a separate compiled program; limiting their number reduces
# compilation overhead and host memory-map usage.
N_PAD_COARSE_BUCKETS = 4
RAGGED_TILE_ROWS = 2048


@dataclasses.dataclass(frozen=True)
class PriorConfig:
    layers: tuple[int, int] = (8, 15)  # BNN depth range (reference)
    hidden: tuple[int, int] = (36, 149)  # BNN width range (reference)
    table_networks: int = 500  # calibration table: distinct BNNs
    table_configs: int = 64  # configurations per calibration BNN
    # "none", "all", or comma-separated names from prior.observations.FLAG_NAMES.
    extensions: str = "none"
    # Probability that each observed step is eligible for query sampling.
    # Repeated observations share their value, including noise.
    recall_probability: float = 0.0
    # Range of the per-task noise multiplier when noise_tails is enabled.
    tail_scale: tuple[float, float] = (0.25, 4.0)
    # Sample continuous, ordinal and nominal hyperparameter types.
    hp_types_poisson: bool = False
    # Draw num_params = 1 + TruncPoisson(Lambda), bounded by config_dim,
    # with Lambda ~ LogUniform(1, 20).
    arity_trunc_poisson: bool = False


@dataclasses.dataclass(frozen=True)
class DataConfig:
    seq_len: int = 1000
    batch_size: int = 32
    n_steps_max: int = 1000  # max fidelity-grid size


@dataclasses.dataclass(frozen=True)
class OptimConfig:
    peak_lr: float = 1e-3
    warmup_steps: int = 500
    # Absolute step at which WSD decay begins; None holds the peak learning rate.
    decay_start: int | None = None
    decay_steps: int = 5_000
    grad_clip: float = 1.0
    beta2: float = 0.999  # Adam second-moment decay


@dataclasses.dataclass(frozen=True)
class RunConfig:
    num_steps: int = 60_000
    print_every: int = 100
    val_every: int = 1_000
    val_batches: int = 4  # VAL_CHUNK tasks per batch
    seed: int = 0
    validation_seed: int = 0
    checkpoint_dir: str | None = None  # None -> no checkpointing; else save (and resume) here
    checkpoint_every: int = 5_000
    max_checkpoints: int | None = None  # None keeps every checkpoint


@dataclasses.dataclass(frozen=True)
class Experiment:
    model: ModelConfig
    optim: OptimConfig
    prior: PriorConfig
    data: DataConfig
    run: RunConfig


def load_recipe(path: Path) -> Experiment:
    groups = json.loads(path.read_text())
    config_types = {
        "model": ModelConfig,
        "optim": OptimConfig,
        "prior": PriorConfig,
        "data": DataConfig,
        "run": RunConfig,
    }
    if groups.keys() != config_types.keys():
        raise ValueError("training config must contain model, optim, prior, data and run")

    configs = {}
    for name, cls in config_types.items():
        values = dict(groups[name])
        for field in dataclasses.fields(cls):
            if field.name in values and isinstance(field.default, tuple):
                values[field.name] = tuple(values[field.name])
        configs[name] = cls(**values)
    return Experiment(**configs)


def wsd_schedule(peak_lr, warmup_steps, decay_start, decay_steps):
    """Warm up to peak_lr, hold until decay_start, then decay linearly.

    decay_start is an absolute step. None keeps the learning rate at its peak.
    """
    warm = optax.linear_schedule(0.0, peak_lr, warmup_steps)
    if decay_start is None:
        return optax.join_schedules([warm, optax.constant_schedule(peak_lr)], [warmup_steps])
    return optax.join_schedules(
        [
            warm,
            optax.constant_schedule(peak_lr),
            optax.linear_schedule(peak_lr, 0.0, decay_steps),
        ],
        [warmup_steps, decay_start],
    )


def make_optimizer(schedule, grad_clip, beta2=0.999):
    """Clip gradients by global norm, then apply Adam."""
    return optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(schedule, b2=beta2),
    )


def query_nll(criterion, logits, y, single_eval_pos):
    """Mean bar NLL over query positions (>= single_eval_pos).

    ``criterion.nll`` already zeroes NaN targets, so we divide by the count of valid (query,
    non-NaN) targets to keep the mean a true per-token average.
    """
    nll = criterion.nll(logits.astype(jnp.float32), y)  # (B, P); NaN targets contribute 0
    is_query = jnp.arange(logits.shape[1]) >= single_eval_pos
    valid = is_query[None, :] & ~jnp.isnan(y)
    total = jnp.where(is_query[None, :], nll, 0.0).sum()
    return total / jnp.maximum(valid.sum(), 1)


def _draw_step_params(key, model_cfg, data_cfg, prior_cfg):
    """Draw the shared HP count, fidelity-grid size and context split for a batch."""
    k_arity, k_steps, k_split, k_rate = jax.random.split(key, 4)
    if prior_cfg.arity_trunc_poisson:
        rate = jnp.exp(jax.random.uniform(k_rate, minval=jnp.log(1.0), maxval=jnp.log(20.0)))
        num_params = 1 + truncated_poisson(k_arity, rate, model_cfg.config_dim - 1).astype(jnp.int32)
    else:
        num_params = jax.random.randint(k_arity, (), 1, model_cfg.config_dim + 1)
    exponent = jax.random.uniform(k_steps, (), minval=0.0, maxval=float(np.log10(data_cfg.n_steps_max)))
    n_steps = jnp.clip(jnp.round(10.0**exponent).astype(jnp.int32), 1, data_cfg.n_steps_max)
    single_eval_pos = jax.random.randint(k_split, (), 1, data_cfg.seq_len)
    return num_params, n_steps, single_eval_pos


def _print_now(*args):
    print(*args, flush=True)


def resolve_prior(prior_cfg, log=_print_now):
    """Resolve the observation-model options."""
    observation_config = parse_observations(prior_cfg.extensions)
    if tuple(prior_cfg.tail_scale) != tuple(PriorConfig.tail_scale) and not observation_config.noise_tails:
        raise ValueError("prior.tail_scale only takes effect with the 'noise_tails' flag")
    observation_config = dataclasses.replace(observation_config, tail_scale=tuple(prior_cfg.tail_scale))
    log(f"prior: extensions: {prior_cfg.extensions}")

    return observation_config


def build_table(model_cfg, prior_cfg, log=_print_now):
    """Build the empirical output distribution used to sample curve parameters."""
    log(f"building calibration table (BNN depth {prior_cfg.layers})...")
    t0 = time.time()
    table = build_output_sorted(
        jax.random.key(0),
        config_dim=model_cfg.config_dim,
        layers=prior_cfg.layers,
        hidden=prior_cfg.hidden,
        num_networks=prior_cfg.table_networks,
        configs_per_network=prior_cfg.table_configs,
    )
    log(f"  table: {table.size:,} samples in {time.time() - t0:.1f}s")
    return table


def make_samplers(model_cfg, prior_cfg, data_cfg, *, table, observation_config):
    """Build training and validation samplers. Validation queries unobserved steps."""
    task_options = dict(
        seq_len=data_cfg.seq_len,
        config_dim=model_cfg.config_dim,
        table=table,
        layers=prior_cfg.layers,
        hidden=prior_cfg.hidden,
        n_steps_max=data_cfg.n_steps_max,
        observation_config=observation_config,
        typed_hyperparameters=prior_cfg.hp_types_poisson,
    )

    def make_sampler(batch_size, recall_probability):
        @jax.jit
        def sample(step_key, num_params, n_steps, single_eval_pos):
            keys = jax.random.split(jax.random.fold_in(step_key, 0), batch_size)
            return jax.vmap(lambda key: sample_task(
                key, num_params=num_params, n_steps=n_steps,
                single_eval_pos=single_eval_pos, recall_probability=recall_probability,
                **task_options,
            ))(keys)

        return sample

    return (
        make_sampler(data_cfg.batch_size, prior_cfg.recall_probability),
        make_sampler(VAL_CHUNK, 0.0),
    )


def ragged_bucket_bounds(capacity: int) -> tuple[int, ...]:
    """Return the five measured static bounds used by the ragged config encoder."""
    if capacity < 1:
        raise ValueError("ragged capacity must be positive")
    quantum = max(capacity // N_PAD_COARSE_BUCKETS, 1)
    bounds = [min(capacity, quantum * index) for index in range(1, N_PAD_COARSE_BUCKETS + 1)]
    bounds[-1] = capacity
    target = -(-capacity // (2 * N_PAD_COARSE_BUCKETS))
    fine_min = -(-target // RAGGED_TILE_ROWS) * RAGGED_TILE_ROWS
    bounds.append(min(capacity, max(fine_min, 1)))
    return tuple(sorted(set(bounds)))


count_referenced_configs = jax.jit(ragged.count_referenced_configs, static_argnums=(1, 2))


def batch_buffer_size(batch, count, groups):
    """Choose a compiled buffer size large enough for each device's configurations."""
    capacity = (batch.y.shape[0] // groups) * batch.config_hps.shape[1]
    return next(bound for bound in ragged_bucket_bounds(capacity) if count <= bound)


def make_train_functions(criterion, n_dev=1):
    """Compile training and validation steps."""
    def loss_of(model, batch, feature_key, num_params, sep, n_pad):
        args = (batch.config_hps, num_params, batch.point_config_idx, batch.fidelities, batch.y, sep)
        if n_pad:
            logits = model.forward_ragged(*args, n_pad, feature_key=feature_key, groups=n_dev)
        else:
            logits = model(*args, feature_key=feature_key)
        return query_nll(criterion, logits, batch.y, sep)

    @partial(nnx.jit, static_argnames="n_pad")
    def train_step(model, optimizer, batch, feature_key, num_params, sep, n_pad):
        loss, grads = nnx.value_and_grad(
            lambda m: loss_of(m, batch, feature_key, num_params, sep, n_pad)
        )(model)
        grad_norm = optax.global_norm(grads)  # before clipping
        optimizer.update(model, grads)
        return loss, grad_norm

    @partial(nnx.jit, static_argnames="n_pad")
    def val_step(model, batch, feature_key, num_params, sep, n_pad):
        return loss_of(model, batch, feature_key, num_params, sep, n_pad)

    return train_step, val_step


def train(model_cfg, optim_cfg, prior_cfg, data_cfg, run_cfg, *, log=_print_now):
    """Train ICICLE or an FT-PFN ablation with WSD and the shared batch sampler."""
    schedule = wsd_schedule(
        optim_cfg.peak_lr, optim_cfg.warmup_steps, optim_cfg.decay_start, optim_cfg.decay_steps
    )
    return fit(
        model_cfg, optim_cfg, prior_cfg, data_cfg, run_cfg,
        schedule=schedule, sampler_factory=make_samplers, batch_parameters=_draw_step_params,
        trainer="common", log=log,
    )


def check_resume_config(saved, requested):
    """Require the same scientific recipe when continuing an existing run."""
    operational = {"print_every", "val_every", "val_batches", "checkpoint_dir",
                   "checkpoint_every", "max_checkpoints"}
    if requested["trainer"] == "common":
        operational.add("num_steps")  # WSD uses absolute decay steps.

    def recipe(config):
        result = {name: config[name] for name in ("model", "optim", "prior", "data", "run")}
        result["run"] = {k: v for k, v in result["run"].items() if k not in operational}
        result["trainer"] = config.get("trainer", "common")
        result["icl_backend_resolved"] = config.get("icl_backend_resolved")
        return result

    old, new = recipe(saved), recipe(requested)
    changed = [name for name in old if old[name] != new[name]]
    if changed:
        raise ValueError(f"resume changes {', '.join(changed)}; use a new run directory")


def fit(
    model_cfg, optim_cfg, prior_cfg, data_cfg, run_cfg, *,
    schedule, sampler_factory, batch_parameters, trainer, log=_print_now,
):
    """Shared optimization, validation and checkpoint loop for the two training commands."""
    if model_cfg.architecture == "ftpfn" and data_cfg.seq_len > model_cfg.ftpfn_max_curves:
        raise ValueError("FT-PFN training config table exceeds its curve vocabulary")
    if jax.process_count() != 1:
        raise ValueError("training supports one process with one or more local devices")
    device_count = jax.device_count()
    if data_cfg.batch_size % device_count:
        raise ValueError(f"training batch {data_cfg.batch_size} must divide {device_count} devices")
    if VAL_CHUNK % device_count:
        raise ValueError(f"validation batch {VAL_CHUNK} must divide {device_count} devices")
    observation_config = resolve_prior(prior_cfg, log)
    table = build_table(model_cfg, prior_cfg, log)

    model = build_model(model_cfg, nnx.Rngs(run_cfg.seed))
    backend_resolved = (
        model.icl_layers[0].attn.backend if model_cfg.architecture == "icicle" else "xla"
    )
    log(f"icl_backend: {model_cfg.icl_backend} -> {backend_resolved}")
    tx = make_optimizer(schedule, optim_cfg.grad_clip, optim_cfg.beta2)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # Auto sharding handles the model's gathers and scatters. Place state before
    # restoring so Orbax uses the mesh layout for checkpoint arrays.
    mesh = jax.make_mesh((device_count,), ("data",), axis_types=(AxisType.Auto,))
    batch_sharding = NamedSharding(mesh, PartitionSpec("data"))
    replicated = NamedSharding(mesh, PartitionSpec())
    graphdef, state = nnx.split((model, optimizer))
    state = jax.device_put(state, replicated)
    model, optimizer = nnx.merge(graphdef, state)
    criterion = BarDistribution(
        borders=jax.device_put(jnp.linspace(0.0, 1.0, model_cfg.num_bars + 1), replicated)
    )
    if device_count > 1:
        log(f"data-parallel over {device_count} local devices")

    with ExitStack() as resources:
        manager = (
            resources.enter_context(checkpoint.make_manager(
                run_cfg.checkpoint_dir, max_to_keep=run_cfg.max_checkpoints
            )) if run_cfg.checkpoint_dir else None
        )
        metrics_file = None
        start_step = 0

        def metric(step, name, value):
            if metrics_file is not None:
                metrics_file.write(f"{step},{name},{value}\n")
                metrics_file.flush()

        if manager is not None:
            run_dir = Path(run_cfg.checkpoint_dir).parent
            config_path = run_dir / "config.json"
            config = {
                **{name: dataclasses.asdict(value) for name, value in (
                    ("model", model_cfg), ("optim", optim_cfg), ("prior", prior_cfg),
                    ("data", data_cfg), ("run", run_cfg),
                )},
                "trainer": trainer,
                "icl_backend_resolved": backend_resolved,
            }
            # JSON normalizes tuple-valued ranges in both configurations.
            config = json.loads(json.dumps(config))
            if manager.latest_step() is not None:
                check_resume_config(json.loads(config_path.read_text()), config)
                start_step = checkpoint.restore_latest(manager, model, optimizer)
                log(f"resuming from step {start_step}")
            else:
                config_path.write_text(json.dumps(config, indent=2) + "\n")

            metrics_path = run_dir / "metrics.csv"
            header = "step,metric,value\n"
            if start_step and metrics_path.exists():
                rows = metrics_path.read_text().splitlines(keepends=True)[1:]
                kept = [row for row in rows if int(row.split(",", 1)[0]) < start_step]
                metrics_path.write_text(header + "".join(kept))
            else:
                metrics_path.write_text(header)
            metrics_file = resources.enter_context(metrics_path.open("a"))

        sample_fn, val_sample_fn = sampler_factory(
            model_cfg,
            prior_cfg,
            data_cfg,
            table=table,
            observation_config=observation_config,
        )

        def sharded_batch(step_key, num_params, n_steps, sep, fn):
            # Explicit placement keeps the sampler's table lookups outside auto sharding.
            batch = fn(step_key, num_params, n_steps, sep)
            return jax.device_put(batch, batch_sharding)

        train_step, val_step = make_train_functions(criterion, device_count)

        key = jax.random.key(run_cfg.seed)
        # Separate validation from training even when both seeds are zero.
        validation_key = jax.random.fold_in(jax.random.key(run_cfg.validation_seed), 1)
        t0 = time.time()
        last_print = (start_step, 0.0)
        window: list = []  # per-step loss and gradient norm
        last_n_pad = None
        seen_n_pad = set()

        def prepare(step):
            # Start sampling and counting before the next training step needs the batch.
            k = jax.random.fold_in(key, step)
            num_params, n_steps, sep = batch_parameters(k, model_cfg, data_cfg, prior_cfg)
            batch = sharded_batch(k, num_params, n_steps, sep, sample_fn)
            count = (
                count_referenced_configs(batch.point_config_idx, data_cfg.seq_len, device_count)
                if model_cfg.ragged else None
            )
            return k, num_params, sep, batch, count

        pending = prepare(start_step)
        for step in range(start_step, run_cfg.num_steps):
            is_final = step == run_cfg.num_steps - 1
            k, num_params, sep, batch, count = pending
            if not is_final:
                pending = prepare(step + 1)
            n_pad = batch_buffer_size(batch, int(count), device_count) if model_cfg.ragged else 0
            if n_pad != last_n_pad:
                # Previously seen buckets reuse their cached executable.
                event = "compile" if n_pad not in seen_n_pad else "bucket"
                log(f"  [{event}] step {step:6d}  n_pad bucket -> {n_pad}")
                metric(step, "n_pad", n_pad)
                seen_n_pad.add(n_pad)
                last_n_pad = n_pad
            loss, grad_norm = train_step(
                model, optimizer, batch, jax.random.fold_in(k, 1), num_params, sep, n_pad=n_pad
            )
            window.append((loss, grad_norm))
            if step % run_cfg.print_every == 0 or is_final:
                w = np.asarray(jax.device_get(window))
                elapsed = time.time() - t0
                loss_value, norm_value = w[-1]
                learning_rate = float(schedule(step))
                window.clear()
                gmax, clip_frac = w[:, 1].max(), float((w[:, 1] > optim_cfg.grad_clip).mean())
                log(
                    f"step {step:6d}  loss {loss_value:8.4f}  gnorm {norm_value:8.2f}  "
                    f"gmax {gmax:8.2f}  clip {clip_frac:4.0%}  "
                    f"lr {learning_rate:.2e}  t={elapsed:.1f}s"
                )
                metric(step, "loss", loss_value)
                metric(step, "gnorm", norm_value)
                metric(step, "loss_mean", float(w[:, 0].mean()))
                metric(step, "gnorm_mean", float(w[:, 1].mean()))
                metric(step, "gnorm_max", float(gmax))
                metric(step, "clip_frac", clip_frac)
                metric(step, "lr", learning_rate)
                if step > last_print[0]:
                    rate = (step - last_print[0]) / max(elapsed - last_print[1], 1e-9)
                    metric(step, "steps_per_sec", rate)
                last_print = (step, elapsed)
            if step % run_cfg.val_every == 0 or is_final:
                vloss = []
                for j in range(run_cfg.val_batches):
                    kj = jax.random.fold_in(validation_key, j)
                    vnp, vns, vsep = batch_parameters(kj, model_cfg, data_cfg, prior_cfg)
                    vbatch = sharded_batch(kj, vnp, vns, vsep, fn=val_sample_fn)
                    vn_pad = 0
                    if model_cfg.ragged:
                        count = count_referenced_configs(
                            vbatch.point_config_idx, data_cfg.seq_len, device_count
                        )
                        vn_pad = batch_buffer_size(vbatch, int(count), device_count)
                    vloss.append(
                        float(
                            val_step(model, vbatch, jax.random.fold_in(kj, 1), vnp, vsep, n_pad=vn_pad)
                        )
                    )
                log(f"  [val] step {step:6d}  nll {np.mean(vloss):8.4f}")
                metric(step, "val_nll", float(np.mean(vloss)))
                metric(step, "val_nll_std", float(np.std(vloss)))
                metric(step, "param_norm", float(optax.global_norm(nnx.state(model, nnx.Param))))
            # State after this step reflects steps [start_step, step]; restore resumes at step+1.
            if manager and (is_final or (step > 0 and step % run_cfg.checkpoint_every == 0)):
                checkpoint.save(manager, step, model, optimizer)

    log("done.")
    return model
