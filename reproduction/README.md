# Training and paper experiments

To use ICICLE for tuning, start with the [main README](../README.md).
Run the commands below from the repository root in a Python 3.12+ environment.

```bash
uv pip install '.[dev,train,export,serving,experiments]'
```

Alternatively, use `python -m pip install` with the same arguments. Install the
JAX accelerator plugin for your hardware separately.

## Train and export

The [release configuration](../configs/icicle.json) uses width 128, 80,000 updates,
batch 128 and bfloat16. Each run saves its resolved configuration beside the
checkpoints; using an existing checkpoint directory resumes that run.

```bash
python -m experiments.train icicle --run.seed 0 --run.checkpoint-dir runs/icicle/ckpt
python -m icicle_jax.export runs/icicle --step 79999 --out models/icicle.safetensors --onnx
```

For a short CPU check, use the `smoke` preset with `--model.dtype float32`
and `JAX_PLATFORMS=cpu`. It runs eight updates. Use `--help` for other settings.

Our appendix FT-PFN-128 models use 33 input positions and the ICICLE prior:

```bash
python -m experiments.train ftpfn-icicle-prior \
  --model.emsize 128 --model.hidden-dim 256 --model.config-dim 33 \
  --optim.peak-lr 0.001 --optim.decay-start 72000 --optim.decay-steps 8000 \
  --run.num-steps 80000 --run.seed 0 \
  --run.checkpoint-dir runs/ftpfn128-s0/ckpt
```

Repeat with seeds 1 and 2 and separate output directories. The inherited settings
use four attention heads, batch 128 and 4,000 warmup updates. Without these
overrides, the preset uses width 512, 10 inputs and 16,000 updates.

The separate `python -m icicle_jax.ftpfn.train` command implements a JAX
approximation of the [published FT-PFN training recipe](../configs/ftpfn_paper.json).
Calibration, curve numerics, query identities, random streams and scheduler stepping
differ from upstream; reproducing the published training run has not been established.
It is separate from both our appendix variants and the published checkpoint below.

### LUMI

ICICLE's `tools/lumi/setup_env.sh` prepares its pinned ROCm/JAX environment;
`tools/lumi/lumi_env.sh` records the image, environment and project paths.
For cluster usage and scheduling, see the [LUMI documentation](https://docs.lumi-supercomputer.eu/).

The evaluation launcher supplies the validated GPU settings and compilation cache:

```bash
ICICLE_EVAL_MODULE=experiments.qtmeta_hpo sbatch --time=02:00:00 \
  tools/lumi/submit_eval.sh --method icicle --model /scratch/path/to/model \
  --split micro --chain-batch 128 --out /scratch/path/to/qtmeta-micro.parquet
```

The launcher also accepts
`ICICLE_EVAL_MODULE=experiments.hpo_regret`; predictive evaluation is the default.

## Prepare benchmark data

```bash
python -m reproduction.prepare.prepare_lcbench --data-dir /path/to/tables
python -m reproduction.prepare.prepare_pd1 --data-dir /path/to/tables
python -m reproduction.prepare.prepare_taskset --data-dir /path/to/tables
python -m reproduction.prepare.verify_benchmarks --data-dir /path/to/tables
```

For static ARLBench and Meta-Album:

```bash
python -m reproduction.prepare.prepare_optional_benchmarks --suite arlbench --data-dir /path/to/tables
python -m reproduction.prepare.prepare_optional_benchmarks --suite qtmeta --data-dir /path/to/tables
python -m reproduction.prepare.verify_benchmarks --data-dir /path/to/tables \
  --manifest benchmarks/optional_inputs.json --suites arlbench_mean qtmeta qtmeta_epoch
```

Preparation verifies download hashes and requires fresh output tables. Use `--archive`
for an existing LCBench/PD1 download, or `--source-dir` for ARLBench/Meta-Album.
Verification checks task lists, input values and ordering against the
[core](../benchmarks/core_inputs.json) and [optional](../benchmarks/optional_inputs.json)
input checks. [Source records](../benchmarks/optional_sources.json) give optional-data versions
and checksums. Dataset use is governed by the publishers' licenses and terms.

## Evaluate

| Benchmark | Paper view | Population |
|---|---|---|
| LCBench | `lcbench_ifbo` | 35 tasks |
| PD1 | `pd1_ifbo` | 28 instances |
| TaskSet | `taskset_ifbo` | 24 instances |
| ARLBench | `arlbench_mean` | 42 static tasks |
| Meta-Album | `qtmeta` | 86 tasks, time-based fidelity |

Evaluate each benchmark and trained model separately. For the paper's core ICICLE comparison:

```bash
python -m experiments.hpo_regret --method icicle --model /path/to/model --step 79999 \
  --benchmark pd1_ifbo --iters 1000 --value-map adaptive \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
  --out results/icicle-pd1.parquet --data-dir /path/to/tables
```

Repeat for the other core views. For a tiny check, use `--iters 4 --seeds 0`.
The command's defaults are three BO seeds and no value map, so keep the explicit
paper settings above. Published FT-PFN uses `--method ftpfn`, its `.safetensors`
checkpoint and `--value-map off`; omit `--step`. Our FT-PFN-128 variants use their
training directories and adaptive mapping. Baselines such as `asha` and
`random_search` need no `--model` or `--step`.

`ICICLE_BENCH_DATA` can replace `--data-dir`. Feature identities use
`--feature-seed 0`; training validation has a separate `--run.validation-seed 0`.

### Meta-Album: 1/4/16-hour budgets

The runner charges recorded training plus evaluation time and rejects the first
over-budget proposal without observing its score. It defaults to BO seeds 0–19,
adaptive mapping and cost-aware MFPI.

```bash
python -m experiments.qtmeta_hpo --method icicle --model /path/to/model --step 79999 \
  --split micro --chain-batch 128 \
  --out results/icicle-micro.parquet --data-dir /path/to/tables
```

Run mini and extended with `--chain-batch 32` and separate output files. Our
FT-PFN-128 variants use batch 32 on all splits. Batches group independent BO runs;
each run retains its own budget. Keep checkpoint and batch size fixed within a split.
For smaller GPUs, use smaller batches and record the setting. Compilation and warm
runtime are distinct from the table-time budgets plotted in the paper.

The command also supports `asha`, `random_search` and retrained `ftpfn` models.
`--seeds` selects disjoint seed shards and `--tasks` selects tasks. Keep trained models
in separate outputs. Native Quick-Tune runs through its external implementation;
this runner covers the other methods.

### Published FT-PFN

Convert the published checkpoint for the JAX evaluator. First, prepare its weights
in a separate CPU PyTorch environment:

```bash
python3.12 -m venv .reference-venv
.reference-venv/bin/python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.reference-venv/bin/python -m pip install numpy==1.26.4 scipy==1.14.1 requests==2.34.0
.reference-venv/bin/python -m reproduction.prepare.prepare_ftpfn_reference --out ./reference-preparation
```

Then convert and check them in the ICICLE environment:

```bash
python -m icicle_jax.ftpfn.weights reference-preparation/state_dict.npz reference-preparation/ftpfn_v0.0.1.safetensors
ICICLE_FTPFN_WEIGHTS="$PWD/reference-preparation/ftpfn_v0.0.1.safetensors" python -m pytest tests/icicle_jax/ftpfn/test_ftpfn_parity.py
```

Preparation downloads the pinned upstream model and checks its weights and
predictions against the [reference specification](../benchmarks/ftpfn_reference.json).
See [third-party notices](../THIRD_PARTY_NOTICES) for licensing and attribution.

## Paper figures

Aggregate the explicit inputs once, then render from the resulting cache:

```bash
python -m reproduction.figures.agg_figure_cache --paper \
  --manifest /path/to/inputs.json --out /path/to/cache --data-dir /path/to/tables
python -m reproduction.figures.render_regret \
  --cache /path/to/cache/paper-statistics.json --output-dir /path/to/figures
```

For appendix task cards, set `ICICLE_FIGCACHE=/path/to/cache` and run the appropriate
module in `reproduction.figures`: `render_ftpfn_task_cards`, `render_qtmeta_task_appendix`,
`render_arlbench_mean` or `render_arlbench_task_cards`, with `--output-dir /path/to/figures`.

The result bundle provides `inputs.json` (format `icicle-paper-inputs-v1`):

- `hpo` maps benchmark views to model entries; `qtmeta` maps model names to entries.
- Parquet entries specify `paths`, `training_seed` and `settings` identifying the
  method, checkpoint, step, value map and numerical settings. ICICLE entries also
  need `model_config`, pointing to the bundled training configuration.
- Native Quick-Tune entries specify `training_seed` and `native_root`; files are named
  `<task-with-slashes-replaced-by-__>__b<BO-seed>.json.gz`.
- For split-specific batches, use `chain_batch_by_split`, such as
  `{"micro": 128, "mini": 32, "extended": 32}`. Run from the bundle root for relative paths.

Aggregation checks model identity, numerical settings, complete task coverage and
BO seeds 0–19. Main curves use one specified trained model; training-seed comparisons
remain separate. Ribbons show pointwise 95% bootstrap intervals, resampling tasks and then BO runs
20,000 times. Figure 2 is linear and Figure 3 logarithmic. Render from existing CPU cores;
on LUMI, heavy aggregation needs a bounded CPU job. Rerendering needs no reaggregation.

For predictive evaluations, use `python -m experiments.eval --help`. Average
feature-identity draws within each scenario before comparing model predictions.
