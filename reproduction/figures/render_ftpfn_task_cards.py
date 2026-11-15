"""Render task-level final regret for fixed checkpoints and 20 BO seeds."""

import argparse
import json
import textwrap
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from reproduction.baselines.qtmeta_common import content_sha256  # noqa: E402
from reproduction.figures.figcache import final_task_runs, paper_statistics
from reproduction.figures.render_regret import SUITES, style  # noqa: E402

CARD_SERIES = tuple(
    (method, style(key, True)["label"], None, style(key, True)["color"], "-", 1.5)
    for method, key in [
        ("adaptive", "icicle-s0"),
        ("ftpfn", "ftpfn-published"),
        ("asha", "asha"),
        ("random_search", "random_search"),
    ]
)
METHODS = tuple(key for key, *_rest in CARD_SERIES)
METHOD_STYLE = {
    key: {"label": label, "color": color, "marker": marker}
    for (key, label, _tag, color, _linestyle, _width), marker in zip(
        CARD_SERIES, ("o", "s", "^", "D"), strict=True
    )
}
EXPECTED_TASKS = {"lcbench_ifbo": 35, "pd1_ifbo": 28, "taskset_ifbo": 24}
LAYOUT = {
    "lcbench_ifbo": (7, 5, (5.5, 7.4)),
    "pd1_ifbo": (7, 4, (5.5, 7.2)),
    "taskset_ifbo": (6, 4, (5.5, 6.4)),
}
BO_JITTER = np.linspace(-0.09, 0.09, 20)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"FT-PFN task-card gate failed: {message}")


def compact_task_label(suite: str, task: str) -> str:
    """Return a compact but identifying dataset/task label for a card title."""
    label = task.split("/", 1)[1]
    if suite == "lcbench_ifbo":
        return textwrap.fill(label.replace("_", " "), width=17, max_lines=2, placeholder="…")
    if suite == "pd1_ifbo":
        replacements = {
            "fashion_mnist": "Fashion",
            "svhn_no_extra": "SVHN",
            "translate_wmt": "WMT",
            "wide_resnet": "WRN",
            "simple_cnn": "CNN",
            "xformer_translate": "XFormer",
            "transformer": "Transformer",
        }
        for source, target in replacements.items():
            label = label.replace(source, target)
        return textwrap.fill(label, width=20, max_lines=2, placeholder="…")
    label = label.split("imdb_", 1)[1].replace("_wide_grid_1k", "")
    body, adam = label.rsplit("-adam", 1)
    replacements = {
        "patch": "p",
        "embed": "E",
        "relu": "R",
        "tanh": "T",
        "avg": "A",
        "last": "L",
        "bs": "b",
    }
    for source, target in replacements.items():
        body = body.replace(source, target)
    return f"{body.replace('_', '-')}\nAdam{adam}"


def bo_conditioned_final(suite: str):
    statistics = paper_statistics()
    methods = dict(
        adaptive="icicle-s0", ftpfn="ftpfn-published", asha="asha", random_search="random_search"
    )
    return pd.DataFrame(final_task_runs(statistics, suite, methods)), None


def ordered_tasks(conditioned: pd.DataFrame) -> list[str]:
    """Sort by method-symmetric mean task difficulty with task-name tie breaks."""
    difficulty = conditioned.groupby("task").bo_conditioned_regret.mean()
    return sorted(difficulty.index, key=lambda task: (difficulty[task], task))


def draw_cards(
    conditioned: pd.DataFrame,
    *,
    suite: str,
    suite_label: str,
    task_order: list[str],
) -> plt.Figure:
    """Draw square-root-spaced final-regret cards for one benchmark family."""
    nrows, ncols, figsize = LAYOUT[suite]
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True, squeeze=False)
    fig.subplots_adjust(
        left=0.09,
        right=0.99,
        bottom=0.06,
        top=0.82,
        hspace=0.92,
        wspace=0.25,
    )
    x = np.arange(len(METHODS), dtype=np.float64)
    ticks = np.asarray((0.0, 0.05, 0.2, 0.5, 1.0))
    for task_index, ax in enumerate(axes.flat):
        if task_index >= len(task_order):
            ax.set_visible(False)
            continue
        task = task_order[task_index]
        means = []
        for method_index, method in enumerate(METHODS):
            style = METHOD_STYLE[method]
            values = (
                conditioned[(conditioned.task == task) & (conditioned.method == method)]
                .sort_values("seed")
                .bo_conditioned_regret.to_numpy(np.float64)
            )
            require(len(values) == 20, f"{suite}/{task}/{method}: BO point count differs")
            ax.scatter(
                method_index + BO_JITTER,
                np.sqrt(values),
                s=9,
                color=style["color"],
                alpha=0.38,
                edgecolor="none",
                zorder=2,
            )
            means.append(float(np.sqrt(values.mean())))
            ax.scatter(
                method_index,
                means[-1],
                s=25,
                marker=style["marker"],
                color=style["color"],
                edgecolor="white",
                linewidth=0.45,
                zorder=4,
            )
        ax.plot(x, means, color="#585C60", linewidth=0.55, alpha=0.7, zorder=1)
        ax.set_xlim(-0.35, len(METHODS) - 0.65)
        ax.set_ylim(0, 1)
        ax.set_yticks(np.sqrt(ticks), [f"{tick:g}" for tick in ticks])
        ax.set_title(compact_task_label(suite, task), fontsize=6.1, pad=2.0)
        ax.grid(axis="y", color="#D9DCDF", linewidth=0.35, alpha=0.8)
        ax.tick_params(axis="both", length=0, pad=1.0, labelsize=5.5)
        ax.set_xticks([])
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
    fig.suptitle(
        f"{suite_label} task cards — final normalized regret (square-root spacing)",
        x=0.09,
        ha="left",
        fontsize=8.1,
        y=0.98,
    )
    handles = [
        plt.Line2D(
            [],
            [],
            color=METHOD_STYLE[method]["color"],
            marker=METHOD_STYLE[method]["marker"],
            linewidth=1.0,
            markersize=4.2,
            label=METHOD_STYLE[method]["label"],
        )
        for method in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.94),
        ncol=3,
        frameon=False,
        fontsize=6.4,
        handlelength=1.3,
        columnspacing=1.0,
    )
    fig.text(
        0.025,
        0.47,
        "final normalized regret\n(square-root spacing)",
        rotation=90,
        ha="center",
        va="center",
        fontsize=6.4,
    )
    return fig


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> dict[str, str]:
    path = output_dir / f"{stem}.pdf"
    fig.savefig(path)
    return {path.name: content_sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "figures",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_provenance = paper_statistics()["inputs"]
    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans"],
            "font.size": 7.2,
            "axes.linewidth": 0.55,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )

    outputs = {}
    task_orders = {}
    for suite, suite_label in SUITES:
        conditioned, _ladders = bo_conditioned_final(suite)
        task_order = ordered_tasks(conditioned)
        require(len(task_order) == EXPECTED_TASKS[suite], f"{suite}: ordered task count differs")
        task_orders[suite] = task_order
        fig = draw_cards(
            conditioned,
            suite=suite,
            suite_label=suite_label,
            task_order=task_order,
        )
        suite_stem = suite.removesuffix("_ifbo")
        outputs |= save_figure(fig, args.output_dir, f"figure_ftpfn_tasks_{suite_stem}")
        plt.close(fig)

    render_manifest = {
        "schema": "ftpfn-task-card-figure-v1",
        "review_only": False,
        "suites": [suite for suite, _label in SUITES],
        "methods": list(METHODS),
        "seed_hierarchy": {
            "bo_seeds": list(range(20)),
            "icicle_training_seed": 0,
            "ftpfn_checkpoint": "published checkpoint; one value per BO seed",
            "asha_random": "one value per BO seed",
        },
        "normalization": "task table achievable per-configuration-best span, clipped to [0,1]",
        "axis": "square-root spacing; original normalized-regret tick labels",
        "cache_provenance": cache_provenance,
        "display_order": "ascending mean across all methods and BO seeds; task-name tie break",
        "task_order": task_orders,
        "renderers": {
            "cards": {
                "path": "reproduction/figures/render_ftpfn_task_cards.py",
                "sha256": content_sha256(Path(__file__)),
            },
        },
        "outputs": outputs,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    manifest_path = args.output_dir / "figure_ftpfn_tasks_render_manifest.json"
    manifest_path.write_text(json.dumps(render_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"outputs": len(outputs), "manifest": str(manifest_path)}))


if __name__ == "__main__":
    main()
