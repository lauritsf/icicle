"""Static ARLBench task cards: one trained model and 20 BO seeds per task."""

import argparse
import json
import textwrap
from datetime import UTC, datetime
from pathlib import Path

from reproduction.figures import arlbench_mean_common as core
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "figures"
STEM = "figure_arlbench_tasks"
FINAL_ITERATION = 999
METHODS = ("icicle-s0", "asha", "hyperband", "random_search")
PLOT_SERIES = tuple(series for series in core.SERIES if series[0] in METHODS)
METHOD_STYLE = {
    key: {"label": label, "color": color, "marker": marker}
    for (key, label, color, _linestyle, _width), marker in zip(
        PLOT_SERIES, ("o", "s", "^", "D"), strict=True
    )
}
TICKS = np.asarray((0.0, 0.05, 0.2, 0.5, 1.0))
BO_JITTER = np.linspace(-0.09, 0.09, 20)
LAYOUT = (7, 6, (5.5, 8.6))


def compact_task_label(task: str) -> str:
    return textwrap.fill(task.split("/")[-1].replace("_", " "), width=16, max_lines=2)


def draw_cards(conditioned: pd.DataFrame, task_order: list[str]) -> plt.Figure:
    nrows, ncols, figsize = LAYOUT
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.05, top=0.86, hspace=0.98, wspace=0.25)
    x = np.arange(len(METHODS), dtype=np.float64)
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
            core.require(len(values) == 20, f"{task}/{method}: BO point count differs")
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
        ax.set_yticks(np.sqrt(TICKS), [f"{tick:g}" for tick in TICKS])
        ax.set_title(compact_task_label(task), fontsize=6.1, pad=2.0)
        ax.grid(axis="y", color="#D9DCDF", linewidth=0.35, alpha=0.8)
        ax.tick_params(axis="both", length=0, pad=1.0, labelsize=5.5)
        ax.set_xticks([])
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
    fig.suptitle(
        "ARLBench mean-seed task cards — final normalized regret (square-root spacing)",
        x=0.09,
        ha="left",
        fontsize=8.1,
        y=0.985,
    )
    handles = [
        plt.Line2D(
            [],
            [],
            color=METHOD_STYLE[m]["color"],
            marker=METHOD_STYLE[m]["marker"],
            linewidth=1.0,
            markersize=4.2,
            label=METHOD_STYLE[m]["label"],
        )
        for m in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.965),
        ncol=3,
        frameon=False,
        fontsize=6.4,
        handlelength=1.3,
        columnspacing=1.0,
    )
    fig.text(
        0.012,
        0.46,
        "final normalized regret (square-root spacing)",
        rotation=90,
        ha="center",
        va="center",
        fontsize=6.4,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    core.configure_matplotlib()
    from reproduction.figures.figcache import paper_statistics, final_task_runs

    statistics = paper_statistics()
    cache_provenance = statistics["inputs"]["hpo"][core.BENCH]
    conditioned = pd.DataFrame(final_task_runs(statistics, core.BENCH, {key: key for key in METHODS}))
    difficulty = conditioned.groupby("task").bo_conditioned_regret.mean()
    task_order = sorted(difficulty.index, key=lambda task: (difficulty[task], task))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = draw_cards(conditioned, task_order)
    outputs = {}
    for ext in ("pdf", "png", "svg"):
        path = args.output_dir / f"{STEM}.{ext}"
        fig.savefig(path, dpi=200)
        if ext == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
        outputs[path.name] = core.content_sha256(path)
    manifest = {
        "schema": "arlbench-task-cards-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "estimand": (
            "final normalized regret at iteration 999; ICICLE training seed 0; "
            "20 BO seeds; per-task achievable-range spans; crash-as-minimum"
        ),
        "methods": list(METHODS),
        "display_order": "ascending mean across all methods and BO seeds; task-name tie break",
        "axis": "square-root spacing; original normalized-regret tick labels",
        "cache_provenance": cache_provenance,
        "outputs": outputs,
    }
    (args.output_dir / f"{STEM}_render_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {args.output_dir}/{STEM}.{{pdf,png,svg}} + manifest for {len(task_order)} tasks")


if __name__ == "__main__":
    main()
