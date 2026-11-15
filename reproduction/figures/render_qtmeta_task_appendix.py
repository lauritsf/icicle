"""Render Quick-Tune appendix task cards at the paper's text width.

Show absolute final normalized regret, 20 BO-seed values and their mean.
Use square-root axis spacing with labels in original units. Outputs include
figures, captions, alt text and source hashes.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from reproduction.baselines.qtmeta_common import content_sha256  # noqa: E402
from reproduction.figures.qtmeta_common import (  # noqa: E402
    EXPECTED_TASKS,
    FACET_LABELS,
    METHOD_STYLE,
    METHODS,
    SPLITS,
    configure_matplotlib,
    require,
    save_figure,
    task_label,
)

APPENDIX_FIGSIZE = (5.5, 7.0)
CAPTION = (
    "Task-level final normalized regret on the {split} split ({n_tasks} tasks). Tasks are "
    "grouped by held-out Meta-Album set and sorted within each set by their mean across all "
    "four methods and BO seeds. The vertical axis "
    "uses square-root spacing to expand the low-regret region; tick labels remain in the "
    "original normalized-regret units. Large markers are per-task method means and the 20 "
    "small markers are individual BO runs for ICICLE training seed 0 and Quick-Tune "
    "meta-training seed 0 within each task's fixed held-out fold. ASHA and random "
    "search contribute one run per BO seed. Lower is better."
)


METHOD_SHORT = {
    "icicle": "IC",
    "quicktune": "QT",
    "asha": "ASHA",
    "random_search": "RS",
}
SET_NAMES = ("set0", "set1", "set2")
BO_X_JITTER = np.linspace(-0.10, 0.10, 20)


def task_set(task: str) -> str:
    """Return the frozen Meta-Album set prefix from a canonical task name."""
    return task_label(task).split("-", 1)[0]


def bare_task_label(task: str) -> str:
    """Return the dataset label without the redundant set prefix."""
    return task_label(task).split("-", 1)[1]


def ordered_tasks(conditioned: pd.DataFrame, split: str) -> list[str]:
    """Group by set and sort by method-symmetric final-regret difficulty."""
    subset = conditioned[conditioned.split == split]
    difficulty = subset.groupby("task", sort=False).bo_conditioned_regret.mean()
    result = []
    for set_name in SET_NAMES:
        tasks = [task for task in difficulty.index if task_set(task) == set_name]
        result.extend(sorted(tasks, key=lambda task: (difficulty[task], bare_task_label(task))))
    require(len(result) == EXPECTED_TASKS[split], f"{split}: display task count differs")
    require(len(set(result)) == len(result), f"{split}: duplicate display task")
    return result


def task_values(
    conditioned: pd.DataFrame,
    *,
    split: str,
    task: str,
    method: str,
) -> np.ndarray:
    """Read the 20 BO-conditioned values in BO-seed order."""
    values = (
        conditioned[
            (conditioned.split == split)
            & (conditioned.task == task)
            & (conditioned.method == method)
        ]
        .sort_values("bo_seed")
        .bo_conditioned_regret.to_numpy(np.float64)
    )
    require(len(values) == 20, f"{split}/{task}/{method}: BO point count differs")
    return values


def add_figure_legend(fig: plt.Figure, *, y: float = 0.985) -> None:
    handles = []
    for method in METHODS:
        style = METHOD_STYLE[method]
        handles.append(
            plt.Line2D(
                [],
                [],
                color=style["color"],
                marker=style["marker"],
                linewidth=1.0,
                markersize=4.2,
                label=style["label"],
            )
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.58, y),
        ncol=4,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.0,
    )


def card_axis_scale(ax: plt.Axes, scale: str) -> None:
    if scale == "linear":
        ax.set_ylim(0, 1)
        ax.set_yticks((0, 0.5, 1.0))
    else:
        ticks = np.asarray((0.0, 0.05, 0.2, 0.5, 1.0))
        ax.set_ylim(0, 1)
        ax.set_yticks(np.sqrt(ticks), [f"{tick:g}" for tick in ticks])


def draw_cards(
    conditioned: pd.DataFrame,
    *,
    split: str,
    task_order: list[str],
    scale: str,
    figsize: tuple[float, float] = (7.0, 8.6),
    title_prefix: str = "A. ",
    subtitle: str | None = (
        "20 BO-seed values, large mean, and a neutral line connecting method means"
    ),
    axes_top: float = 0.915,
) -> plt.Figure:
    """Draw one compact method profile per task on a shared regret scale."""
    require(scale in {"linear", "sqrt"}, f"unknown card scale {scale}")
    fig, axes = plt.subplots(6, 5, figsize=figsize, sharey=True)
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.055,
        top=axes_top,
        hspace=0.76,
        wspace=0.22,
    )
    x = np.arange(len(METHODS), dtype=np.float64)
    slot = 0
    for set_index, set_name in enumerate(SET_NAMES):
        set_tasks = [task for task in task_order if task_set(task) == set_name]
        for local_index in range(10):
            row = set_index * 2 + local_index // 5
            col = local_index % 5
            ax = axes[row, col]
            if local_index >= len(set_tasks):
                ax.set_visible(False)
                continue
            task = set_tasks[local_index]
            means = []
            for method_index, method in enumerate(METHODS):
                style = METHOD_STYLE[method]
                values = task_values(conditioned, split=split, task=task, method=method)
                plotted = values if scale == "linear" else np.sqrt(values)
                means.append(plotted.mean() if scale == "linear" else np.sqrt(values.mean()))
                ax.scatter(
                    method_index + BO_X_JITTER,
                    plotted,
                    s=8,
                    color=style["color"],
                    alpha=0.35,
                    edgecolor="none",
                    zorder=2,
                )
                ax.scatter(
                    method_index,
                    means[-1],
                    s=24,
                    marker=style["marker"],
                    color=style["color"],
                    edgecolor="white",
                    linewidth=0.45,
                    zorder=4,
                )
            ax.plot(x, means, color="#585C60", linewidth=0.55, alpha=0.7, zorder=1)
            card_axis_scale(ax, scale)
            ax.set_xlim(-0.35, 3.35)
            ax.set_title(bare_task_label(task), fontsize=6.7, pad=2.0)
            ax.grid(axis="y", color="#D9DCDF", linewidth=0.35, alpha=0.8)
            ax.tick_params(axis="both", length=0, pad=1.0, labelsize=5.8)
            if local_index >= 5:
                ax.set_xticks(x, [METHOD_SHORT[method] for method in METHODS], fontsize=5.3)
            else:
                ax.set_xticks(x, [])
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
            slot += 1
        y_center = 0.795 - set_index * 0.287
        fig.text(
            0.016, y_center, set_name.upper(), rotation=90, va="center", ha="left", fontsize=7.0
        )
        if set_index:
            fig.add_artist(
                plt.Line2D(
                    (0.012, 0.99),
                    (0.922 - set_index * 0.287,) * 2,
                    color="#9CA1A6",
                    linewidth=0.55,
                )
            )
    require(slot == EXPECTED_TASKS[split], f"{split}: rendered card count differs")
    spacing = "linear scale" if scale == "linear" else "square-root axis spacing"
    title = f"{title_prefix}Task cards — {FACET_LABELS[split]}, {spacing}"
    if subtitle is not None:
        title += f"\n{subtitle}"
    fig.suptitle(
        title,
        x=0.075,
        ha="left",
        fontsize=8.1,
        y=0.978,
    )
    add_figure_legend(fig, y=0.958 if subtitle is not None else 0.948)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from reproduction.figures.figcache import final_task_runs, paper_statistics

    statistics = paper_statistics()
    methods = dict(
        icicle="icicle-s0", quicktune="quicktune-s0", asha="asha", random_search="random_search"
    )
    conditioned = pd.DataFrame(
        [row for split in SPLITS for row in final_task_runs(statistics, "qtmeta_" + split, methods)]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stems = [f"figure_qtmeta_tasks_{split}" for split in SPLITS]
    configure_matplotlib()

    outputs = {}
    captions = {}
    for split, stem in zip(SPLITS, stems, strict=True):
        task_order = ordered_tasks(conditioned, split)
        fig = draw_cards(
            conditioned,
            split=split,
            task_order=task_order,
            scale="sqrt",
            figsize=APPENDIX_FIGSIZE,
            title_prefix="",
            subtitle=None,
            axes_top=0.88,
        )
        outputs |= save_figure(fig, args.output_dir, stem)
        plt.close(fig)
        captions[split] = CAPTION.format(split=FACET_LABELS[split], n_tasks=EXPECTED_TASKS[split])
        alt = (
            f"A grid of {EXPECTED_TASKS[split]} task cards for the {split} split, grouped into "
            "three held-out Meta-Album sets. Every card compares ICICLE, Quick-Tune, ASHA, and "
            "random search on a shared square-root-spaced final normalized-regret axis. Twenty "
            "small translucent points per method show BO-conditioned values and a larger marker "
            "shows their mean; a thin neutral line connects the four means within each task."
        )
        alt_path = args.output_dir / f"{stem}_alt.txt"
        alt_path.write_text(alt + "\n")
        outputs[alt_path.name] = content_sha256(alt_path)

    caption_path = args.output_dir / "figure_qtmeta_tasks_caption.txt"
    caption_path.write_text("\n\n".join(captions[split] for split in SPLITS) + "\n")
    outputs[caption_path.name] = content_sha256(caption_path)
    render_manifest = {
        "schema": "qtmeta-task-appendix-render-v1",
        "inputs": statistics["inputs"],
        "renderers": {
            "appendix": {
                "path": "reproduction/figures/render_qtmeta_task_appendix.py",
                "sha256": content_sha256(Path(__file__)),
            },
            "shared_styles": {
                "path": "reproduction/figures/qtmeta_common.py",
                "sha256": content_sha256(Path("reproduction/figures/qtmeta_common.py")),
            },
        },
        "display_order": (
            "held-out set blocks; within each set ascending mean final normalized regret "
            "across all four methods and 20 BO seeds"
        ),
        "axis": {
            "data": "final normalized regret in [0,1]",
            "spacing": "square root",
            "tick_labels": "original normalized-regret units",
        },
        "figure_size_inches": APPENDIX_FIGSIZE,
        "outputs": outputs,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    manifest_path = args.output_dir / "figure_qtmeta_tasks_render_manifest.json"
    manifest_path.write_text(json.dumps(render_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"outputs": len(outputs), "manifest": str(manifest_path)}))


if __name__ == "__main__":
    main()
