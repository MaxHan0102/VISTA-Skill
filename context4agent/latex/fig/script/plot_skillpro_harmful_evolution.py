#!/usr/bin/env python3
"""Plot average Skill-Pro results on EB-Habitat as a three-bar chart.

Data source:
    ../../Experiments/Skill-Pro-EmbodiedBench/docs/experiment_progress.md

The official-aligned setup uses Qwen3-VL-8B, the official visual VLMPlanner,
10-shot prompting, and 50 evaluation episodes for each EB-Habitat subset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


METHODS = ["No Skill", "Initial Skill", "Unverified\nEvolution"]
AVERAGE_SUCCESS = np.array([29.3, 34.7, 29.3])


def configure_matplotlib() -> None:
    """Use publication-friendly, editable fonts and compact CVPR styling."""

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.0,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.7,
        }
    )


def make_figure() -> mpl.figure.Figure:
    """Create a compact, upright bar chart using only average success rates."""

    np.testing.assert_allclose(AVERAGE_SUCCESS, [29.3, 34.7, 29.3])

    x = np.arange(len(METHODS))
    fig, ax = plt.subplots(figsize=(3.75, 2.45), constrained_layout=False)

    no_skill_gray = "#89939E"
    seed_blue = "#4C78A8"
    evolution_red = "#D95B5B"
    neutral = "#333333"
    grid = "#D7D7D7"
    colors = [no_skill_gray, seed_blue, evolution_red]

    bars = ax.bar(
        x,
        AVERAGE_SUCCESS,
        width=0.58,
        color=colors,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=grid, linewidth=0.55, linestyle=(0, (2, 2)))
    ax.xaxis.grid(False)

    ax.set_xticks(x, METHODS)
    ax.set_ylabel("Avg. success rate (%)")
    ax.set_ylim(0, 40)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_xlim(-0.6, len(METHODS) - 0.4)

    ax.set_title(
        "Pilot Study: Skill Evolution",
        loc="left",
        fontweight="bold",
        pad=5.0,
    )

    for bar, value in zip(bars, AVERAGE_SUCCESS):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.7,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            color=neutral,
            fontsize=8.0,
            fontweight="bold",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(axis="x", length=0, pad=2.5)
    ax.tick_params(axis="y", width=0.6, length=2.5, color="#777777")

    fig.subplots_adjust(left=0.17, right=0.99, top=0.83, bottom=0.20)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Directory for PDF and PNG outputs (default: the parent fig directory).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    fig = make_figure()

    output_stem = args.output_dir / "motivation_unverified_skill_evolution"
    fig.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.025,
        transparent=True,
    )
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.025,
        transparent=True,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
