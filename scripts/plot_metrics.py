#!/usr/bin/env python3
"""
Plot metrics from one or more OpenEvolve checkpoint directories.

Usage:
    python scripts/plot_metrics.py --checkpoints run_1/checkpoints run_2/checkpoints
    python scripts/plot_metrics.py --checkpoints run_1/checkpoints --labels "Experiment 1"
    python scripts/plot_metrics.py --checkpoints run_1/checkpoints --metric score --save
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_iteration_metrics_from_checkpoints(checkpoint_folder: str) -> List[Dict[str, Any]]:
    """
    Scan for sibling checkpoints and extract best program metrics per iteration.
    Returns a list of {iteration, metrics} dictionaries sorted by iteration.

    This function is adapted from visualizer.py to work standalone.
    """
    iteration_metrics = []

    # Get parent directory (e.g., "checkpoints/")
    parent_dir = os.path.dirname(checkpoint_folder)
    if not parent_dir or not os.path.exists(parent_dir):
        # If checkpoint_folder is already the parent checkpoints dir
        parent_dir = checkpoint_folder

    # Find all checkpoint directories
    try:
        all_items = os.listdir(parent_dir)
    except (OSError, PermissionError) as e:
        logger.warning(f"Cannot read directory {parent_dir}: {e}")
        return iteration_metrics

    checkpoint_dirs = []
    for item in all_items:
        item_path = os.path.join(parent_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint_"):
            checkpoint_dirs.append(item_path)

    if not checkpoint_dirs:
        logger.warning(f"No checkpoint directories found in {parent_dir}")
        return iteration_metrics

    # Extract iteration number and metrics from each checkpoint
    for ckpt_dir in checkpoint_dirs:
        best_info_path = os.path.join(ckpt_dir, "best_program_info.json")
        if not os.path.exists(best_info_path):
            continue

        try:
            with open(best_info_path, "r") as f:
                best_info = json.load(f)

            # Extract iteration number (prefer current_iteration, fallback to iteration)
            iteration = best_info.get("current_iteration", best_info.get("iteration", 0))
            metrics = best_info.get("metrics", {})

            if metrics:
                iteration_metrics.append({"iteration": iteration, "metrics": metrics})
        except (json.JSONDecodeError, KeyError, IOError) as e:
            logger.debug(f"Error loading best_program_info from {ckpt_dir}: {e}")
            continue

    # Sort by iteration number
    iteration_metrics.sort(key=lambda x: x["iteration"])

    if iteration_metrics:
        logger.info(f"Loaded {len(iteration_metrics)} iteration snapshots from {parent_dir}")

    return iteration_metrics


def get_common_metrics(all_run_data: List[List[Dict[str, Any]]]) -> List[str]:
    """
    Find metrics that are present in all runs.
    """
    if not all_run_data or not all_run_data[0]:
        return []

    # Get metrics from first run's first data point
    metric_sets = []
    for run_data in all_run_data:
        if run_data:
            run_metrics = set()
            for point in run_data:
                if "metrics" in point and isinstance(point["metrics"], dict):
                    run_metrics.update(point["metrics"].keys())
            metric_sets.append(run_metrics)

    if not metric_sets:
        return []

    # Find intersection of all metric sets
    common = set.intersection(*metric_sets)
    return sorted(list(common))


def compute_grouped_statistics(
    group_data: List[List[Dict[str, Any]]], metric_name: str
) -> Tuple[List[int], List[float], List[float], List[float]]:
    """
    Compute mean and range statistics for a group of runs.
    Returns: (iterations, means, mins, maxs)
    """
    # Collect all unique iterations across the group
    all_iterations = set()
    for run_data in group_data:
        for point in run_data:
            all_iterations.add(point["iteration"])

    iterations = sorted(list(all_iterations))

    means = []
    mins = []
    maxs = []

    for iteration in iterations:
        values = []
        for run_data in group_data:
            # Find this iteration in this run
            for point in run_data:
                if point["iteration"] == iteration:
                    if (
                        "metrics" in point
                        and isinstance(point["metrics"], dict)
                        and metric_name in point["metrics"]
                    ):
                        val = point["metrics"][metric_name]
                        if isinstance(val, (int, float)) and not (
                            isinstance(val, float) and (val != val)
                        ):
                            values.append(val)
                    break

        if values:
            means.append(np.mean(values))
            mins.append(np.min(values))
            maxs.append(np.max(values))
        else:
            # Skip this iteration if no data
            continue

    return iterations[: len(means)], means, mins, maxs


def plot_metrics(
    checkpoint_groups: List[List[str]],
    labels: List[str],
    metrics: List[str] = None,
    save: bool = False,
    output: str = None,
    style: str = "seaborn-v0_8-darkgrid",
):
    """
    Plot metrics from multiple checkpoint directories or groups of directories.
    Each group will show mean line with shaded range.
    """
    # Load data from all checkpoints
    all_group_data = []
    for group in checkpoint_groups:
        group_data = []
        for ckpt_dir in group:
            data = load_iteration_metrics_from_checkpoints(ckpt_dir)
            group_data.append(data)
        all_group_data.append(group_data)

    # Flatten for validation
    all_run_data_flat = [run for group in all_group_data for run in group]

    # Validate we have data
    if not any(all_run_data_flat):
        logger.error("No data loaded from any checkpoint directory")
        return

    # Determine which metrics to plot
    if metrics is None:
        metrics = get_common_metrics(all_run_data_flat)
        if not metrics:
            logger.error("No common metrics found across all runs")
            return
        logger.info(f"Plotting all common metrics: {', '.join(metrics)}")
    else:
        # Validate requested metrics exist
        available_metrics = get_common_metrics(all_run_data_flat)
        invalid = set(metrics) - set(available_metrics)
        if invalid:
            logger.warning(f"Requested metrics not available in all runs: {invalid}")
            metrics = [m for m in metrics if m in available_metrics]
        if not metrics:
            logger.error("None of the requested metrics are available")
            return

    # Apply plot style
    try:
        plt.style.use(style)
    except:
        logger.warning(f"Style '{style}' not available, using default")

    # Create subplots (one per metric, max 3 rows)
    num_metrics = len(metrics)
    num_cols = (num_metrics + 2) // 3  # Ceiling division to get number of columns
    num_rows = min(num_metrics, 3)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10 * num_cols, 3 * num_rows), squeeze=False)
    axes = axes.flatten()

    # Plot each metric
    for metric_idx, metric_name in enumerate(metrics):
        ax = axes[metric_idx]

        for group_idx, (group_data, label) in enumerate(zip(all_group_data, labels)):
            # Compute statistics for this group
            iterations, means, mins, maxs = compute_grouped_statistics(group_data, metric_name)

            if iterations:
                # Plot mean line
                line = ax.plot(
                    iterations, means, marker="o", label=label, linewidth=2, markersize=4
                )
                color = line[0].get_color()

                # Plot shaded range
                ax.fill_between(iterations, mins, maxs, alpha=0.2, color=color)

        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f"{metric_name} over iterations", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    # Hide any unused subplots
    for idx in range(num_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    # Save or show
    if save:
        if output is None:
            output = "metrics_plot.png"
        plt.savefig(output, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot metrics from OpenEvolve checkpoint directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot metrics from a single run
  python scripts/plot_metrics.py --checkpoints examples/circle_packing/openevolve_output/checkpoints

  # Compare two groups (averaged with range shading)
  python scripts/plot_metrics.py \\
    --checkpoints run1/ckpt run2/ckpt run3/ckpt : run4/ckpt run5/ckpt \\
    --labels "Method A" "Method B"

  # Single runs (each treated as its own group)
  python scripts/plot_metrics.py \\
    --checkpoints run_1/checkpoints : run_2/checkpoints \\
    --labels "Run 1" "Run 2"

  # Plot specific metric and save to file
  python scripts/plot_metrics.py \\
    --checkpoints run_1/checkpoints \\
    --metric score \\
    --save --output results.png
        """,
    )

    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help='Checkpoint directories to plot. Use ":" to separate groups. Example: run1/ckpt run2/ckpt : run3/ckpt run4/ckpt',
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Labels for each run (defaults to checkpoint directory names)",
    )
    parser.add_argument(
        "--metric",
        "--metrics",
        nargs="+",
        dest="metrics",
        help="Specific metric(s) to plot (default: all common metrics)",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save plot to file instead of displaying"
    )
    parser.add_argument("--output", help="Output filename (default: metrics_plot.png)")
    parser.add_argument(
        "--style",
        default="seaborn-v0_8-darkgrid",
        help="Matplotlib style to use (default: seaborn-v0_8-darkgrid)",
    )

    args = parser.parse_args()

    # Parse checkpoint groups (split by ":")
    checkpoint_groups = []
    current_group = []
    for item in args.checkpoints:
        if item == ":":
            if current_group:
                checkpoint_groups.append(current_group)
                current_group = []
        else:
            if not os.path.exists(item):
                logger.error(f"Checkpoint directory does not exist: {item}")
                sys.exit(1)
            current_group.append(item)

    # Add the last group
    if current_group:
        checkpoint_groups.append(current_group)

    if not checkpoint_groups:
        logger.error("No checkpoint directories provided")
        sys.exit(1)

    # Generate labels if not provided
    if args.labels:
        if len(args.labels) != len(checkpoint_groups):
            logger.error(
                f"Number of labels ({len(args.labels)}) must match number of groups ({len(checkpoint_groups)})"
            )
            sys.exit(1)
        labels = args.labels
    else:
        # Default labels: show group size
        labels = []
        for group in checkpoint_groups:
            if len(group) == 1:
                labels.append(os.path.basename(os.path.dirname(group[0])) or os.path.basename(group[0]))
            else:
                labels.append(f"Group of {len(group)} runs")

    # Plot
    plot_metrics(
        checkpoint_groups=checkpoint_groups,
        labels=labels,
        metrics=args.metrics,
        save=args.save,
        output=args.output,
        style=args.style,
    )


if __name__ == "__main__":
    main()
