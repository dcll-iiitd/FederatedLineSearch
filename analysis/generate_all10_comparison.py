#!/usr/bin/env python3
"""Generate five-seed round and wall-clock comparisons for the paper algorithms."""

from __future__ import annotations

import csv
import math
import re
from bisect import bisect_right
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ROUNDS = tuple(range(1000))
SEEDS = tuple(range(5))
TIME_GRID_SECONDS = 60.0

DATASETS = {
    "cifar10": {
        "title": "CIFAR-10",
        "run_dir": ROOT / "runs/cifar10",
        "axis_labels": True,
    },
    "cifar100": {
        "title": "CIFAR-100",
        "run_dir": ROOT / "runs/cifar100",
        "axis_labels": False,
    },
    "femnist": {
        "title": "FEMNIST",
        "run_dir": ROOT / "runs/femnist_full_3597_m20_r1000",
        "axis_labels": False,
    },
    "shakespeare": {
        "title": "SHAKESPEARE",
        "run_dir": ROOT / "runs/shakespeare",
        "axis_labels": False,
    },
}

ALGORITHMS = {
    "FedAvg": {
        "default": "fedavg_seed{seed}.log",
        "color": "#008000",
        "linestyle": "-",
    },
    "FedExp": {
        "default": "fedexp_seed{seed}.log",
        "color": "#800080",
        "linestyle": "-",
    },
    "FedExProx": {
        "default": "fedexprox_constant_a1p5_seed{seed}.log",
        "femnist": "fedexprox_seed{seed}.log",
        "color": "#D62728",
        "linestyle": "-",
    },
    "FedDyn": {
        "default": "feddyn_seed{seed}.log",
        "color": "#E377C2",
        "linestyle": "-",
    },
    "SCAFFOLD": {
        "default": "scaffold_seed{seed}.log",
        "color": "#8C564B",
        "linestyle": "-",
    },
    "FedAdam": {
        "default": "fedadam_canonical_seed{seed}.log",
        "color": "#2E9787",
        "linestyle": "-",
    },
    "FedSLS": {
        "default": "fedsls_seed{seed}.log",
        "color": "#FF7F0E",
        "linestyle": "-",
    },
    "FedSLS (reg.)": {
        "default": "fedsls_regularized_seed{seed}.log",
        "color": "#BCBD22",
        "linestyle": "-",
    },
    "FedExpSLS": {
        "default": "fedexpsls_seed{seed}.log",
        "color": "#0000FF",
        "linestyle": "-",
    },
    "FedExpSLS (reg.)": {
        "default": "fedexpsls_regularized_seed{seed}.log",
        "color": "#17BECF",
        "linestyle": "-",
    },
}

KV_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)="
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\b"
)


def parse_log(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "[WALLCLOCK]" not in line:
                continue
            values = {key: float(value) for key, value in KV_RE.findall(line)}
            if not {"round", "elapsed_total_sec", "train_loss", "test_acc"} <= values.keys():
                continue
            round_idx = int(values["round"])
            rows[round_idx] = values
    missing = sorted(set(ROUNDS) - rows.keys())
    if missing:
        raise ValueError(
            f"{path} is incomplete: found {len(rows)}/1000 rounds; "
            f"first missing round={missing[0]}"
        )
    return rows


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def average_by_round(
    seed_rows: list[dict[int, dict[str, float]]], metric: str
) -> list[tuple[float, float, float, int]]:
    output = []
    for round_idx in ROUNDS:
        values = [rows[round_idx][metric] for rows in seed_rows]
        mean, std = mean_std(values)
        output.append((float(round_idx), mean, std, len(values)))
    return output


def average_by_time(
    seed_rows: list[dict[int, dict[str, float]]], metric: str
) -> list[tuple[float, float, float, int]]:
    series = []
    for rows in seed_rows:
        ordered = [rows[round_idx] for round_idx in ROUNDS]
        times = [row["elapsed_total_sec"] for row in ordered]
        values = [row[metric] for row in ordered]
        series.append((times, values))

    common_start = max(times[0] for times, _ in series)
    common_end = min(times[-1] for times, _ in series)
    first_grid = math.ceil(common_start / TIME_GRID_SECONDS) * TIME_GRID_SECONDS

    output = []
    time_sec = first_grid
    while time_sec <= common_end:
        values = []
        for times, metric_values in series:
            index = bisect_right(times, time_sec) - 1
            if index < 0:
                raise RuntimeError("Time grid precedes a seed's first measurement")
            values.append(metric_values[index])
        mean, std = mean_std(values)
        output.append((time_sec, mean, std, len(values)))
        time_sec += TIME_GRID_SECONDS
    return output


def write_average(path: Path, x_name: str, rows: list[tuple[float, float, float, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([x_name, "mean", "std", "n"])
        for x_value, mean, std, count in rows:
            if x_name == "round":
                x_value = int(x_value)
            writer.writerow([x_value, f"{mean:.10g}", f"{std:.10g}", count])


def plot_comparison(
    path: Path,
    series: dict[str, list[tuple[float, float, float, int]]],
    xlabel: str,
    ylabel: str,
    show_axis_labels: bool,
) -> None:
    # Same 790x632 canvas used by the repository's existing paper plots.
    dpi = 192
    fig, axis = plt.subplots(figsize=(790 / dpi, 632 / dpi), dpi=dpi)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.17, top=0.97)

    for name, rows in series.items():
        style = ALGORITHMS[name]
        axis.plot(
            [row[0] for row in rows],
            [row[1] for row in rows],
            label=name,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.2,
        )

    if show_axis_labels:
        axis.set_xlabel(xlabel, fontsize=12)
        axis.set_ylabel(ylabel, fontsize=12)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=6.8, ncol=2, frameon=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    fig.savefig(path.with_suffix(".png"), dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> None:
    for dataset_key, dataset in DATASETS.items():
        output_dir = ROOT / "analysis" / dataset_key / "all10_comparison"
        round_series = {"train_loss": {}, "test_acc": {}}
        time_series = {"train_loss": {}, "test_acc": {}}

        for algorithm, config in ALGORITHMS.items():
            pattern = config.get(dataset_key, config["default"])
            paths = [dataset["run_dir"] / pattern.format(seed=seed) for seed in SEEDS]
            seed_rows = [parse_log(path) for path in paths]

            safe_name = (
                algorithm.lower()
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace(".", "")
            )
            for metric in ("train_loss", "test_acc"):
                averaged_rounds = average_by_round(seed_rows, metric)
                averaged_times = average_by_time(seed_rows, metric)
                round_series[metric][algorithm] = averaged_rounds
                time_series[metric][algorithm] = averaged_times
                write_average(
                    output_dir / f"{safe_name}_{metric}_vs_round.csv",
                    "round",
                    averaged_rounds,
                )
                write_average(
                    output_dir / f"{safe_name}_{metric}_vs_wallclock.csv",
                    "time_sec",
                    averaged_times,
                )

        show_labels = bool(dataset["axis_labels"])
        plot_comparison(
            output_dir / "all10_train_loss_vs_round.pdf",
            round_series["train_loss"],
            "Communication round",
            "Training loss",
            show_labels,
        )
        plot_comparison(
            output_dir / "all10_test_accuracy_vs_round.pdf",
            round_series["test_acc"],
            "Communication round",
            "Test accuracy (%)",
            show_labels,
        )
        plot_comparison(
            output_dir / "all10_train_loss_vs_wallclock.pdf",
            time_series["train_loss"],
            "Wall-clock time (seconds)",
            "Training loss",
            show_labels,
        )
        plot_comparison(
            output_dir / "all10_test_accuracy_vs_wallclock.pdf",
            time_series["test_acc"],
            "Wall-clock time (seconds)",
            "Test accuracy (%)",
            show_labels,
        )
        print(f"Generated {dataset['title']}: {output_dir}")


if __name__ == "__main__":
    main()
