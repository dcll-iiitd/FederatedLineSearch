#!/usr/bin/env python3
"""Generate the main-paper and appendix kappa_f figures.

This script deliberately imports the established CSV loading and ratio-of-sums
aggregation from ``plot_kappa.py``. It performs no training and never writes to
the measurement CSVs.
"""

import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from plot_kappa import aggregate_run, load_run


BOUND = 0.5
MAIN_Y_LIMITS = (0.0, 3.0)


def validate_run(run):
    bound = run["armijo_c"] / (2.0 * run["eta_lmax"])
    if not np.isclose(bound, BOUND):
        raise ValueError(
            f"{run['path']} has bound {bound:g}; expected {BOUND:g}"
        )


def load_aggregated(path):
    run = load_run(path)
    validate_run(run)
    ratios, eta_stats, failure_rates = aggregate_run(run)
    if not ratios:
        raise ValueError(f"No valid aggregated rounds in {path}")
    return {
        "run": run,
        "ratios": ratios,
        "eta_stats": eta_stats,
        "failure_rates": failure_rates,
    }


def require_paths(paths):
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("Missing required input files: " + ", ".join(missing))


def grouped_ratio_statistics(aggregated_runs):
    by_round = defaultdict(list)
    for item in aggregated_runs:
        for round_number, values in item["ratios"].items():
            by_round[round_number].append(values)

    rounds = sorted(by_round)
    expected_count = len(aggregated_runs)
    for round_number in rounds:
        if len(by_round[round_number]) != expected_count:
            raise ValueError(
                f"Round {round_number} is missing one or more seed measurements"
            )

    values = np.asarray([by_round[round_number] for round_number in rounds])
    return (
        np.asarray(rounds),
        values[:, :, 0].mean(axis=1),
        values[:, :, 0].std(axis=1, ddof=0),
        values[:, :, 1].mean(axis=1),
        values[:, :, 1].std(axis=1, ddof=0),
    )


def style_ratio_axis(axis, title):
    axis.axhline(
        BOUND,
        color="black",
        linestyle=":",
        linewidth=1.8,
        label=rf"Bound $c/(2\eta_{{l,\max}})={BOUND:g}$",
        zorder=2,
    )
    axis.set_title(title)
    axis.set_xlabel("Communication round")
    axis.set_ylabel(r"$\kappa_f$ ratio of sums")
    axis.set_ylim(*MAIN_Y_LIMITS)
    axis.grid(True, alpha=0.25, linewidth=0.8)
    axis.legend(loc="best", fontsize=9.5, frameon=True)


def plot_main(cifar_item, femnist_items, output_path):
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), sharey=True)
    color_a = "#1f77b4"
    color_b = "#d95f02"

    cifar_rounds = np.asarray(sorted(cifar_item["ratios"]))
    cifar_values = np.asarray(
        [cifar_item["ratios"][round_number] for round_number in cifar_rounds]
    )
    axes[0].plot(
        cifar_rounds, cifar_values[:, 0],
        color=color_a, linestyle="-", linewidth=2.0, label=r"$R_a$",
    )
    axes[0].plot(
        cifar_rounds, cifar_values[:, 1],
        color=color_b, linestyle="--", linewidth=2.0, label=r"$R_b$",
    )
    style_ratio_axis(axes[0], "CIFAR-10 / ResNet-18 (b=50)")

    rounds, a_mean, a_std, b_mean, b_std = grouped_ratio_statistics(femnist_items)
    axes[1].plot(
        rounds, a_mean,
        color=color_a, linestyle="-", linewidth=2.0, label=r"$R_a$",
    )
    axes[1].fill_between(
        rounds, np.maximum(a_mean - a_std, 0.0), a_mean + a_std,
        color=color_a, alpha=0.18, linewidth=0,
    )
    axes[1].plot(
        rounds, b_mean,
        color=color_b, linestyle="--", linewidth=2.0, label=r"$R_b$",
    )
    axes[1].fill_between(
        rounds, np.maximum(b_mean - b_std, 0.0), b_mean + b_std,
        color=color_b, alpha=0.18, linewidth=0,
    )
    style_ratio_axis(axes[1], "FEMNIST / Logistic Reg (b=50)")

    figure.tight_layout()
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)

    cifar_final_round = int(cifar_rounds[-1])
    cifar_final_a, cifar_final_b = cifar_item["ratios"][cifar_final_round]
    femnist_final_round = int(rounds[-1])
    print(
        "FINAL panel=CIFAR-10/ResNet-18(b=50) "
        f"round={cifar_final_round} R_a={cifar_final_a:.8g} "
        f"R_b={cifar_final_b:.8g}"
    )
    print(
        "FINAL panel=FEMNIST/Logistic-Reg(b=50) "
        f"round={femnist_final_round} R_a={a_mean[-1]:.8g} "
        f"R_b={b_mean[-1]:.8g} R_a_std={a_std[-1]:.8g} "
        f"R_b_std={b_std[-1]:.8g} seeds={len(femnist_items)}"
    )

    clipped = int(np.sum(cifar_values > MAIN_Y_LIMITS[1]))
    clipped += int(np.sum(a_mean + a_std > MAIN_Y_LIMITS[1]))
    clipped += int(np.sum(b_mean + b_std > MAIN_Y_LIMITS[1]))
    if clipped:
        print(
            f"NOTE main_y_limits={MAIN_Y_LIMITS} clip {clipped} plotted "
            "mean/curve points or band endpoints above the requested range"
        )
    print(f"Wrote {output_path}")


def plot_appendix(cifar_items, output_path):
    figure, (ratio_axis, eta_axis) = plt.subplots(
        2, 1, figsize=(11.0, 8.5), sharex=True,
        gridspec_kw={"height_ratios": [1.45, 1.0]},
    )
    colors = {50: "#1f77b4", 128: "#d95f02", 256: "#2ca02c"}
    all_positive_ratios = []

    for item in sorted(cifar_items, key=lambda value: value["run"]["batch_size"]):
        batch_size = item["run"]["batch_size"]
        color = colors.get(batch_size)
        rounds = sorted(item["ratios"])
        ratios = np.asarray([item["ratios"][round_number] for round_number in rounds])
        all_positive_ratios.extend(ratios[ratios > 0])

        ratio_axis.plot(
            rounds, ratios[:, 0], color=color, linestyle="-", linewidth=1.9,
            label=rf"$R_a$, $b={batch_size}$",
        )
        ratio_axis.plot(
            rounds, ratios[:, 1], color=color, linestyle="--", linewidth=1.9,
            label=rf"$R_b$, $b={batch_size}$",
        )

        eta_rounds = sorted(item["eta_stats"])
        eta_values = np.asarray(
            [item["eta_stats"][round_number] for round_number in eta_rounds]
        )
        eta_axis.plot(
            eta_rounds, eta_values[:, 0], color=color, linestyle="-", linewidth=1.8,
            label=rf"Mean, $b={batch_size}$",
        )
        eta_axis.plot(
            eta_rounds, eta_values[:, 1], color=color, linestyle="--", linewidth=1.8,
            label=rf"Median, $b={batch_size}$",
        )
        eta_axis.plot(
            eta_rounds, eta_values[:, 2], color=color, linestyle=":", linewidth=1.8,
            label=rf"p95, $b={batch_size}$",
        )

    ratio_axis.axhline(
        BOUND, color="black", linestyle=":", linewidth=1.8,
        label=rf"Bound $c/(2\eta_{{l,\max}})={BOUND:g}$",
    )
    if all_positive_ratios:
        smallest = min(all_positive_ratios)
        largest = max(all_positive_ratios)
        if largest / smallest > 100:
            ratio_axis.set_yscale("log")

    ratio_axis.set_title("CIFAR-10 / ResNet-18: batch-size effect")
    ratio_axis.set_ylabel(r"$\kappa_f$ ratio of sums")
    ratio_axis.grid(True, which="both", alpha=0.25, linewidth=0.8)
    ratio_axis.legend(ncol=2, fontsize=9.5, frameon=True)

    eta_axis.set_xlabel("Communication round")
    eta_axis.set_ylabel("Accepted step size")
    eta_axis.grid(True, alpha=0.25, linewidth=0.8)
    eta_axis.legend(ncol=3, fontsize=9.0, frameon=True)

    figure.tight_layout()
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--main-output", default="results/kappa_main.png")
    parser.add_argument(
        "--appendix-output", default="results/kappa_batchsize_appendix.png"
    )
    args = parser.parse_args()

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 9.5,
    })

    cifar_paths = [
        os.path.join(
            args.results_dir,
            f"kappa_measurements_cifar10_resnet18_b{batch}_s0.csv",
        )
        for batch in (50, 128, 256)
    ]
    femnist_paths = [
        os.path.join(
            args.results_dir,
            f"kappa_measurements_femnist_logreg_b50_s{seed}.csv",
        )
        for seed in (0, 1)
    ]
    require_paths(cifar_paths + femnist_paths)

    cifar_items = [load_aggregated(path) for path in cifar_paths]
    femnist_items = [load_aggregated(path) for path in femnist_paths]
    if {item["run"]["batch_size"] for item in cifar_items} != {50, 128, 256}:
        raise ValueError("Expected CIFAR-10 batch sizes {50, 128, 256}")
    if {item["run"]["seed"] for item in femnist_items} != {0, 1}:
        raise ValueError("Expected FEMNIST seeds {0, 1}")

    os.makedirs(os.path.dirname(args.main_output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.appendix_output) or ".", exist_ok=True)
    cifar_b50 = next(
        item for item in cifar_items if item["run"]["batch_size"] == 50
    )
    plot_main(cifar_b50, femnist_items, args.main_output)
    plot_appendix(cifar_items, args.appendix_output)


if __name__ == "__main__":
    main()
