#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def parse_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_run(csv_path):
    metadata_path = os.path.splitext(csv_path)[0] + ".json"
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Missing run metadata for {csv_path}: expected {metadata_path}"
        )

    with open(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)

    for key in ("seed", "eta_lmax", "armijo_c"):
        if key not in metadata:
            raise ValueError(f"{metadata_path} does not contain required key {key}")

    eta_lmax = float(metadata["eta_lmax"])
    armijo_c = float(metadata["armijo_c"])
    if eta_lmax <= 0:
        raise ValueError(f"eta_lmax must be positive in {metadata_path}")

    rows = []
    with open(csv_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {
            "round", "client_id", "local_step", "eta_returned",
            "loss_prev_batch", "loss_curr_batch", "f_ref_prev",
            "f_ref_curr", "grad_sq_norm", "line_search_failed", "seed"
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

        for row in reader:
            rows.append({
                "round": int(row["round"]),
                "eta": float(row["eta_returned"]),
                "loss_prev": float(row["loss_prev_batch"]),
                "loss_curr": float(row["loss_curr_batch"]),
                "f_ref_prev": float(row["f_ref_prev"]),
                "f_ref_curr": float(row["f_ref_curr"]),
                "grad_sq_norm": float(row["grad_sq_norm"]),
                "failed": parse_bool(row["line_search_failed"]),
            })

    return {
        "path": csv_path,
        "seed": int(metadata["seed"]),
        "eta_lmax": eta_lmax,
        "armijo_c": armijo_c,
        "rows": rows,
    }


def aggregate_run(run):
    by_round = defaultdict(list)
    for row in run["rows"]:
        by_round[row["round"]].append(row)

    ratios = {}
    eta_stats = {}
    failure_rates = {}

    for round_number, rows in sorted(by_round.items()):
        failure_rates[round_number] = (
            sum(row["failed"] for row in rows) / len(rows) if rows else math.nan
        )
        valid = [
            row for row in rows
            if not row["failed"]
            and math.isfinite(row["loss_curr"])
            and math.isfinite(row["grad_sq_norm"])
            and math.isfinite(row["eta"])
        ]

        denominator = sum(
            row["eta"] ** 2 * row["grad_sq_norm"] for row in valid
        )
        if denominator > 0:
            numerator_a = sum(
                abs(row["loss_prev"] - row["f_ref_prev"]) for row in valid
            )
            numerator_b = sum(
                abs(row["loss_curr"] - row["f_ref_curr"]) for row in valid
            )
            ratios[round_number] = (
                numerator_a / denominator,
                numerator_b / denominator,
            )

        if valid:
            eta_values = np.asarray([row["eta"] for row in valid], dtype=float)
            eta_stats[round_number] = (
                float(np.mean(eta_values)),
                float(np.median(eta_values)),
                float(np.percentile(eta_values, 95)),
            )

    return ratios, eta_stats, failure_rates


def mean_std(values):
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), float(np.std(array, ddof=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-glob",
        default="results/kappa_measurements_seed*.csv",
    )
    parser.add_argument(
        "--output",
        default="results/kappa_ratio.png",
    )
    args = parser.parse_args()

    csv_paths = sorted(glob.glob(args.input_glob))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files match {args.input_glob}")

    runs = [load_run(path) for path in csv_paths]
    aggregated = []
    for run in runs:
        ratios, eta_stats, failure_rates = aggregate_run(run)
        aggregated.append((run, ratios, eta_stats))
        for round_number, rate in sorted(failure_rates.items()):
            print(
                f"seed={run['seed']} eta_lmax={run['eta_lmax']:g} "
                f"round={round_number} line_search_failure_rate={rate:.6f}"
            )

    ratio_groups = defaultdict(lambda: defaultdict(list))
    eta_groups = defaultdict(lambda: defaultdict(list))

    for run, ratios, eta_stats in aggregated:
        key = (run["eta_lmax"], run["armijo_c"])

        for round_number, (ratio_a, ratio_b) in ratios.items():
            ratio_groups[key][round_number].append((ratio_a, ratio_b))
        for round_number, stats in eta_stats.items():
            eta_groups[key][round_number].append(stats)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    figure, (ratio_axis, eta_axis) = plt.subplots(
        2, 1, figsize=(10, 10), sharex=True
    )
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(ratio_groups), 1)))
    all_positive_ratios = []

    for color, key in zip(colors, sorted(ratio_groups)):
        cap, armijo_c = key
        round_numbers = sorted(ratio_groups[key])
        ratio_a_mean = []
        ratio_a_std = []
        ratio_b_mean = []
        ratio_b_std = []

        for round_number in round_numbers:
            values = ratio_groups[key][round_number]
            a_mean, a_std = mean_std([value[0] for value in values])
            b_mean, b_std = mean_std([value[1] for value in values])
            ratio_a_mean.append(a_mean)
            ratio_a_std.append(a_std)
            ratio_b_mean.append(b_mean)
            ratio_b_std.append(b_std)

        if np.allclose(ratio_a_mean, ratio_b_mean, rtol=0, atol=0):
            raise RuntimeError(
                f"R_a and R_b are identical for eta_lmax={cap}; "
                "check w_(k-1)/w_k instrumentation"
            )

        x_values = np.asarray(round_numbers)
        a_mean = np.asarray(ratio_a_mean)
        a_std = np.asarray(ratio_a_std)
        b_mean = np.asarray(ratio_b_mean)
        b_std = np.asarray(ratio_b_std)
        all_positive_ratios.extend(a_mean[a_mean > 0])
        all_positive_ratios.extend(b_mean[b_mean > 0])

        ratio_axis.plot(
            x_values, a_mean, color=color, linestyle="-",
            label=rf"$R_a$, $c={armijo_c:g}$, $\eta_{{lmax}}={cap:g}$"
        )
        ratio_axis.fill_between(
            x_values, np.maximum(a_mean - a_std, 0), a_mean + a_std,
            color=color, alpha=0.12
        )
        ratio_axis.plot(
            x_values, b_mean, color=color, linestyle="--",
            label=rf"$R_b$, $c={armijo_c:g}$, $\eta_{{lmax}}={cap:g}$"
        )
        ratio_axis.fill_between(
            x_values, np.maximum(b_mean - b_std, 0), b_mean + b_std,
            color=color, alpha=0.12
        )

        bound = armijo_c / (2.0 * cap)
        ratio_axis.axhline(
            bound, color=color, linestyle=":", linewidth=1.5,
            label=rf"$c={armijo_c:g}$ bound $={bound:g}$"
        )

        eta_rounds = sorted(eta_groups[key])
        pooled = {}
        for round_number in eta_rounds:
            values = np.asarray(eta_groups[key][round_number], dtype=float)
            pooled[round_number] = np.mean(values, axis=0)

        eta_axis.plot(
            eta_rounds, [pooled[r][0] for r in eta_rounds],
            color=color, linestyle="-", label=rf"mean, c={armijo_c:g}, cap={cap:g}"
        )
        eta_axis.plot(
            eta_rounds, [pooled[r][1] for r in eta_rounds],
            color=color, linestyle="--", label=rf"median, c={armijo_c:g}, cap={cap:g}"
        )
        eta_axis.plot(
            eta_rounds, [pooled[r][2] for r in eta_rounds],
            color=color, linestyle=":", label=rf"p95, c={armijo_c:g}, cap={cap:g}"
        )

    if all_positive_ratios:
        smallest = min(all_positive_ratios)
        largest = max(all_positive_ratios)
        if largest / smallest > 100:
            ratio_axis.set_yscale("log")

    ratio_axis.set_ylabel(r"$\kappa_f$ ratio of sums")
    ratio_axis.set_title("Kappa-f accuracy measurements")
    ratio_axis.grid(True, alpha=0.25)
    ratio_axis.legend(ncol=2, fontsize=8)

    eta_axis.set_xlabel("Communication round")
    eta_axis.set_ylabel("Accepted step size")
    eta_axis.grid(True, alpha=0.25)
    eta_axis.legend(ncol=3, fontsize=8)

    figure.tight_layout()
    figure.savefig(args.output, dpi=200)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
