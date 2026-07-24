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
        "dataset": metadata.get("dataset", "unknown"),
        "model": metadata.get("model", "unknown"),
        "batch_size": int(metadata.get("batch_size", 50)),
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
    parser.add_argument("--title", default="Kappa-f accuracy measurements")
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
                f"dataset={run['dataset']} model={run['model']} "
                f"batch={run['batch_size']} seed={run['seed']} "
                f"eta_lmax={run['eta_lmax']:g} "
                f"round={round_number} line_search_failure_rate={rate:.6f}"
            )
        if ratios:
            final_round = max(ratios)
            final_a, final_b = ratios[final_round]
            run_bound = run["armijo_c"] / (2.0 * run["eta_lmax"])
            final_failure = failure_rates.get(final_round, math.nan)
            print(
                f"VERDICT dataset={run['dataset']} model={run['model']} "
                f"batch={run['batch_size']} seed={run['seed']} "
                f"round={final_round} R_a={final_a:.8g} R_b={final_b:.8g} "
                f"failure_rate={final_failure:.6f} bound={run_bound:g} "
                f"R_a_below={final_a < run_bound} R_b_below={final_b < run_bound}"
            )

    caps = {run["eta_lmax"] for run in runs}
    armijo_values = {run["armijo_c"] for run in runs}
    if len(caps) != 1 or len(armijo_values) != 1:
        raise ValueError("All overlaid runs must use the same eta_lmax and armijo_c")
    cap = next(iter(caps))
    armijo_c = next(iter(armijo_values))
    bound = armijo_c / (2.0 * cap)

    ratio_groups = defaultdict(lambda: defaultdict(list))
    eta_groups = defaultdict(lambda: defaultdict(list))

    for run, ratios, eta_stats in aggregated:
        key = (run["dataset"], run["model"], run["batch_size"])

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
        dataset, model, batch_size = key
        group_label = f"{dataset}, {model}, b={batch_size}"
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
            label=f"R_a, {group_label}"
        )
        ratio_axis.fill_between(
            x_values, np.maximum(a_mean - a_std, 0), a_mean + a_std,
            color=color, alpha=0.12
        )
        ratio_axis.plot(
            x_values, b_mean, color=color, linestyle="--",
            label=f"R_b, {group_label}"
        )
        ratio_axis.fill_between(
            x_values, np.maximum(b_mean - b_std, 0), b_mean + b_std,
            color=color, alpha=0.12
        )


        eta_rounds = sorted(eta_groups[key])
        pooled = {}
        for round_number in eta_rounds:
            values = np.asarray(eta_groups[key][round_number], dtype=float)
            pooled[round_number] = np.mean(values, axis=0)

        eta_axis.plot(
            eta_rounds, [pooled[r][0] for r in eta_rounds],
            color=color, linestyle="-", label=f"mean, {group_label}"
        )
        eta_axis.plot(
            eta_rounds, [pooled[r][1] for r in eta_rounds],
            color=color, linestyle="--", label=f"median, {group_label}"
        )
        eta_axis.plot(
            eta_rounds, [pooled[r][2] for r in eta_rounds],
            color=color, linestyle=":", label=f"p95, {group_label}"
        )

    ratio_axis.axhline(
        bound, color="black", linestyle=":", linewidth=1.5,
        label=f"bound c/(2 eta_lmax)={bound:g}"
    )

    if all_positive_ratios:
        smallest = min(all_positive_ratios)
        largest = max(all_positive_ratios)
        if largest / smallest > 100:
            ratio_axis.set_yscale("log")

    ratio_axis.set_ylabel(r"$\kappa_f$ ratio of sums")
    ratio_axis.set_title(args.title)
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
