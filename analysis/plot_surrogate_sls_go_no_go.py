#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_log(path):
    rounds = {}
    current = None
    round_re = re.compile(r"Algo\s+\S+\s+Round No\.\s+(\d+)")
    train_re = re.compile(r"Training Loss\s+([0-9.eE+-]+)")
    test_re = re.compile(r"Test Loss\s+([0-9.eE+-]+)\s+Test Accuracy\s+([0-9.eE+-]+)")
    for line in Path(path).read_text().splitlines():
        match = round_re.search(line)
        if match:
            current = int(match.group(1))
            rounds.setdefault(current, {})
            continue
        if current is None:
            continue
        match = train_re.search(line)
        if match:
            rounds[current]["train_loss"] = float(match.group(1))
        match = test_re.search(line)
        if match:
            rounds[current]["test_loss"] = float(match.group(1))
            rounds[current]["test_acc"] = float(match.group(2))
    return rounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", required=True)
    parser.add_argument("--run-log", required=True)
    parser.add_argument("--scaffold-log", required=True)
    parser.add_argument("--fedsls-log", required=True)
    parser.add_argument("--fedexpsls-log", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.steps)))
    by_round = {}
    for row in rows:
        by_round.setdefault(int(row["round"]), []).append(row)

    rules = {
        "SCAFFOLD-SLS surrogate": parse_log(args.run_log),
        "SCAFFOLD": parse_log(args.scaffold_log),
        "FedSLS": parse_log(args.fedsls_log),
        "FedExpSLS": parse_log(args.fedexpsls_log),
    }
    colors = {
        "SCAFFOLD-SLS surrogate": "#0072B2",
        "SCAFFOLD": "#D55E00",
        "FedSLS": "#009E73",
        "FedExpSLS": "#E69F00",
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name, values in rules.items():
        points = sorted((r, v["test_acc"]) for r, v in values.items()
                        if r < 30 and "test_acc" in v)
        ax.plot([p[0] + 1 for p in points], [p[1] for p in points],
                label=name, color=colors[name], linewidth=2)
    ax.set(xlabel="Communication round", ylabel="Global test accuracy (%)")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "01_test_accuracy.png", dpi=180)
    plt.close(fig)

    round_ids = sorted(by_round)
    medians, p10s, p90s = [], [], []
    h_ascent, f_ascent = [], []
    for round_id in round_ids:
        rs = by_round[round_id]
        etas = np.asarray([float(x["eta"]) for x in rs])
        medians.append(np.median(etas))
        p10s.append(np.quantile(etas, .1))
        p90s.append(np.quantile(etas, .9))
        h_ascent.append(np.mean([float(x["h_after"]) > float(x["h_before"])
                                 for x in rs]))
        f_ascent.append(np.mean([float(x["f_after"]) > float(x["f_before"])
                                 for x in rs]))

    x = np.asarray(round_ids) + 1
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(x, medians, color=colors["SCAFFOLD-SLS surrogate"],
            linewidth=2, label="Median accepted eta")
    ax.fill_between(x, p10s, p90s, color=colors["SCAFFOLD-SLS surrogate"],
                    alpha=.22, label="10th–90th percentile")
    ax.set(xlabel="Communication round", ylabel="Accepted local step size eta")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "02_eta_spread.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(x, h_ascent, linewidth=2, label="Surrogate h ascent",
            color="#CC79A7")
    ax.plot(x, f_ascent, linewidth=2, label="Local f ascent",
            color="#E69F00")
    ax.set(xlabel="Communication round", ylabel="Fraction of accepted steps",
           ylim=(-.005, max(.05, max(f_ascent) * 1.1)))
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "03_ascent_fractions.png", dpi=180)
    plt.close(fig)

    final_round = max(round_ids)
    final_steps = by_round[final_round]
    final_etas = np.asarray([float(x["eta"]) for x in final_steps])
    summary = {
        "final_round_1_based": final_round + 1,
        "eta_median": float(np.median(final_etas)),
        "eta_p10": float(np.quantile(final_etas, .1)),
        "eta_p90": float(np.quantile(final_etas, .9)),
        "h_ascent_fraction": float(np.mean([
            float(x["h_after"]) > float(x["h_before"]) for x in final_steps])),
        "f_ascent_fraction": float(np.mean([
            float(x["f_after"]) > float(x["f_before"]) for x in final_steps])),
    }
    for name, values in rules.items():
        v = values[final_round]
        summary[name] = {k: v[k] for k in ("train_loss", "test_acc")}
    with open(out / "summary.txt", "w") as handle:
        for key, value in summary.items():
            handle.write(f"{key}: {value}\n")
    print(summary)


if __name__ == "__main__":
    main()
