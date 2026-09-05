#!/usr/bin/env python3
"""Plot five-seed average Armijo trials per local step from SLS run logs."""

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SLS_RE = re.compile(
    r"^\[SLS\]\s+round=(?P<round>\d+).*?"
    r"avg_trials_per_step=(?P<value>[-+0-9.eE]+)"
)


def read_log(path):
    values = {}
    with open(path, "r", errors="replace") as handle:
        for line in handle:
            match = SLS_RE.search(line)
            if match:
                values[int(match.group("round"))] = float(match.group("value"))
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--min-seeds", type=int, default=5)
    args = parser.parse_args()

    runs = [read_log(path) for path in args.log]
    if len(runs) < args.min_seeds:
        raise ValueError(f"Expected at least {args.min_seeds} seeds, found {len(runs)}")
    common_rounds = sorted(set.intersection(*(set(run) for run in runs)))
    if not common_rounds:
        raise ValueError("The supplied logs have no common SLS rounds")

    matrix = np.asarray([[run[round_no] for run in runs] for round_no in common_rounds])
    means = matrix.mean(axis=1)
    stds = matrix.std(axis=1, ddof=1)

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["round", "mean", "std", "n"])
        for round_no, mean, std in zip(common_rounds, means, stds):
            writer.writerow([round_no, mean, std, len(runs)])

    # 10.8 x 5.4 inches at 100 dpi corresponds to the requested 1080 x 540 canvas.
    fig, ax = plt.subplots(figsize=(10.8, 5.4), dpi=100)
    color = "#62C5EA"
    ax.plot(common_rounds, means, color=color, linewidth=1.0)
    ax.fill_between(common_rounds, means - stds, means + stds, color=color, alpha=0.30)
    ax.set_xlabel("Communication Round", fontsize=28)
    ax.set_ylabel("Average\nLine Search Steps", fontsize=28)
    ax.tick_params(axis="both", labelsize=16)
    ax.tick_params(axis="x", labelrotation=90)
    ax.grid(True, linestyle="--", alpha=0.55)
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.25, top=0.975)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path} and {csv_path}; rounds={len(common_rounds)}, seeds={len(runs)}")


if __name__ == "__main__":
    main()
