#!/usr/bin/env python3
import argparse
import glob
import os
import math
import csv
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt

def read_round_value_csv(path: str) -> Dict[int, float]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rnd = int(float(row["round"]))
                val = float(row["value"])
                out[rnd] = val
            except Exception:
                continue
    return out

def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    m = sum(values) / len(values)
    if len(values) == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, math.sqrt(var)

def build_avg_series(paths: List[str], min_seeds: int) -> Tuple[List[int], List[float], List[float], List[int]]:
    # read each seed csv -> dict round->value
    seed_maps = [read_round_value_csv(p) for p in paths]

    # union of rounds
    rounds = sorted(set().union(*[m.keys() for m in seed_maps]) if seed_maps else set())

    xs, means, stds, counts = [], [], [], []
    for rnd in rounds:
        vals = [m[rnd] for m in seed_maps if rnd in m]
        if len(vals) < min_seeds:
            continue
        mu, sd = mean_std(vals)
        xs.append(rnd)
        means.append(mu)
        stds.append(sd)
        counts.append(len(vals))
    return xs, means, stds, counts

def dump_avg_csv(path: str, xs: List[int], means: List[float], stds: List[float], counts: List[int]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["round", "mean", "std", "n"])
        for r, m, s, n in zip(xs, means, stds, counts):
            w.writerow([r, f"{m:.10g}", f"{s:.10g}", n])

def plot_avg(xs: List[int], means: List[float], stds: List[float], title: str, ylabel: str,
             out_path: str, pdf: bool, band: str):
    plt.figure(figsize=(8, 6))
    plt.plot(xs, means, label="mean")

    if band in ("std", "sem", "ci95"):
        if band == "std":
            lo = [m - s for m, s in zip(means, stds)]
            hi = [m + s for m, s in zip(means, stds)]
        elif band == "sem":
            # sem = std/sqrt(n) – but we didn’t pass n here; treat as std if not available
            lo = [m - s for m, s in zip(means, stds)]
            hi = [m + s for m, s in zip(means, stds)]
        else:  # ci95 ≈ 1.96*SEM; again, no n here; keep as std-band unless you extend
            lo = [m - s for m, s in zip(means, stds)]
            hi = [m + s for m, s in zip(means, stds)]

        plt.fill_between(xs, lo, hi, alpha=0.2, linewidth=0)

    plt.title(title)
    plt.xlabel("Communication Round")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    if pdf:
        plt.savefig(os.path.splitext(out_path)[0] + ".pdf")
    plt.close()
    print(f"Saved: {out_path}" + (" (+PDF)" if pdf else ""))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True,
                    help="Glob pattern for per-seed CSVs. Must contain metric suffix like *_train_loss.csv or *_test_acc.csv")
    ap.add_argument("--min_seeds", type=int, default=5, help="Require at least this many seeds at each round")
    ap.add_argument("--out_prefix", type=str, default="avg_", help="Output prefix for avg CSV + plots")
    ap.add_argument("--title", type=str, default="Average over seeds")
    ap.add_argument("--ylabel", type=str, default="value")
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--band", choices=["none", "std"], default="none", help="Shaded band type")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No files matched: {args.pattern}")

    xs, means, stds, counts = build_avg_series(paths, args.min_seeds)

    if not xs:
        raise SystemExit("No rounds survived min_seeds filtering. Try --min_seeds 3 or check CSVs.")

    out_csv = f"{args.out_prefix}avg.csv"
    dump_avg_csv(out_csv, xs, means, stds, counts)
    print(f"Wrote: {out_csv}  (rounds={len(xs)}, seeds per round min={min(counts)} max={max(counts)})")

    # output filename from metric suffix
    base_plot = f"{args.out_prefix}plot.png"
    plot_avg(xs, means, stds, args.title, args.ylabel, base_plot, args.pdf, args.band)

if __name__ == "__main__":
    main()