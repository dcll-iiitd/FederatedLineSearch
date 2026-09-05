#!/usr/bin/env python3
import argparse
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

# Match the whole WALLCLOCK line and pull out needed key=vals robustly
KV_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\b")

REQUIRED = ("round", "elapsed_total_sec")  # must exist for time plots


def parse_wallclock(filepath: str, max_round: int | None = None) -> Dict[str, List[Tuple[float, float]]]:
    """
    Returns dict with:
      round_vs_train_loss: [(round, train_loss), ...]
      round_vs_test_acc:   [(round, test_acc), ...]
      time_vs_train_loss:  [(elapsed_sec, train_loss), ...]
      time_vs_test_acc:    [(elapsed_sec, test_acc), ...]

    We only parse lines that start with [WALLCLOCK].
    """
    round_train_loss = {}
    round_test_acc = {}
    time_train_loss = {}
    time_test_acc = {}

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "[WALLCLOCK]" not in line:
                continue

            kvs = {k: float(v) for k, v in KV_RE.findall(line)}

            # must have round + elapsed_total_sec
            if not all(k in kvs for k in REQUIRED):
                continue

            r = int(kvs["round"])
            if max_round is not None and r > max_round:
                continue

            t = kvs["elapsed_total_sec"]

            # optional metrics if present
            if "train_loss" in kvs:
                round_train_loss[r] = kvs["train_loss"]
                time_train_loss[t] = kvs["train_loss"]

            if "test_acc" in kvs:
                round_test_acc[r] = kvs["test_acc"]
                time_test_acc[t] = kvs["test_acc"]

    def sorted_items(d):
        return sorted(d.items(), key=lambda x: x[0])

    return {
        "round_vs_train_loss": sorted_items(round_train_loss),
        "round_vs_test_acc": sorted_items(round_test_acc),
        "time_vs_train_loss": sorted_items(time_train_loss),
        "time_vs_test_acc": sorted_items(time_test_acc),
    }


def ema(values, alpha: float):
    if not values or not (0.0 < alpha < 1.0):
        return values
    out = []
    s = values[0]
    for v in values:
        s = alpha * v + (1 - alpha) * s
        out.append(s)
    return out


def plot_multi(series: Dict[str, List[Tuple[float, float]]],
               title: str, xlabel: str, ylabel: str,
               out_path: str, pdf: bool = False):
    plt.figure(figsize=(8, 6))
    any_plotted = False

    for label, pts in series.items():
        if not pts:
            continue
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        plt.plot(xs, ys, label=label)
        any_plotted = True

    if not any_plotted:
        print(f"[WARN] No data to plot for {out_path}")
        plt.close()
        return

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    if pdf:
        plt.savefig(os.path.splitext(out_path)[0] + ".pdf")
    plt.close()
    print(f"Saved: {out_path}" + (" (+PDF)" if pdf else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", action="append", required=True, help="NAME=PATH (repeatable)")
    ap.add_argument("--max_round", type=int, default=None)
    ap.add_argument("--ema", type=float, default=0.0, help="EMA alpha in (0,1); 0 disables")
    ap.add_argument("--out_prefix", type=str, default="")
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()

    algo_paths = dict(item.split("=", 1) for item in args.algo)
    algo_paths = {k.strip(): v.strip() for k, v in algo_paths.items()}

    # Collect per-plot series
    loss_round = {}
    acc_round = {}
    loss_time = {}
    acc_time = {}

    for algo, path in algo_paths.items():
        data = parse_wallclock(path, args.max_round)

        r_loss = data["round_vs_train_loss"]
        r_acc  = data["round_vs_test_acc"]
        t_loss = data["time_vs_train_loss"]
        t_acc  = data["time_vs_test_acc"]

        # Smooth Y only (keep X)
        if args.ema and r_loss:
            xs = [x for x, _ in r_loss]
            ys = ema([y for _, y in r_loss], args.ema)
            r_loss = list(zip(xs, ys))
        if args.ema and r_acc:
            xs = [x for x, _ in r_acc]
            ys = ema([y for _, y in r_acc], args.ema)
            r_acc = list(zip(xs, ys))
        if args.ema and t_loss:
            xs = [x for x, _ in t_loss]
            ys = ema([y for _, y in t_loss], args.ema)
            t_loss = list(zip(xs, ys))
        if args.ema and t_acc:
            xs = [x for x, _ in t_acc]
            ys = ema([y for _, y in t_acc], args.ema)
            t_acc = list(zip(xs, ys))

        loss_round[algo] = r_loss
        acc_round[algo]  = r_acc
        loss_time[algo]  = t_loss
        acc_time[algo]   = t_acc

        # sanity prints
        print(f"{algo}: WALLCLOCK train_loss points={len(r_loss)}, test_acc points={len(r_acc)}")

    plot_multi(loss_round, "Training Loss vs Communication Round",
               "Communication Round", "Training Loss",
               f"{args.out_prefix}train_loss_vs_round.png", args.pdf)

    plot_multi(acc_round, "Test Accuracy vs Communication Round",
               "Communication Round", "Test Accuracy",
               f"{args.out_prefix}test_acc_vs_round.png", args.pdf)

    plot_multi(loss_time, "Training Loss vs Wallclock Time",
               "Wallclock Time (sec)", "Training Loss",
               f"{args.out_prefix}train_loss_vs_wallclock.png", args.pdf)

    plot_multi(acc_time, "Test Accuracy vs Wallclock Time",
               "Wallclock Time (sec)", "Test Accuracy",
               f"{args.out_prefix}test_acc_vs_wallclock.png", args.pdf)


if __name__ == "__main__":
    main()
