#!/usr/bin/env python3
import argparse
import os
import re
import csv
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt

ALGO_COLOR = {
    "FedExpSls":   "#080bb4",  # blue
    "FedAvg":      "#3eb408bb",  # green
    "FedExp":      "#441a6c",  # purple
    "FedExpProx":  "#b51212",  # red
    "FedSls":      "#ff7f0e",  # orange
    "FedAdam":     "#2E9787",  # teal/cyan
}
# key=value (numeric) pairs like train_loss=..., test_acc=..., round=...
KV_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\b")

# STRICT round patterns (avoid round_time_sec, etc.)
KV_ROUND_RE = re.compile(r"\bround\s*=\s*(\d+)\b", re.IGNORECASE)
BLOCK_ROUND_RE = re.compile(r"\bRound\s+No\.?\s*[:=]?\s*(\d+)\b", re.IGNORECASE)

# tolerant metric patterns (block style)
TRAIN_LOSS_RE = re.compile(r"\bTraining\s+Loss\b\s*[:=]?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.IGNORECASE)
TEST_ACC_RE   = re.compile(r"\bTest\s+Acc(?:uracy)?\b\s*[:=]?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.IGNORECASE)

# also accept shorter variants if your logs have them
TRAIN_LOSS_SHORT_RE = re.compile(r"\bTrain(?:ing)?\s+Loss\b\s*[:=]?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.IGNORECASE)
TEST_ACC_SHORT_RE   = re.compile(r"\bTest\s+Acc\b\s*[:=]?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.IGNORECASE)


def safe_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def ema(values: List[float], alpha: float) -> List[float]:
    if not values or not (0.0 < alpha < 1.0):
        return values
    out = []
    s = values[0]
    for v in values:
        s = alpha * v + (1 - alpha) * s
        out.append(s)
    return out


def parse_log(filepath: str, max_round: Optional[int], debug: bool = False) -> Dict[str, List[Tuple[int, float]]]:
    """
    Returns:
      train_loss: [(round, val), ...]
      test_acc:   [(round, val), ...]
    Keeps LAST value per round.
    """
    train_loss: Dict[int, float] = {}
    test_acc: Dict[int, float] = {}

    current_round: Optional[int] = None

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue

            # 1) strict round parsing
            m = KV_ROUND_RE.search(s)
            if m:
                current_round = int(m.group(1))
            else:
                m = BLOCK_ROUND_RE.search(s)
                if m:
                    current_round = int(m.group(1))

            if current_round is None:
                continue
            if max_round is not None and current_round > max_round:
                continue

            # 2) key=value metrics if present in the SAME line
            kvs = {k.lower(): v for k, v in KV_RE.findall(s)}

            if "train_loss" in kvs:
                v = safe_float(kvs["train_loss"])
                if v is not None:
                    train_loss[current_round] = v
                    if debug:
                        print(f"[{os.path.basename(filepath)}:{line_no}] round={current_round} train_loss={v}")

            if "test_acc" in kvs:
                v = safe_float(kvs["test_acc"])
                if v is not None:
                    test_acc[current_round] = v
                    if debug:
                        print(f"[{os.path.basename(filepath)}:{line_no}] round={current_round} test_acc={v}")

            # 3) block-style metrics (Training Loss ... Test Accuracy ...)
            m = TRAIN_LOSS_RE.search(s) or TRAIN_LOSS_SHORT_RE.search(s)
            if m:
                v = safe_float(m.group(1))
                if v is not None:
                    train_loss[current_round] = v
                    if debug:
                        print(f"[{os.path.basename(filepath)}:{line_no}] round={current_round} train_loss={v}")

            m = TEST_ACC_RE.search(s) or TEST_ACC_SHORT_RE.search(s)
            if m:
                v = safe_float(m.group(1))
                if v is not None:
                    test_acc[current_round] = v
                    if debug:
                        print(f"[{os.path.basename(filepath)}:{line_no}] round={current_round} test_acc={v}")

    out: Dict[str, List[Tuple[int, float]]] = {}
    if train_loss:
        out["train_loss"] = sorted(train_loss.items())
    if test_acc:
        out["test_acc"] = sorted(test_acc.items())
    return out


def dump_csv(path: str, rounds_vals: List[Tuple[int, float]]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["round", "value"])
        w.writerows(rounds_vals)


def plot_metric(all_series: Dict[str, List[Tuple[int, float]]],
                title: str, ylabel: str, out_path: str, pdf: bool):
    plt.figure(figsize=(8, 6))
    any_plotted = False

    for label, pts in all_series.items():
        if not pts:
            continue
        xs = [r for r, _ in pts]
        ys = [v for _, v in pts]

        color = ALGO_COLOR.get(label, None)  # fixed paper colors
        plt.plot(xs, ys, label=label, color=color)

        any_plotted = True

    if not any_plotted:
        print(f"[WARN] No data to plot for {out_path}")
        plt.close()
        return

    plt.title(title)
    plt.xlabel("Communication Round")
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
    ap.add_argument("--ema", type=float, default=0.0)
    ap.add_argument("--out_prefix", type=str, default="")
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--dump_csv", action="store_true", help="Write parsed CSVs for sanity checking")
    args = ap.parse_args()

    algo_paths = {}
    for item in args.algo:
        name, path = item.split("=", 1)
        algo_paths[name.strip()] = path.strip()

    loss_series = {}
    acc_series = {}

    for algo, path in algo_paths.items():
        data = parse_log(path, args.max_round, debug=args.debug)

        loss_pts = data.get("train_loss", [])
        acc_pts  = data.get("test_acc", [])

        # EMA smoothing (preserve same rounds)
        if args.ema and loss_pts:
            xs = [r for r, _ in loss_pts]
            ys = ema([v for _, v in loss_pts], args.ema)
            loss_pts = list(zip(xs, ys))

        if args.ema and acc_pts:
            xs = [r for r, _ in acc_pts]
            ys = ema([v for _, v in acc_pts], args.ema)
            acc_pts = list(zip(xs, ys))

        loss_series[algo] = loss_pts
        acc_series[algo] = acc_pts

        if args.dump_csv:
            if loss_pts:
                dump_csv(f"{args.out_prefix}{algo}_train_loss.csv", loss_pts)
            if acc_pts:
                dump_csv(f"{args.out_prefix}{algo}_test_acc.csv", acc_pts)

        # quick sanity print
        if loss_pts:
            print(f"{algo}: train_loss rounds [{loss_pts[0][0]} .. {loss_pts[-1][0]}], n={len(loss_pts)}")
        else:
            print(f"{algo}: train_loss NOT FOUND")
        if acc_pts:
            print(f"{algo}: test_acc rounds [{acc_pts[0][0]} .. {acc_pts[-1][0]}], n={len(acc_pts)}")
        else:
            print(f"{algo}: test_acc NOT FOUND")

    plot_metric(loss_series, "Training Loss vs Communication Round", "Training Loss",
                f"{args.out_prefix}train_loss.png", args.pdf)
    plot_metric(acc_series, "Test Accuracy vs Communication Round", "Test Accuracy",
                f"{args.out_prefix}test_acc.png", args.pdf)


if __name__ == "__main__":
    main()
