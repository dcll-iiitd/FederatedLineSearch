#!/usr/bin/env python3
"""Plot training metrics from the completed FedSLS kappa-measurement runs."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROUND_RE = re.compile(
    r"\[WALLCLOCK\].*?round=(?P<round>\d+).*?"
    r"train_loss=(?P<train_loss>[-+0-9.eE]+).*?"
    r"test_acc=(?P<test_acc>[-+0-9.eE]+)"
)


def read_metrics(path: Path):
    values = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = ROUND_RE.search(line)
            if match:
                round_number = int(match.group("round"))
                values[round_number] = (
                    float(match.group("train_loss")),
                    float(match.group("test_acc")),
                )
    if not values:
        raise ValueError(f"No [WALLCLOCK] metrics found in {path}")
    rounds = np.asarray(sorted(values), dtype=int)
    if not np.array_equal(rounds, np.arange(rounds[-1] + 1)):
        raise ValueError(f"Missing or non-contiguous rounds in {path}")
    metrics = np.asarray([values[round_number] for round_number in rounds])
    return rounds, metrics[:, 0], metrics[:, 1]


def plot_cifar(axis, runs, metric_index, ylabel):
    colors = {50: "#1f77b4", 128: "#d95f02", 256: "#2ca02c"}
    for batch_size, (rounds, train_loss, test_acc) in sorted(runs.items()):
        values = train_loss if metric_index == 0 else test_acc
        axis.plot(
            rounds,
            values,
            color=colors[batch_size],
            linewidth=1.8,
            label=f"b={batch_size}",
        )
    axis.set_title(f"CIFAR-10 / ResNet-18: {ylabel}")
    axis.set_ylabel(ylabel)
    axis.legend(frameon=True, fontsize=10)


def plot_femnist(axis, runs, metric_index, ylabel):
    rounds = runs[0][0]
    for other in runs[1:]:
        if not np.array_equal(rounds, other[0]):
            raise ValueError("FEMNIST seeds do not contain the same rounds")
    values = np.stack(
        [run[1] if metric_index == 0 else run[2] for run in runs], axis=0
    )
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=0)
    axis.plot(rounds, mean, color="#8c2d9c", linewidth=1.9, label="Mean (2 seeds)")
    axis.fill_between(
        rounds, mean - std, mean + std,
        color="#8c2d9c", alpha=0.20, linewidth=0, label=r"$\pm 1$ std",
    )
    axis.set_title(f"FEMNIST / Logistic Reg: {ylabel}")
    axis.set_ylabel(ylabel)
    axis.legend(frameon=True, fontsize=10)
    return mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/kappa")
    parser.add_argument("--output", default="results/kappa_fedsls_training_curves")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cifar = {
        batch: read_metrics(run_dir / f"cifar10_resnet18_b{batch}_s0.log")
        for batch in (50, 128, 256)
    }
    femnist = [
        read_metrics(run_dir / f"femnist_logreg_b50_s{seed}.log")
        for seed in (0, 1)
    ]

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
    })
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")

    plot_cifar(axes[0, 0], cifar, 0, "Training loss")
    plot_cifar(axes[1, 0], cifar, 1, "Test accuracy (%)")
    fem_loss_mean, fem_loss_std = plot_femnist(
        axes[0, 1], femnist, 0, "Training loss"
    )
    fem_acc_mean, fem_acc_std = plot_femnist(
        axes[1, 1], femnist, 1, "Test accuracy (%)"
    )

    for axis in axes.flat:
        axis.grid(True, alpha=0.25, linewidth=0.8)
        axis.set_xlim(0, 499)
    for axis in axes[1, :]:
        axis.set_xlabel("Communication round")

    figure.suptitle(
        r"FedSLS training curves for $\kappa_f$ measurement runs "
        r"($c=0.1$, $\eta_{l,\max}=0.1$, reset option 2)",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.965))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        path = output.with_suffix(suffix)
        figure.savefig(path, dpi=240, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(figure)

    for batch, (_, loss, acc) in sorted(cifar.items()):
        print(
            f"FINAL dataset=CIFAR10 batch={batch} seed=0 round=499 "
            f"train_loss={loss[-1]:.6f} test_acc={acc[-1]:.6f}"
        )
    print(
        "FINAL dataset=FEMNIST batch=50 seeds=0,1 round=499 "
        f"train_loss_mean={fem_loss_mean[-1]:.6f} "
        f"train_loss_std={fem_loss_std[-1]:.6f} "
        f"test_acc_mean={fem_acc_mean[-1]:.6f} "
        f"test_acc_std={fem_acc_std[-1]:.6f}"
    )


if __name__ == "__main__":
    main()
