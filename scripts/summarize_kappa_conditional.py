#!/usr/bin/env python3
"""Summaries and figures for within-state conditional means, never pooled states."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summarize(root, plots=True):
    manifest = json.loads((root / "manifest.json").read_text())
    settings = manifest["batches"]
    bound = manifest["bound"]
    expected = manifest.get("expected_cells_per_batch", 400)
    cells = [json.loads(path.read_text()) for path in sorted((root / "cells").glob("*.json"))]
    write_csv(root / "conditional_cells.csv", cells)
    clients = []
    batches = []
    for batch in settings:
        group = [cell for cell in cells if cell.get("batch_setting", cell["batch_size"]) == batch]
        finite = [cell for cell in group if cell.get("R_a") is not None and cell.get("R_b") is not None]
        for client in range(manifest.get("num_clients", 100)):
            rows = [cell for cell in group if cell["client_id"] == client]
            if not rows:
                continue
            ratios = [cell for cell in rows if cell.get("R_a") is not None and cell.get("R_b") is not None]
            clients.append({
                "batch_setting": batch, "batch_size": rows[0]["batch_size"],
                "client_id": client, "completed_states": len(rows),
                "expected_states": 4, "worst_R_a": max((r["R_a"] for r in ratios), default=None),
                "worst_R_b": max((r["R_b"] for r in ratios), default=None),
                "failed_searches": sum(r["failed_searches"] for r in rows),
            })
        exact = batch in manifest["exact_batches"]
        all_pass = len(group) == expected and all(
            cell["status"] in ("enumerated_below", "degenerate_exact")
            and cell["failed_searches"] == 0 for cell in group
        )
        max_a = max((r["R_a"] for r in finite), default=None)
        max_b = max((r["R_b"] for r in finite), default=None)
        point_pass = sum(bool(r.get("both_point_estimates_below")) for r in group)
        status = ("complete_enumerated_support_at_tested_states" if all_pass else
                  "complete_with_nonpassing_or_invalid_states" if len(group) == expected and exact else
                  "complete_mc_estimates_only" if len(group) == expected else "in_progress")
        row = {
            "batch_size": batch, "batch_setting": batch,
            "actual_batch_min": min((r["batch_size"] for r in group), default=None),
            "actual_batch_max": max((r["batch_size"] for r in group), default=None),
            "method": "enumeration" if exact else "monte_carlo",
            "completed_states": len(group), "expected_states": expected,
            "completed_trials": sum(r["trials"] for r in group),
            "bound": bound, "worst_R_a": max_a, "worst_R_b": max_b,
            "states_with_both_point_estimates_below": point_pass,
            "failed_searches": sum(r["failed_searches"] for r in group),
            "zero_denominator_trials": sum(r["zero_denominator_trials"] for r in group),
            "positive_denominator_below_1e_8": sum(r["positive_denominator_below_1e_8"] for r in group),
            "status": status,
            "candidate_common_kappa_on_tested_states": (
                (max(max_a or 0, max_b or 0) + bound) / 2 if all_pass else None),
        }
        batches.append(row)
        print(json.dumps(row), flush=True)
    write_csv(root / "client_maxima.csv", clients)
    write_csv(root / "batch_summary.csv", batches)
    if plots and cells:
        plot_maxima(root, batches, manifest)
        plot_heatmaps(root, cells, manifest)
        plot_training(root, manifest)
    return batches


def setting_label(batch):
    return "n_i - 1 (leave one out)" if batch == "loo" else "n_i (full)" if batch == "full" else f"b={batch}"


def plot_maxima(root, batches, manifest):
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12})
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    positions = np.arange(3)
    for offset, field, label, color in ((-0.12, "worst_R_a", r"Max $R_a$", "#1f77b4"),
                                        (0.12, "worst_R_b", r"Max $R_b$", "#d95f02")):
        values = [row[field] if row[field] is not None else np.nan for row in batches]
        ax.scatter(positions + offset, values, color=color, s=48, label=label)
    ax.axhline(manifest["bound"], color="black", linestyle=":", label=f"Bound = {manifest['bound']:g}")
    values = [row[key] for row in batches for key in ("worst_R_a", "worst_R_b")
              if row[key] is not None and row[key] > 0]
    if values and max(values) / min(values) > 100:
        ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{setting_label(row['batch_setting'])}\n{row['completed_states']}/{row['expected_states']} states" for row in batches])
    ax.set_ylabel("Worst conditional ratio across completed states")
    ax.set_xlabel("Batch setting")
    ax.set_title(f"{manifest['dataset']}/{manifest['model']}, reset 2, initial step {manifest['initial_step']:g}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(root / "plots" / f"conditional_maxima.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(root, cells, manifest):
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    times = [(0, 1), (0, 20), (99, 1), (99, 20)]
    matrices = []
    for batch in manifest["batches"]:
        matrix = np.full((manifest.get("num_clients", 100), 4), np.nan)
        for row in cells:
            if row.get("batch_setting", row["batch_size"]) == batch and row.get("R_a") is not None and row.get("R_b") is not None:
                matrix[row["client_id"], times.index((row["round"], row["local_step"]))] = max(row["R_a"], row["R_b"]) / manifest["bound"]
        matrices.append(matrix)
    finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
    upper = max(1.0, float(finite.max())) if len(finite) else 1.0
    for ax, batch, matrix in zip(axes, manifest["batches"], matrices):
        plot = ax.imshow(matrix, vmin=0, vmax=upper, aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_title(setting_label(batch))
        ax.set_xticks(range(4))
        ax.set_xticklabels(["t0/k1", "t0/k20", "t99/k1", "t99/k20"], rotation=35, ha="right")
        ax.set_xlabel("Frozen conditioning state")
    axes[0].set_ylabel("Client ID")
    fig.colorbar(plot, ax=axes.tolist(), label="max(conditional ratios) / bound; <1 is below")
    fig.subplots_adjust(bottom=0.22, right=0.84)
    fig.savefig(root / "plots" / "client_state_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_training(root, manifest):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7), sharex=True)
    found = False
    for batch in manifest["batches"]:
        path = root / "training" / f"b{batch}.csv"
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("train_loss") and row.get("test_acc")]
        if not rows:
            continue
        found = True
        rounds = [int(row["round"]) for row in rows]
        axes[0].plot(rounds, [float(row["train_loss"]) for row in rows], label=setting_label(batch))
        axes[1].plot(rounds, [float(row["test_acc"]) for row in rows], label=setting_label(batch))
    if found:
        axes[0].set_ylabel("Training loss")
        axes[1].set_ylabel("Test accuracy (%)")
        axes[1].set_xlabel("Communication round")
        for ax in axes:
            ax.grid(alpha=0.25)
            ax.legend()
        fig.tight_layout()
        fig.savefig(root / "plots" / "training_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    summarize(args.root.resolve(), plots=not args.no_plots)


if __name__ == "__main__":
    main()
