#!/usr/bin/env python3
"""Summarize completed screening jobs without changing their measurements."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from plot_kappa import aggregate_run, load_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    manifest = json.loads((args.root / "manifest.json").read_text())
    states = {}
    for path in args.root.glob("gpu*_status.jsonl"):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            states[record["name"]] = record["state"]
    report = []
    for job in manifest["jobs"]:
        if args.gpu is not None and job["gpu"] != args.gpu:
            continue
        row = {"name": job["name"], "state": states.get(job["name"], "pending"),
               "full_local_batch_control": job["full_local_batch_control"]}
        if row["state"] == "completed":
            run = load_run(job["csv"])
            ratios, eta_stats, failures = aggregate_run(run)
            metadata = json.loads(Path(job["csv"]).with_suffix(".json").read_text())
            with Path(job["csv"]).open(newline="") as handle:
                actual_batch_sizes = sorted({int(r["batch_size_actual"]) for r in csv.DictReader(handle)})
            row["actual_batch_sizes"] = str(actual_batch_sizes)
            if metadata["client_sizes"] != metadata["reference_sizes"]:
                raise ValueError(f"{job['name']}: not a full-client reference")
            bound = run["armijo_c"] / (2 * run["effective_eta_lmax"])
            if ratios:
                rounds = sorted(ratios)
                last = rounds[-1]
                a, b = ratios[last]
                success = [max(ratios[t]) < bound for t in rounds]
                late = [max(ratios[t]) < bound for t in rounds if t >= job["rounds"] // 2]
                row.update(bound=bound, last_measured_round=last, R_a=a, R_b=b,
                           both_below=bool(max(a, b) < bound),
                           measured_rounds=len(rounds),
                           all_measured_rounds_below=all(success),
                           fraction_measured_rounds_below=sum(success) / len(success),
                           fraction_late_rounds_below=sum(late) / len(late) if late else None,
                           max_failure_rate=max(failures.values()),
                           final_eta_mean=eta_stats[last][0])
            valid = [r for r in run["rows"] if not r["failed"]]
            denominators = np.asarray([r["eta"] ** 2 * r["grad_sq_norm"] for r in valid])
            row.update(zero_denominator_rows=int(np.sum(denominators == 0)),
                       positive_denominator_below_1e_8=int(np.sum((denominators > 0) & (denominators < 1e-8))),
                       min_step_denominator=float(denominators.min()) if len(denominators) else None,
                       max_observed_eta=max((r["eta"] for r in valid), default=None))
        report.append(row)
        print(json.dumps(row), flush=True)
    suffix = f"_gpu{args.gpu}" if args.gpu is not None else ""
    fields = list(dict.fromkeys(k for row in report for k in row))
    with (args.root / f"summary{suffix}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report)


if __name__ == "__main__":
    main()
