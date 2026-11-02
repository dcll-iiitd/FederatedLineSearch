#!/usr/bin/env python3
"""Prepare or run detached per-GPU queues for the 2026 kappa screening study."""

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


CODE = Path(__file__).resolve().parents[1]
SOURCE = CODE.with_name("FederatedLineSearch")
PYTHON = Path(sys.executable).resolve()


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def make_jobs(gpus, rounds):
    jobs = []

    def pair(worker, dataset, model, batch, initial=0.1, seed=0, control=False):
        for option in (2, 1):
            name = (
                f"{dataset.lower()}_{model.lower()}_b{batch}"
                f"_init{initial:g}_opt{option}_s{seed}"
            )
            jobs.append({
                "name": name, "worker": worker, "gpu": gpus[worker],
                "dataset": dataset, "model": model, "batch_size": batch,
                "eta_initial": initial, "reset_option": option, "seed": seed,
                "rounds": rounds, "full_local_batch_control": control,
            })

    pair(0, "femnist", "LOGISTIC_REGRESSION", 512, control=True)
    pair(0, "femnist", "LOGISTIC_REGRESSION", 256)
    pair(0, "femnist", "LOGISTIC_REGRESSION", 128)
    pair(0, "femnist", "LOGISTIC_REGRESSION", 128, initial=1.0)
    pair(0, "femnist", "LOGISTIC_REGRESSION", 256, seed=1)
    pair(0, "femnist", "LOGISTIC_REGRESSION", 128, initial=1.0, seed=1)

    pair(1, "CIFAR10", "CNN", 450)
    pair(1, "CIFAR10", "CNN", 500, control=True)
    pair(1, "CIFAR10", "CNN", 250)
    pair(1, "CIFAR10", "resnet18", 450)

    pair(2, "CIFAR100", "CNN", 450)
    pair(2, "CIFAR100", "CNN", 500, control=True)
    pair(2, "CIFAR100", "CNN", 250)
    return jobs


def event(root, gpu, name, state, **fields):
    value = {"time_utc": timestamp(), "gpu": gpu, "name": name,
             "state": state, **fields}
    with (root / f"gpu{gpu}_status.jsonl").open("a") as handle:
        handle.write(json.dumps(value) + "\n")
    print(json.dumps(value), flush=True)


def prepare(root, gpus, rounds):
    if len(gpus) != 3:
        raise ValueError("This screen uses exactly the three GPUs exposed by this host")
    root.mkdir(parents=True, exist_ok=False)
    jobs = make_jobs(gpus, rounds)
    for job in jobs:
        workdir = root / "jobs" / job["name"]
        workdir.mkdir(parents=True)
        for name in ("data", "femnist_data", "shakespeare"):
            (workdir / name).symlink_to(SOURCE / name, target_is_directory=True)
        job["workdir"] = str(workdir)
        job["stdout"] = str(root / f"{job['name']}.log")
        job["csv"] = str(root / f"{job['name']}.csv")
        job["command"] = [
            str(PYTHON), "-u", str(CODE / "main.py"),
            "--seed", str(job["seed"]), "--algorithm", "fedsls",
            "--dataset", job["dataset"], "--model", job["model"],
            "--num_clients", "100", "--num_participating_clients", "20",
            "--num_rounds", str(rounds), "--alpha", "0.3",
            "--reset-option", str(job["reset_option"]),
            "--eta-lmax", str(job["eta_initial"]), "--armijo-c", "0.1",
            "--batch-size", str(job["batch_size"]),
            "--measure-kappa", "--kappa-measure-every", "10",
            "--deterministic-sls-seed", "--no-augmentation",
            "--kappa-output", job["csv"],
        ]
        job["drop_last"] = job["dataset"] in ("CIFAR10", "CIFAR100") and job["batch_size"] == 450
        if job["drop_last"]:
            job["command"].append("--sls-drop-last")
    manifest = {
        "created_utc": timestamp(), "code": str(CODE), "python": str(PYTHON),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=CODE, text=True
        ).strip(),
        "gpus": gpus, "jobs": jobs,
        "notes": [
            "Exploratory 100-round screen, not a proof of the global assumption.",
            "Same-minibatch reuse and ratio-of-sums aggregation are intentional.",
            "No augmentation; model modes agree for LR, CNN, GroupNorm ResNet.",
            "All reference sets cover all local examples in the selected partitions.",
            "FEMNIST full control uses nominal b512: all 100 clients have <=392 examples.",
            "CIFAR full control uses b500: every client has exactly 500 examples.",
            "CIFAR b450 uses explicit drop_last=True so each step has 450, not alternating 450 and 50.",
            "Opt1 initial value is not a cap; use metadata effective_eta_lmax.",
            "Closure seeds are deterministic; deterministic_algorithms is not enabled.",
            "CUBLAS_WORKSPACE_CONFIG is unset. One GPU queue runs per available GPU.",
            "Promote promising stochastic settings to 500 rounds and more seeds later.",
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "code_changes.patch").write_text(subprocess.check_output(
        ["git", "diff", "--", "main.py", "utils_general.py", "scripts/plot_kappa.py", "README.md"],
        cwd=CODE, text=True,
    ))
    print(f"Prepared {len(jobs)} jobs in {root}")
    for gpu in gpus:
        print("setsid nohup " + shlex.join([
            str(PYTHON), "-u", str(Path(__file__).resolve()),
            "--root", str(root), "--worker-gpu", str(gpu),
        ]) + f" > {shlex.quote(str(root / f'gpu{gpu}_driver.log'))} 2>&1 < /dev/null &")


def work(root, gpu):
    manifest = json.loads((root / "manifest.json").read_text())
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4",
               MKL_NUM_THREADS="4", MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    for job in manifest["jobs"]:
        if job["gpu"] != gpu:
            continue
        if Path(job["csv"]).exists():
            event(root, gpu, job["name"], "skipped_existing_csv")
            continue
        with Path(job["stdout"]).open("x") as log:
            log.write("COMMAND " + shlex.join(job["command"]) + "\n")
            log.flush()
            process = subprocess.Popen(job["command"], cwd=job["workdir"],
                                       env=env, stdout=log, stderr=subprocess.STDOUT)
            event(root, gpu, job["name"], "running", pid=process.pid)
            returncode = process.wait()
        event(root, gpu, job["name"], "completed" if returncode == 0 else "failed",
              returncode=returncode)
        subprocess.run([
            str(PYTHON), str(CODE / "scripts/summarize_kappa_screen.py"),
            "--root", str(root), "--gpu", str(gpu),
        ], cwd=CODE, env=env, check=False)
    event(root, gpu, "queue", "finished")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--worker-gpu", type=int)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.root.resolve(), args.gpus, args.rounds)
    elif args.worker_gpu is not None:
        work(args.root.resolve(), args.worker_gpu)
    else:
        parser.error("Use --prepare or --worker-gpu")


if __name__ == "__main__":
    main()
