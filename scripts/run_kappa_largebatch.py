#!/usr/bin/env python3
"""Detached, shared-queue follow-up with strictly stochastic large batches."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from run_kappa_screen import CODE, SOURCE, PYTHON, event, timestamp


def matrix():
    jobs = []

    def pair(dataset, model, batch=50, initial=0.01, seed=0, fraction=None):
        batch_token = f"frac{fraction:g}" if fraction is not None else f"b{batch}"
        for option in (2, 1):
            jobs.append({
                "name": f"{dataset.lower()}_{model.lower()}_{batch_token}_init{initial:g}_opt{option}_s{seed}",
                "dataset": dataset, "model": model, "batch_size": batch,
                "fraction": fraction, "eta_initial": initial,
                "reset_option": option, "seed": seed,
                "full_local_batch_control": False, "drop_last": True,
            })

    pair("femnist", "LOGISTIC_REGRESSION", initial=0.1, fraction=0.99)
    pair("CIFAR100", "CNN", batch=499)
    pair("femnist", "LOGISTIC_REGRESSION", initial=0.1, fraction=0.99, seed=1)
    pair("CIFAR100", "CNN", batch=450)
    pair("CIFAR100", "CNN", batch=490)
    pair("CIFAR100", "CNN", batch=499, seed=1)
    pair("CIFAR100", "CNN", batch=499, initial=0.1)
    pair("CIFAR10", "CNN", batch=499)
    pair("CIFAR10", "CNN", batch=490)
    pair("CIFAR10", "CNN", batch=499, seed=1)
    pair("CIFAR10", "CNN", batch=499, initial=0.1)
    pair("femnist", "LOGISTIC_REGRESSION", initial=0.1, fraction=0.95)
    pair("femnist", "LOGISTIC_REGRESSION", fraction=0.99)
    pair("femnist", "LOGISTIC_REGRESSION", fraction=0.99, seed=1)
    return jobs


def prepare(root, gpus, rounds):
    root.mkdir(parents=True, exist_ok=False)
    jobs = matrix()
    for job in jobs:
        job["rounds"] = rounds
        job["gpu"] = None  # A free worker claims the next job under a file lock.
        workdir = root / "jobs" / job["name"]
        workdir.mkdir(parents=True)
        for name in ("data", "femnist_data", "shakespeare"):
            (workdir / name).symlink_to(SOURCE / name, target_is_directory=True)
        job.update(workdir=str(workdir), stdout=str(root / f"{job['name']}.log"),
                   csv=str(root / f"{job['name']}.csv"))
        command = [
            str(PYTHON), "-u", str(CODE / "main.py"),
            "--seed", str(job["seed"]), "--algorithm", "fedsls",
            "--dataset", job["dataset"], "--model", job["model"],
            "--num_clients", "100", "--num_participating_clients", "20",
            "--num_rounds", str(rounds), "--alpha", "0.3",
            "--reset-option", str(job["reset_option"]),
            "--eta-lmax", str(job["eta_initial"]), "--armijo-c", "0.1",
            "--batch-size", str(job["batch_size"]),
            "--measure-kappa", "--kappa-measure-every", "10",
            "--deterministic-sls-seed", "--no-augmentation", "--sls-drop-last",
            "--kappa-output", job["csv"],
        ]
        if job["fraction"] is not None:
            command += ["--sls-batch-fraction", str(job["fraction"])]
        job["command"] = command
    manifest = {
        "created_utc": timestamp(), "code": str(CODE), "python": str(PYTHON),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODE, text=True).strip(),
        "gpus": gpus, "jobs": jobs,
        "notes": [
            "28-job predeclared exploratory follow-up; retain all outcomes.",
            "All batches strictly smaller than each client's full dataset.",
            "CIFAR b499/500=99.8%, b490/500=98%; b450/500=90% matched smaller-cap follow-up.",
            "FEMNIST uses floor(fraction*n_i) with fraction 0.95 or 0.99, including small clients.",
            "drop_last=True for every job to avoid tiny remainder batches.",
            "Full-client reference data, same-minibatch reuse, ratio of sums, no denominator epsilon.",
            "No augmentation, no gradient clipping, no changes to Armijo/backtracking/reset defaults.",
            "Deterministic closure seeds; CUDA deterministic algorithms are not forced.",
            "Opt1 bound uses effective_eta_lmax, not initial_step_size.",
            "A worker waits while any other compute process occupies its GPU.",
            "One job runs per available worker; no existing process is stopped or shared.",
            "Preplanned seed-1 replications do not depend on seed-0 passing.",
            "A pass means both aggregate ratios below the bound at all measured rounds; it is not a global proof.",
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


def states(root):
    latest = {}
    for path in root.glob("gpu*_status.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["name"] not in ("queue", "worker"):
                latest[row["name"]] = row["state"]
    return latest


def gpu_free(gpu):
    processes = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    memory = int(subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ], text=True).strip())
    return not processes and memory < 1024


def update_report(root, env):
    with (root / "summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with (root / "summary_generation.log").open("a") as output:
            subprocess.run([
                str(PYTHON), str(CODE / "scripts/summarize_kappa_screen.py"),
                "--root", str(root),
            ], cwd=CODE, env=env, stdout=output, stderr=subprocess.STDOUT, check=False)


def work(root, gpu):
    manifest = json.loads((root / "manifest.json").read_text())
    if gpu not in manifest["gpus"]:
        raise ValueError("GPU is not listed in the manifest")
    worker_lock = (root / f"worker_gpu{gpu}.lock").open("a")
    fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    event(root, gpu, "worker", "started", pid=os.getpid())
    while True:
        status = states(root)
        if all(job["name"] in status for job in manifest["jobs"]):
            break
        if not gpu_free(gpu):
            event(root, gpu, "worker", "waiting_for_free_gpu")
            time.sleep(30)
            continue
        with (root / "queue.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            status = states(root)
            job = next((j for j in manifest["jobs"] if j["name"] not in status), None)
            if job is None:
                break
            event(root, gpu, job["name"], "claimed", worker_pid=os.getpid())
        if Path(job["csv"]).exists() or Path(job["stdout"]).exists():
            event(root, gpu, job["name"], "refused_existing_output")
            continue
        with Path(job["stdout"]).open("x") as log:
            log.write("COMMAND " + shlex.join(job["command"]) + "\n")
            log.flush()
            process = subprocess.Popen(job["command"], cwd=job["workdir"], env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            event(root, gpu, job["name"], "running", pid=process.pid)
            returncode = process.wait()
        event(root, gpu, job["name"], "completed" if returncode == 0 else "failed",
              returncode=returncode)
        update_report(root, env)
        if returncode == 0:
            plots = root / "plots"
            plots.mkdir(exist_ok=True)
            with (plots / f"{job['name']}.log").open("w") as plot_log:
                subprocess.run([
                    str(PYTHON), str(CODE / "scripts/plot_kappa.py"),
                    "--input-glob", job["csv"],
                    "--output", str(plots / f"{job['name']}.pdf"),
                    "--title", job["name"],
                ], cwd=CODE, env=env, stdout=plot_log, stderr=subprocess.STDOUT, check=False)
    event(root, gpu, "queue", "finished")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 2])
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
