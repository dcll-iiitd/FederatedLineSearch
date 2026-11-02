#!/usr/bin/env python3
"""Reduced fixed-state kappa study: b50 MC, b499 exhaustive, b500 control."""

import argparse
import contextlib
import copy
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

CODE = Path(__file__).resolve().parents[1]
SOURCE = CODE.with_name("FederatedLineSearch")
sys.path.insert(0, str(CODE))

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader

from SLS.sls import Sls
from utils_models import get_model
from scripts.kappa_conditional_core import (
    conditional_rows, reference_loss, seed_key, state_hash, summarize_rows, uniform_batch,
)


def atomic_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def emit(root, gpu, name, state, **fields):
    record = {"time_utc": datetime.now(timezone.utc).isoformat(), "gpu": gpu,
              "name": name, "state": state, **fields}
    with (root / f"gpu{gpu}_status.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def read_config(root):
    return json.loads((root / "manifest.json").read_text())


def effective_batch(setting, n):
    value = n - 1 if setting == "loo" else n if setting == "full" else int(setting)
    if not 1 <= value <= n:
        raise ValueError(f"Invalid batch setting {setting} for client size {n}")
    return value


def natural_cohort_indices(sizes, count, maximum):
    eligible = [i for i, n in enumerate(sizes) if 50 < n <= maximum]
    if len(eligible) < count:
        raise ValueError("Not enough natural clients in the declared size range")
    return eligible[:count]


def expected_trials(batch_size, n, mc_trials=128):
    return 1 if batch_size == n else n if batch_size == n - 1 else mc_trials


def make_model(device, seed=0, config=None):
    config = config or {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = get_model(config.get("model", "CNN"), config.get("num_classes", 10))
    return model.to(device).train()


def make_optimizer(model, config=None):
    config = config or {}
    return Sls(model.parameters(), init_step_size=config.get("initial_step", 0.01),
               reset_option=2, c=config.get("armijo_c", 0.1),
               max_grad_norm=None)


def load_data(root, label):
    from utils_data import get_dataset
    config = read_config(root)
    cwd = Path.cwd()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    try:
        os.chdir(SOURCE)
        np.random.seed(0)
        torch.manual_seed(0)
        with (root / f"partition_{label}.log").open("w") as log, contextlib.redirect_stdout(log):
            datasets, test = get_dataset(config["dataset"], config["num_clients"],
                                         config.get("num_classes", 10), 0.3, True)
        selected_ids = None
        if config["dataset"] == "shakespeare":
            # Keep the original character mapping and every example of each
            # selected natural writer. Only the experimental cohort is filtered.
            raw = json.loads(Path("shakespeare/all_data_niid_2_keep_0_train_8.json").read_text())
            selected = natural_cohort_indices([len(dataset) for dataset in datasets],
                                              config["num_clients"], config["max_client_size"])
            selected_ids = [raw["users"][i] for i in selected]
            datasets = [datasets[i] for i in selected]
            test = torch.utils.data.ConcatDataset([test.datasets[i] for i in selected])
        clients = [dataset.tensors if hasattr(dataset, "tensors") else (dataset.x, dataset.y)
                   for dataset in datasets]
        if len(clients) != config["num_clients"] or any(len(y) <= 50 for x, y in clients):
            raise ValueError("Protocol client count or minimum dataset size does not match")
        if config["dataset"] in ("CIFAR10", "CIFAR100") and any(len(y) != 500 for x, y in clients):
            raise ValueError("Equal-partition CIFAR protocol requires 500 examples/client")
        loader = DataLoader(test, batch_size=1000, shuffle=False,
                            generator=torch.Generator().manual_seed(300))
        test_batches = list(loader)
        digest = hashlib.sha256()
        for x, y in clients:
            digest.update(x.numpy().tobytes())
            digest.update(y.numpy().tobytes())
        fingerprint = {"sha256": digest.hexdigest(), "clients": len(clients),
                       "client_sizes": [len(y) for x, y in clients]}
        if selected_ids is not None:
            fingerprint["selected_writer_ids"] = selected_ids
            fingerprint["max_client_size"] = config["max_client_size"]
            fingerprint["sequence_length"] = int(clients[0][0].shape[1])
            if any(int(x.max()) >= config["num_classes"] or int(y.max()) >= config["num_classes"]
                   for x, y in clients):
                raise ValueError("Shakespeare indices exceed the LSTM vocabulary")
        with (root / "partition.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = root / "partition.json"
            if path.exists():
                previous = json.loads(path.read_text())
                if previous["sha256"] != fingerprint["sha256"] or previous["clients"] != len(clients):
                    raise RuntimeError("Partition differs across workers")
            if not path.exists():
                atomic_json(path, fingerprint)
        return clients, test_batches
    finally:
        os.chdir(cwd)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def snapshot_code(root):
    if any(root.glob("gpu*_status.jsonl")):
        raise RuntimeError("Snapshot source before launching workers")
    config = read_config(root)
    paths = set(config["source_sha256"]) | {"utils_libs.py", "scripts/summarize_kappa_conditional.py"}
    target = root / "source_snapshot"
    target.mkdir(exist_ok=False)
    hashes = {}
    for relative in sorted(paths):
        source = CODE / relative
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    config["source_sha256"] = hashes
    config["source_snapshot"] = str(target)
    atomic_json(root / "manifest.json", config)


def local_step(model, optimizer, x, y, batch_size, generator, closure_seed):
    indices = uniform_batch(len(y), batch_size, generator).to(x.device)

    def closure():
        optimizer.zero_grad()
        return F.cross_entropy(model(x[indices]), y[indices])

    return optimizer.step_with_seed(closure, closure_seed)


def evaluate(model, clients, tests, device):
    result = []
    for batches in (clients, tests):
        loss_sum = 0.0
        correct = 0
        count = 0
        model.eval()
        with torch.no_grad():
            for x_cpu, y_cpu in batches:
                for start in range(0, len(y_cpu), 256):
                    x, y = x_cpu[start:start+256].to(device), y_cpu[start:start+256].to(device)
                    output = model(x)
                    loss_sum += F.cross_entropy(output, y, reduction="sum").item()
                    correct += int((output.argmax(1) == y).sum().item())
                    count += len(y)
        result.extend((loss_sum / count, 100.0 * correct / count))
    model.train()
    return result


def train_job(root, job, clients, tests, device, gpu):
    config = read_config(root)
    setting = job["batch_size"]
    model, local = make_model(device, config=config), make_model(device, config=config)
    total_samples = sum(len(y) for x, y in clients)
    selection = np.random.default_rng(seed_key("client_selection", 0))
    started = time.perf_counter()
    path = root / "training" / f"b{setting}.csv"
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "round", "elapsed_sec", "round_sec", "train_loss", "train_acc",
            "test_loss", "test_acc", "failed_searches",
        ])
        writer.writeheader()
        for t in range(100):
            round_start = time.perf_counter()
            if t in (0, 99):
                checkpoint = root / "checkpoints" / f"b{setting}_t{t}.pt"
                value = {"model": cpu_state(model), "batch_setting": setting,
                         "round": t, "seed": 0, "partition_seed": 0}
                temporary = checkpoint.with_suffix(".tmp")
                torch.save(value, temporary)
                temporary.replace(checkpoint)
            base = parameters_to_vector(model.parameters()).detach().clone()
            update_sum = torch.zeros_like(base)
            selected = selection.choice(len(clients), 20, replace=False)
            failures = 0
            weight_sum = 0.0
            for client in selected:
                local.load_state_dict(model.state_dict())
                local.train()
                optimizer = make_optimizer(local, config)
                x_cpu, y_cpu = clients[int(client)]
                x, y = x_cpu.to(device), y_cpu.to(device)
                batch_size = effective_batch(setting, len(y))
                weight = len(y) / total_samples
                generator = torch.Generator().manual_seed(seed_key("training_batch", 0, t, int(client)))
                for k in range(1, 21):
                    _, _, failed = local_step(local, optimizer, x, y, batch_size, generator,
                                               seed_key("training_closure", 0, t, int(client), k))
                    failures += int(failed)
                with torch.no_grad():
                    update_sum += weight * (parameters_to_vector(local.parameters()) - base)
                    weight_sum += weight
            with torch.no_grad():
                vector_to_parameters(base + update_sum / weight_sum, model.parameters())
            metrics = evaluate(model, clients, tests, device) if t % 10 == 0 or t == 99 else [None] * 4
            row = dict(round=t, elapsed_sec=time.perf_counter()-started,
                       round_sec=time.perf_counter()-round_start,
                       train_loss=metrics[0], train_acc=metrics[1], test_loss=metrics[2],
                       test_acc=metrics[3], failed_searches=failures)
            writer.writerow(row)
            handle.flush()
            emit(root, gpu, job["name"], "progress", round=t, **{k:v for k,v in row.items() if k != "round"})
    torch.save({"model": cpu_state(model), "seed": 0, "batch_setting": setting},
               root / "checkpoints" / f"b{setting}_final.pt")


def diagnostic_job(root, job, clients, device, gpu):
    config = read_config(root)
    setting, t = job["batch_size"], job["round"]
    checkpoint = root / "checkpoints" / f"b{setting}_t{t}.pt"
    saved = torch.load(checkpoint, map_location="cpu")
    rollout, probe = make_model(device, config=config), make_model(device, config=config)
    for client in range(job["client_start"], job["client_end"]):
        rollout.load_state_dict(saved["model"])
        rollout.train()
        optimizer = make_optimizer(rollout, config)
        x_cpu, y_cpu = clients[client]
        x, y = x_cpu.to(device), y_cpu.to(device)
        batch_size = effective_batch(setting, len(y))
        generator = torch.Generator().manual_seed(seed_key("shadow_batch", 0, setting, t, client))
        for k in range(1, 21):
            if k in (1, 20):
                cell_name = f"b{setting}_t{t}_i{client:03d}_k{k}"
                probe.load_state_dict(rollout.state_dict())
                probe.train()
                probe_optimizer = make_optimizer(probe, config)
                probe_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                anchor = {"global_checkpoint": str(checkpoint), "round": t,
                          "client_id": client, "local_step": k, "batch_size": batch_size,
                          "batch_setting": setting, "client_size": len(y),
                          "training_seed": 0, "partition_seed": 0,
                          "anchor_kind": "independent_diagnostic_client_rollout",
                          "optimizer": copy.deepcopy(optimizer.state_dict()),
                          "model_hash": state_hash(rollout.state_dict()),
                          "rollout_seed": seed_key("shadow_batch", 0, setting, t, client)}
                if k == 20:
                    anchor["model"] = cpu_state(rollout)
                torch.save(anchor, root / "anchors" / f"{cell_name}.pt")
                context = {"batch_size": batch_size, "training_seed": 0,
                           "round": t, "client_id": client, "local_step": k,
                           "batch_setting": setting, "client_size": len(y)}
                observations = []
                path = root / "trials" / f"{cell_name}.csv"
                with path.open("x", newline="") as handle:
                    writer = None
                    for row in conditional_rows(
                            probe, probe_optimizer, x, y, batch_size, config["mc_trials"],
                            seed_key("diagnostic_batches", 0, setting, t, client, k)):
                        complete_row = {**context, **row}
                        if writer is None:
                            writer = csv.DictWriter(handle, fieldnames=list(complete_row))
                            writer.writeheader()
                        writer.writerow(complete_row)
                        observations.append(row)
                expected = expected_trials(batch_size, len(y), config["mc_trials"])
                summary = {**context, "anchor_hash": anchor["model_hash"],
                           "anchor_path": str(root / "anchors" / f"{cell_name}.pt"),
                           **summarize_rows(observations, expected, config["bound"])}
                atomic_json(root / "cells" / f"{cell_name}.json", summary)
                emit(root, gpu, job["name"], "cell_complete", cell=cell_name,
                     trials=expected, R_a=summary.get("R_a"), R_b=summary.get("R_b"),
                     cell_status=summary["status"])
            if k < 20:
                local_step(rollout, optimizer, x, y, batch_size, generator,
                           seed_key("shadow_closure", 0, setting, t, client, k))


def prepare(root, dataset="CIFAR10", eta_lmax=0.01, gpus=None,
            num_clients=100, max_client_size=None):
    if not np.isfinite(eta_lmax) or eta_lmax <= 0:
        raise ValueError("eta_lmax must be finite and positive")
    cifar = dataset in ("CIFAR10", "CIFAR100")
    if num_clients < 20 or (cifar and num_clients != 100):
        raise ValueError("Need at least 20 clients; CIFAR protocol uses exactly 100")
    if dataset == "shakespeare" and max_client_size is None:
        raise ValueError("Declare the Shakespeare cohort's maximum client size explicitly")
    settings = [50, 499, 500] if cifar else [50, "loo", "full"]
    root.mkdir(parents=True, exist_ok=False)
    for name in ("checkpoints", "anchors", "trials", "cells", "training", "plots", "logs"):
        (root / name).mkdir()
    jobs = [{"name": f"train_b{b}", "kind": "training", "batch_size": b}
            for b in settings]
    # Once a checkpoint exists, any free GPU can diagnose a ten-client shard.
    for b in (settings[1], settings[0], settings[2]):
        for t in (0, 99):
            for start in range(0, num_clients, 10):
                jobs.append({"name": f"diagnose_b{b}_t{t}_i{start:03d}",
                             "kind": "diagnostics", "batch_size": b, "round": t,
                             "client_start": start, "client_end": min(num_clients, start + 10)})
    sources = [CODE / "SLS/sls.py", CODE / "SLS/utils.py", CODE / "utils_data.py",
               CODE / "utils_models.py", Path(__file__), CODE / "scripts/kappa_conditional_core.py"]
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "gpus": gpus or [0, 1, 2],
        "code": str(CODE), "python": sys.executable, "training_seed": 0,
        "partition_seed": 0, "dataset": dataset,
        "model": "CNN" if cifar else "LSTM" if dataset == "shakespeare" else "LOGISTIC_REGRESSION",
        "num_classes": 100 if dataset == "CIFAR100" else 10 if cifar else 68 if dataset == "shakespeare" else 62,
        "num_clients": num_clients, "max_client_size": max_client_size,
        "participating_clients": 20,
        "local_steps": 20, "training_rounds": 100, "global_checkpoints": [0, 99],
        "local_checkpoints": [1, 20], "batches": settings,
        "reset_option": 2, "initial_step": eta_lmax, "eta_upper": eta_lmax,
        "armijo_c": 0.1, "bound": 0.1 / (2 * eta_lmax), "mc_trials": 128,
        "expected_cells_per_batch": 4 * num_clients,
        "sampling": "IID uniform subsets across steps; distinct within batch; sorted indices",
        "reference": "all client-local training examples, eval/no_grad, chunks 256",
        "aggregation": "ratio of trial means at a single frozen state; never pool states",
        "augmentation": False, "gradient_clipping": False, "weight_decay": 0,
        "failed_searches": "include fallback outcome and flag theoretical compatibility",
        "denominator_epsilon": 0, "exact_batches": settings[1:],
        "mc_interpretation": "exploratory conditional mean estimate, not simultaneous confidence certification",
        "torch": torch.__version__, "source_sha256": {
            str(path.relative_to(CODE)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources}, "jobs": jobs,
    }
    atomic_json(root / "manifest.json", config)
    clients, _ = load_data(root, "prepare")
    config["client_sizes"] = [len(y) for x, y in clients]
    if dataset == "shakespeare":
        fingerprint = json.loads((root / "partition.json").read_text())
        config["selected_writer_ids"] = fingerprint["selected_writer_ids"]
        config["sequence_length"] = fingerprint["sequence_length"]
        config["cohort_rule"] = "natural writers with 50 < training samples <= max_client_size; no truncation"
    config["expected_trials_by_setting"] = {
        str(setting): sum(4 * expected_trials(effective_batch(setting, len(y)), len(y))
                          for x, y in clients) for setting in settings
    }
    config["expected_trials"] = sum(config["expected_trials_by_setting"].values())
    config["server_aggregation"] = "sample-count-weighted client deltas normalized over selected clients"
    atomic_json(root / "manifest.json", config)
    print(f"Prepared {len(jobs)} tasks; {12 * num_clients} frozen states; {config['expected_trials']} diagnostic trials")


def latest_states(root):
    result = {}
    for path in root.glob("gpu*_status.jsonl"):
        for line in path.read_text().splitlines():
            value = json.loads(line)
            if value["state"] != "progress" and value["state"] != "cell_complete":
                result[value["name"]] = value["state"]
    return result


def gpu_free(gpu):
    output = subprocess.check_output(["nvidia-smi", "-i", str(gpu),
        "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    return not output.strip()


def update_reports(root):
    with (root / "report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with (root / "report_generation.log").open("a") as log:
            subprocess.run([sys.executable, str(CODE / "scripts/summarize_kappa_conditional.py"),
                            "--root", str(root)], stdout=log, stderr=subprocess.STDOUT, check=False)


def worker(root, gpu):
    manifest = json.loads((root / "manifest.json").read_text())
    if gpu not in manifest["gpus"]:
        raise ValueError("GPU not permitted by manifest")
    worker_lock = (root / f"gpu{gpu}.lock").open("a")
    fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while not gpu_free(gpu):
        emit(root, gpu, "worker", "waiting_for_free_gpu")
        time.sleep(30)
    clients, tests = load_data(root, f"gpu{gpu}")
    device = torch.device("cuda:0")
    emit(root, gpu, "worker", "started", pid=os.getpid())
    while True:
        if shutil.disk_usage(root).free < 2 * 1024**3:
            emit(root, gpu, "worker", "stopped_low_disk_space")
            return
        with (root / "queue.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            states = latest_states(root)
            job = None
            for candidate in manifest["jobs"]:
                if candidate["name"] in states:
                    continue
                if candidate["kind"] == "diagnostics":
                    checkpoint = root / "checkpoints" / f"b{candidate['batch_size']}_t{candidate['round']}.pt"
                    if not checkpoint.exists():
                        continue
                job = candidate
                emit(root, gpu, job["name"], "running", pid=os.getpid())
                break
            if job is None:
                pending = [j for j in manifest["jobs"] if j["name"] not in states]
                active = any(v == "running" for k, v in states.items() if k.startswith("train_"))
                if not pending or not active:
                    break
        if job is None:
            time.sleep(15)
            continue
        try:
            if job["kind"] == "training":
                train_job(root, job, clients, tests, device, gpu)
            else:
                diagnostic_job(root, job, clients, device, gpu)
            emit(root, gpu, job["name"], "completed")
        except Exception as error:
            import traceback
            traceback.print_exc()
            emit(root, gpu, job["name"], "failed", error=str(error))
        update_reports(root)
    emit(root, gpu, "worker", "finished")


def benchmark(root, trials):
    config = read_config(root)
    clients, tests = load_data(root, "benchmark")
    device = torch.device("cuda:0")
    model = make_model(device, config=config)
    x, y = clients[0][0].to(device), clients[0][1].to(device)
    baseline = copy.deepcopy(model.state_dict())
    timing = {}
    for setting in config["batches"]:
        b = effective_batch(setting, len(y))
        model.load_state_dict(baseline)
        opt = make_optimizer(model, config)
        list(conditional_rows(model, opt, x, y, b, trials, 12, limit=2))
        torch.cuda.synchronize()
        start = time.perf_counter()
        rows = list(conditional_rows(model, opt, x, y, b, trials, 13,
                                     limit=trials if b != 500 else 1))
        torch.cuda.synchronize()
        diagnostic_sec = (time.perf_counter() - start) / len(rows)
        generator = torch.Generator().manual_seed(33)
        model.load_state_dict(baseline)
        opt = make_optimizer(model, config)
        start = time.perf_counter()
        for k in range(trials):
            local_step(model, opt, x, y, b, generator, k + 900)
        torch.cuda.synchronize()
        training_sec = (time.perf_counter() - start) / trials
        # Later local states can require more backtracking than initialization.
        start = time.perf_counter()
        later_rows = list(conditional_rows(model, opt, x, y, b, trials, 14,
                                          limit=trials if b != len(y) else 1))
        torch.cuda.synchronize()
        later_sec = (time.perf_counter() - start) / len(later_rows)
        timing[str(setting)] = {"diagnostic_seconds_per_trial": max(diagnostic_sec, later_sec),
                          "early_diagnostic_seconds_per_trial": diagnostic_sec,
                          "later_diagnostic_seconds_per_trial": later_sec,
                          "training_seconds_per_step": training_sec,
                          "actual_batch": b, "reference_size": len(y),
                          "measured_trials": len(rows)}
        print(f"BENCHMARK setting={setting} b={b} early={diagnostic_sec:.6f}s later={later_sec:.6f}s training={training_sec:.6f}s", flush=True)
    training_total = sum(v["training_seconds_per_step"] * 40000 for v in timing.values())
    counts = config.get("expected_trials_by_setting", {"50": 51200, "499": 200000, "500": 400})
    diagnostic_total = sum(timing[str(setting)]["diagnostic_seconds_per_trial"] * n
                           for setting, n in counts.items())
    shadow_total = sum(v["training_seconds_per_step"] * 3800 for v in timing.values())
    gpu_count = len(config["gpus"])
    ideal_hours = (training_total + diagnostic_total + shadow_total) / (gpu_count * 3600)
    result = {"timing": timing, "gpus": gpu_count,
              "estimated_gpu_work_hours": (training_total + diagnostic_total + shadow_total) / 3600,
              "ideal_parallel_hours": ideal_hours,
              "estimated_hours_low": ideal_hours * 1.2,
              "estimated_hours_high": ideal_hours * 2.0,
              "basis": "early and after-local-updates pilot; later global states, I/O, and sharing may increase time"}
    atomic_json(root / "benchmark.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-trials", type=int, default=30)
    parser.add_argument("--worker-gpu", type=int)
    parser.add_argument("--dataset", choices=("CIFAR10", "CIFAR100", "femnist", "shakespeare"), default="CIFAR10")
    parser.add_argument("--eta-lmax", type=float, default=0.01)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--num-clients", type=int, default=100)
    parser.add_argument("--max-client-size", type=int, default=None)
    parser.add_argument("--snapshot-code", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if args.prepare:
        prepare(args.root.resolve(), dataset=args.dataset, eta_lmax=args.eta_lmax,
                gpus=args.gpus, num_clients=args.num_clients,
                max_client_size=args.max_client_size)
    elif args.snapshot_code:
        snapshot_code(args.root.resolve())
    elif args.benchmark:
        benchmark(args.root.resolve(), args.benchmark_trials)
    elif args.worker_gpu is not None:
        worker(args.root.resolve(), args.worker_gpu)
    else:
        parser.error("Choose --prepare, --benchmark, or --worker-gpu")


if __name__ == "__main__":
    main()
