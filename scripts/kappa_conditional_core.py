"""Fixed-state conditional trials. The existing SLS solver is not modified."""

import copy
import hashlib
import math

import numpy as np
import torch
from torch.nn import functional as F


def seed_key(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:4], "little")


def uniform_batch(n, b, generator):
    if not 1 <= b <= n:
        raise ValueError("Invalid batch size")
    if b == n:
        return torch.arange(n)
    return torch.randperm(n, generator=generator)[:b].sort().values


def batch_trials(n, b, count, seed):
    if b == n:
        yield "full_enumeration", -1, torch.arange(n)
    elif b == n - 1:
        all_indices = torch.arange(n)
        for omitted in range(n):
            yield "leave_one_out_enumeration", omitted, all_indices[all_indices != omitted]
    else:
        generator = torch.Generator().manual_seed(seed)
        for _ in range(count):
            yield "monte_carlo", -1, uniform_batch(n, b, generator)


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def reference_loss(model, images, labels, chunk=256):
    modules = list(model.modules())
    modes = [module.training for module in modules]
    total = 0.0
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, len(labels), chunk):
                total += F.cross_entropy(
                    model(images[start:start + chunk]), labels[start:start + chunk],
                    reduction="sum",
                ).item()
    finally:
        for module, mode in zip(modules, modes):
            module.training = mode
    return total / len(labels)


def conditional_rows(model, optimizer, images, labels, batch_size, mc_trials, seed,
                     limit=None, observer=None):
    """Yield one A/B/D observation per independent branch, resetting every time.

    All minibatch outcomes are retained, including failed-search fallback updates.
    limit is a timing/test-only cap; production callers leave it unset.
    """
    frozen = {name: value.detach().clone() for name, value in model.state_dict().items()}
    frozen_optimizer = copy.deepcopy(optimizer.state_dict())
    frozen_grads = [None if p.grad is None else p.grad.detach().clone()
                    for p in model.parameters()]
    frozen_numpy = np.random.get_state()
    anchor_hash = state_hash(frozen)
    f_prev = reference_loss(model, images, labels)
    try:
        for trial, (method, omitted, indices_cpu) in enumerate(
                batch_trials(len(labels), batch_size, mc_trials, seed)):
            if limit is not None and trial >= limit:
                break
            model.load_state_dict(frozen)
            optimizer.load_state_dict(copy.deepcopy(frozen_optimizer))
            if observer is not None:
                observer(trial, model, optimizer)
            indices = indices_cpu.to(images.device)
            batch_x, batch_y = images[indices], labels[indices]

            def closure():
                optimizer.zero_grad()
                return F.cross_entropy(model(batch_x), batch_y)

            calls = 0

            def reference():
                nonlocal calls
                calls += 1
                return f_prev if calls == 1 else reference_loss(model, images, labels)

            _, forwards, failed, raw = optimizer.step_with_kappa(
                closure, reference, closure_seed=seed_key(seed, "closure", trial)
            )
            if calls != 2:
                raise RuntimeError("Conditional evaluation requires one SLS parameter group")
            if failed:
                # Existing SLS does not evaluate its fallback batch loss. Evaluate
                # it here so failed minibatches remain in the unconditional mean.
                with torch.no_grad():
                    raw["loss_curr_batch"] = float(F.cross_entropy(model(batch_x), batch_y).cpu())
            a = abs(raw["loss_prev_batch"] - raw["f_ref_prev"])
            b = abs(raw["loss_curr_batch"] - raw["f_ref_curr"])
            d = raw["eta_returned"] ** 2 * raw["grad_sq_norm"]
            finite = all(math.isfinite(value) for value in (
                a, b, d, raw["eta_returned"], raw["grad_sq_norm"],
                raw["loss_prev_batch"], raw["loss_curr_batch"],
                raw["f_ref_prev"], raw["f_ref_curr"],
            ))
            yield {
                "trial_id": trial, "sampling_method": method,
                "excluded_index": omitted,
                "batch_indices_sha256": hashlib.sha256(indices_cpu.numpy().tobytes()).hexdigest(),
                "batch_size_actual": len(indices_cpu), "anchor_hash": anchor_hash,
                **raw,
                "abs_fdiff_prev": a, "abs_fdiff_curr": b, "denominator": d,
                "line_search_forwards": forwards, "finite": finite,
            }
    finally:
        model.load_state_dict(frozen)
        optimizer.load_state_dict(frozen_optimizer)
        for parameter, gradient in zip(model.parameters(), frozen_grads):
            parameter.grad = gradient
        np.random.set_state(frozen_numpy)


def summarize_rows(rows, expected, bound):
    count = len(rows)
    finite = all(row["finite"] for row in rows)
    exact = bool(rows) and rows[0]["sampling_method"] != "monte_carlo"
    complete = count == expected
    failed = sum(bool(row["line_search_failed"]) for row in rows)
    result = {
        "trials": count, "expected_trials": expected, "complete": complete,
        "exact_enumeration": exact, "all_finite": finite,
        "failed_searches": failed, "bound": bound,
        "zero_denominator_trials": sum(row["denominator"] == 0 for row in rows),
        "positive_denominator_below_1e_8": sum(0 < row["denominator"] < 1e-8 for row in rows),
    }
    if not count or not finite or not complete:
        return {**result, "status": "invalid_or_incomplete", "R_a": None, "R_b": None}
    mean_a = math.fsum(row["abs_fdiff_prev"] for row in rows) / count
    mean_b = math.fsum(row["abs_fdiff_curr"] for row in rows) / count
    mean_d = math.fsum(row["denominator"] for row in rows) / count
    result.update(mean_abs_fdiff_prev=mean_a, mean_abs_fdiff_curr=mean_b,
                  mean_denominator=mean_d,
                  mean_slack_a=mean_a - bound * mean_d,
                  mean_slack_b=mean_b - bound * mean_d,
                  mean_eta=math.fsum(row["eta_returned"] for row in rows) / count,
                  max_eta=max(row["eta_returned"] for row in rows))
    if mean_d <= 0:
        status = ("degenerate_exact" if mean_a == mean_b == 0 else "violated_zero_denominator") if exact else "inconclusive_zero_denominator"
        return {**result, "status": status, "R_a": None, "R_b": None}
    a, b = mean_a / mean_d, mean_b / mean_d
    below = max(a, b) < bound
    status = ("enumerated_below" if below else "enumerated_above") if exact else (
        "mc_estimate_below" if below else "mc_estimate_above")
    return {**result, "status": status, "R_a": a, "R_b": b,
            "both_point_estimates_below": below,
            "armijo_compatible": failed == 0}
