"""SCAFFOLD with the requested control-variate-aware stochastic line search."""

import copy
import time

import torch
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader
from torchvision import transforms

from SLS import utils as sls_utils


def _unflatten(flat, params):
    result, offset = [], 0
    for param in params:
        result.append(flat[offset:offset + param.numel()].view_as(param))
        offset += param.numel()
    if offset != flat.numel():
        raise ValueError("Control variate and model parameters have different sizes")
    return result


def _norm_sq(values):
    return sum((torch.sum(value * value) for value in values), start=0.0)


class ScaffoldSlsNew(torch.optim.Optimizer):
    def __init__(self, params, n_batches_per_epoch=500, init_step_size=1.0,
                 eta_max=10.0, beta=0.9, sigma=0.1, gamma=2.0,
                 reset_option=1, max_backtracks=100, fallback_step_size=1e-6,
                 acceptance_rule="original", alpha=0.1, beta_slack=None, eta_cap=None):
        if eta_max <= 0:
            raise ValueError("eta_max must be positive")
        if not 0 < beta < 1:
            raise ValueError("beta must lie in (0, 1)")
        if not 0 < sigma < 0.5:
            raise ValueError("sigma must lie in (0, 1/2)")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if acceptance_rule in ("drift_slack", "quadratic_drift_slack") and (beta_slack is None or beta_slack < 0):
            raise ValueError("drift_slack requires a non-negative beta_slack")
        if eta_cap is not None and eta_cap <= 0:
            raise ValueError("eta_cap must be positive")
        defaults = dict(n_batches_per_epoch=n_batches_per_epoch,
                        init_step_size=init_step_size, eta_max=eta_max,
                        beta=beta, sigma=sigma, gamma=gamma,
                        reset_option=reset_option, max_backtracks=max_backtracks,
                        fallback_step_size=fallback_step_size,
                        acceptance_rule=acceptance_rule, alpha=alpha,
                        beta_slack=beta_slack, eta_cap=eta_cap)
        super().__init__(params, defaults)
        self.state["step"] = 0
        self.state["step_size"] = init_step_size
        self.state["n_forwards"] = 0
        self.state["n_backwards"] = 0

    def step(self, closure, scaffold_correction, collect_diagnostics=False):
        # Reuse the stochastic realization for the gradient and all trial losses.
        seed = time.time()

        def deterministic_closure():
            with sls_utils.random_seed_torch(int(seed)):
                return closure()

        loss = deterministic_closure()
        loss.backward()
        self.state["n_forwards"] += 1
        self.state["n_backwards"] += 1
        accepted_eta = self.state["step_size"]
        search_forwards, search_failed = 0, False
        backtrack_trials = 0
        diagnostic = None

        for group in self.param_groups:
            params = group["params"]
            params_current = copy.deepcopy(params)
            gradients = [g.detach().clone() for g in sls_utils.get_grad_list(params)]
            correction = _unflatten(scaffold_correction, params)
            direction = [g + h for g, h in zip(gradients, correction)]
            grad_norm_sq = _norm_sq(gradients)
            correction_norm_sq = _norm_sq(correction)
            direction_norm_sq = _norm_sq(direction)

            eta_before_cap = float(sls_utils.reset_step(
                self.state["step_size"], group["n_batches_per_epoch"],
                group["gamma"], group["reset_option"], group["init_step_size"]))
            cap_active = (group["eta_cap"] is not None
                          and eta_before_cap > group["eta_cap"])
            eta = (min(eta_before_cap, float(group["eta_cap"]))
                   if group["eta_cap"] is not None else eta_before_cap)

            with torch.no_grad():
                loss_next = loss.detach()
                if direction_norm_sq >= 1e-12:
                    found = False
                    for _ in range(group["max_backtracks"]):
                        sls_utils.try_sgd_update(params, eta, params_current, direction)
                        loss_next = deterministic_closure()
                        self.state["n_forwards"] += 1
                        backtrack_trials += 1
                        search_forwards += 1
                        if group["acceptance_rule"] == "original":
                            rhs = (loss - 0.5 * eta * grad_norm_sq
                                   + 0.5 * eta * correction_norm_sq
                                   - 0.5 * group["sigma"] * eta * direction_norm_sq)
                        elif group["acceptance_rule"] == "grad_control":
                            rhs = (loss
                                   - 0.5 * (1.0 - group["sigma"]) * eta
                                   * grad_norm_sq
                                   + 0.25 * (2.0 + group["sigma"]) * eta
                                   * correction_norm_sq)
                        elif group["acceptance_rule"] == "no_control_reward":
                            rhs = (loss
                                   - 0.5 * eta * grad_norm_sq
                                   - 0.5 * group["sigma"] * eta
                                   * direction_norm_sq)
                        elif group["acceptance_rule"] == "quadratic_drift_slack":
                            rhs = (loss
                                   - group["alpha"] * eta * grad_norm_sq
                                   + 0.5 * group["beta_slack"] * eta * eta
                                   * correction_norm_sq)
                        elif group["acceptance_rule"] == "drift_slack":
                            rhs = (loss
                                   - group["alpha"] * eta * grad_norm_sq
                                   + group["beta_slack"] * eta
                                   * correction_norm_sq)
                        elif group["acceptance_rule"] == "surrogate_armijo":
                            # h_i(w) = f_i(w) + <c-c_i, w>.
                            surrogate_delta = sum(
                                torch.sum(h * (p_next - p_current))
                                for h, p_next, p_current in zip(
                                    correction, params, params_current)
                            )
                            rhs = -group["sigma"] * eta * direction_norm_sq
                        else:
                            raise ValueError(
                                f"Unknown acceptance rule: {group['acceptance_rule']}"
                            )
                        if (loss_next - loss + surrogate_delta <= rhs
                                if group["acceptance_rule"] == "surrogate_armijo"
                                else loss_next <= rhs):
                            found = True
                            break
                        eta *= group["beta"]
                    if not found:
                        search_failed = True
                        eta = group["fallback_step_size"]
                        sls_utils.try_sgd_update(params, eta, params_current, direction)

                        if collect_diagnostics:
                            # The fallback was not one of the evaluated trials.
                            # Evaluate it so f_after is the actual same-batch loss.
                            loss_next = deterministic_closure()
                            self.state["n_forwards"] += 1
                            search_forwards += 1

            if collect_diagnostics:
                affine_before = sum(
                    torch.sum(h * p_current)
                    for h, p_current in zip(correction, params_current)
                )
                affine_after = sum(
                    torch.sum(h * p_next)
                    for h, p_next in zip(correction, params)
                )
                measured_surrogate_change = (
                    loss_next.detach() - loss.detach()
                    + sum(torch.sum(h * (p_next - p_current))
                          for h, p_next, p_current in zip(
                              correction, params, params_current))
                )
                measured_armijo_rhs = (
                    -group["sigma"] * eta * direction_norm_sq
                )
                diagnostic = {
                    "rule": group["acceptance_rule"],
                    "beta_slack": group["beta_slack"],
                    "sls_alpha": group["alpha"],
                    "eta": float(eta),
                    "eta_before_cap": eta_before_cap,
                    "eta_after_cap": float(eta),
                    "cap_active": int(cap_active),
                    "grad_norm": float(torch.sqrt(grad_norm_sq).item()),
                    "direction_norm": float(torch.sqrt(direction_norm_sq).item()),
                    "f_before": float(loss.detach().item()),
                    "f_after": float(loss_next.detach().item()),
                    "h_before": float((loss.detach() + affine_before).item()),
                    "h_after": float((loss_next.detach() + affine_after).item()),
                    "surrogate_change": float(measured_surrogate_change.item()),
                    "armijo_rhs": float(measured_armijo_rhs.item()),
                    "armijo_margin": float(
                        (measured_armijo_rhs - measured_surrogate_change).item()),
                    "accepted_update_norm": float(
                        eta * torch.sqrt(direction_norm_sq).item()),
                    "backtrack_trials": int(backtrack_trials),
                    "search_failed": int(search_failed),
                }

            accepted_eta = eta
            self.state["step_size"] = eta
            self.state["step"] += 1
        return loss, accepted_eta, search_forwards, search_failed, diagnostic


class LocalUpdateScaffoldSlsNew:
    def __init__(self, args, args_hyperparameters, dataset=None,
                 acceptance_rule="original", beta_slack=None, alpha=0.1, eta_cap=None):
        self.args = args
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(dataset, batch_size=args["bs"], shuffle=True)
        self.use_augmentation = args_hyperparameters["use_augmentation"]
        self.acceptance_rule = acceptance_rule
        self.beta_slack = beta_slack
        self.alpha = alpha
        self.eta_cap = eta_cap
        self.reset_option = args_hyperparameters.get("sls_reset_option", 1)
        self.transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])

    def train_and_sketch(self, net, idx, mem_mat, server_control,
                         round_idx=None, collect_diagnostics=False):
        net.train()
        optimizer = ScaffoldSlsNew(
            net.parameters(), n_batches_per_epoch=500,
            acceptance_rule=self.acceptance_rule,
            beta_slack=self.beta_slack,
            alpha=self.alpha,
            eta_cap=self.eta_cap,
            reset_option=self.reset_option
        )
        previous_net = copy.deepcopy(net)
        client_control = mem_mat[idx].to(self.args["device"])
        correction = server_control - client_control
        drift = float(torch.linalg.vector_norm(correction).item())
        steps = 0
        eta_sum = 0.0
        search_forwards = 0
        failed_searches = 0
        diagnostics = []

        while steps < self.args["cp"]:
            for images, labels in self.ldr_train:
                images, labels = images.to(self.args["device"]), labels.to(self.args["device"])
                if self.use_augmentation:
                    images = self.transform(images)

                def closure():
                    optimizer.zero_grad()
                    return self.loss_func(net(images), labels)

                _, eta, forwards, failed, diagnostic = optimizer.step(
                    closure, correction,
                    collect_diagnostics=collect_diagnostics
                )
                if diagnostic is not None:
                    diagnostic.update({
                        "round": int(round_idx),
                        "client_id": int(idx),
                        "local_step": int(steps),
                        "drift": drift,
                    })
                    diagnostics.append(diagnostic)
                eta_sum += float(eta)
                search_forwards += forwards
                failed_searches += int(failed)
                steps += 1
                if steps >= self.args["cp"]:
                    break

        with torch.no_grad():
            params_delta = (parameters_to_vector(net.parameters())
                            - parameters_to_vector(previous_net.parameters()))
            # Variable-step counterpart of the standard SCAFFOLD K*eta update.
            new_client_control = (client_control - server_control
                                  - params_delta / max(eta_sum, 1e-12))
            mem_mat[idx].copy_(new_client_control.detach().cpu())

            client_update_norm = float(torch.linalg.vector_norm(params_delta).item())
            control_update_norm = float(torch.linalg.vector_norm(
                new_client_control - client_control).item())
            normalized_client_update_norm = float(
                client_update_norm / max(eta_sum, 1e-12))

        for diagnostic in diagnostics:
            diagnostic.update({
                "eta_sum": float(eta_sum),
                "client_update_norm": client_update_norm,
                "normalized_client_update_norm": normalized_client_update_norm,
                "control_update_norm": control_update_norm,
            })

        stats = {"local_steps": steps,
                 "line_search_forwards": search_forwards,
                 "total_forwards": steps + search_forwards,
                 "total_backwards": steps,
                 "failed_searches": failed_searches,
                 "final_step_size": float(optimizer.state["step_size"]),
                 "diagnostics": diagnostics}
        return params_delta, stats
