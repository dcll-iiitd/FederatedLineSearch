import copy
import time

import torch

from utils_libs import *
from SLS import utils as ut


def _flat_to_tensor_list(flat_tensor, params):
    tensors = []
    offset = 0
    for param in params:
        numel = param.numel()
        tensors.append(flat_tensor[offset:offset + numel].view_as(param))
        offset += numel
    return tensors


def _squared_norm(tensor_list):
    total = None
    for tensor in tensor_list:
        if tensor is None:
            continue
        value = torch.sum(tensor * tensor)
        total = value if total is None else total + value
    if total is None:
        return torch.tensor(0.0)
    return total


class ScaffoldSls(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        n_batches_per_epoch=500,
        init_step_size=1,
        c=0.1,
        beta_b=0.9,
        gamma=2.0,
        beta_f=2.0,
        reset_option=1,
        eta_max=10,
        bound_step_size=True,
        line_search_fn="armijo",
    ):
        defaults = dict(
            n_batches_per_epoch=n_batches_per_epoch,
            init_step_size=init_step_size,
            c=c,
            beta_b=beta_b,
            gamma=gamma,
            beta_f=beta_f,
            reset_option=reset_option,
            eta_max=eta_max,
            bound_step_size=bound_step_size,
            line_search_fn=line_search_fn,
        )
        super().__init__(params, defaults)

        self.state["step"] = 0
        self.state["step_size"] = init_step_size
        self.state["n_forwards"] = 0
        self.state["n_backwards"] = 0

    def step(self, closure, control_variate):
        seed = time.time()

        def closure_deterministic():
            with ut.random_seed_torch(int(seed)):
                return closure()

        batch_step_size = self.state["step_size"]

        loss = closure_deterministic()
        loss.backward()

        self.state["n_forwards"] += 1
        self.state["n_backwards"] += 1

        accepted_step_size = batch_step_size

        for group in self.param_groups:
            params = group["params"]
            params_current = copy.deepcopy(params)
            grad_current = ut.get_grad_list(params)
            correction_list = _flat_to_tensor_list(control_variate, params)
            direction_list = [g - corr for g, corr in zip(grad_current, correction_list)]

            direction_norm_sq = _squared_norm(direction_list)
            grad_norm_sq = _squared_norm(grad_current)
            _ = _squared_norm(correction_list)

            step_size = ut.reset_step(
                step_size=batch_step_size,
                n_batches_per_epoch=group["n_batches_per_epoch"],
                gamma=group["gamma"],
                reset_option=group["reset_option"],
                init_step_size=group["init_step_size"],
            )

            with torch.no_grad():
                if direction_norm_sq >= 1e-12:
                    found = 0

                    for _ in range(100):
                        ut.try_sgd_update(params, step_size, params_current, direction_list)

                        loss_next = closure_deterministic()
                        self.state["n_forwards"] += 1

                        if group["line_search_fn"] != "armijo":
                            raise ValueError("ScaffoldSls currently supports only armijo line search.")

                        c_sls = group["c"]
                        rhs = (
                            loss
                            - (step_size / 2.0) * grad_norm_sq
                            - c_sls * (step_size / 2.0) * direction_norm_sq
                        )

                        if loss_next <= rhs:
                            found = 1
                            break

                        step_size = step_size * group["beta_b"]

                    if found == 0:
                        ut.try_sgd_update(params, 1e-6, params_current, direction_list)
                        step_size = 1e-6

            accepted_step_size = step_size
            self.state["step_size"] = step_size
            self.state["step"] += 1

        return loss, accepted_step_size


class LocalUpdate_ScaffoldSls(object):
    def __init__(self, args, args_hyperparameters, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args["bs"], shuffle=True)
        self.lr = args_hyperparameters["eta_l"]
        self.use_data_augmentation = args_hyperparameters["use_augmentation"]
        self.transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        )

    def train_and_sketch(self, net, idx, mem_mat, c):
        net.train()

        optimizer = ScaffoldSls(
            net.parameters(),
            n_batches_per_epoch=max(1, len(self.ldr_train)),
            init_step_size=1,
        )

        prev_net = copy.deepcopy(net)
        control_variate = mem_mat[idx] - c
        step_count = 0
        total_accepted_step_size = 0.0

        while True:
            for _, (images, labels) in enumerate(self.ldr_train):
                images = images.to(self.args["device"])
                labels = labels.to(self.args["device"])

                if self.use_data_augmentation:
                    images = self.transform_train(images)

                def closure():
                    optimizer.zero_grad()
                    output = net(images)
                    return self.loss_func(output, labels)

                _, accepted_step_size = optimizer.step(closure, control_variate)
                total_accepted_step_size += float(accepted_step_size)
                step_count += 1

                if step_count >= self.args["cp"]:
                    break

            if step_count >= self.args["cp"]:
                break

        with torch.no_grad():
            vec_curr = parameters_to_vector(net.parameters())
            vec_prev = parameters_to_vector(prev_net.parameters())
            params_delta_vec = vec_curr - vec_prev
            effective_step_size = max(total_accepted_step_size, 1e-12)
            mem_mat[idx] = (mem_mat[idx] - c) - params_delta_vec / effective_step_size

        return params_delta_vec
