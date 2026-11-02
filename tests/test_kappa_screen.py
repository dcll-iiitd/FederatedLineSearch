"""Small checks for extending measurement to both existing reset policies."""

import copy
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from SLS.sls import Sls
from utils_general import LocalUpdate_Sls


class KappaScreenTests(unittest.TestCase):
    def test_drop_last_is_explicit_and_tail_sizes_are_visible(self):
        dataset = TensorDataset(torch.zeros(500, 4), torch.zeros(500, dtype=torch.long))
        args = {"bs": 450, "device": "cpu", "cp": 1}
        hyper = {"eta_l": 0.1, "use_augmentation": False,
                 "use_gradient_clipping": False, "max_norm": 10,
                 "weight_decay": 0, "reset_option": 2, "eta_lmax": 0.1}
        historical = LocalUpdate_Sls(args, hyper, dataset)
        explicit = LocalUpdate_Sls({**args, "sls_drop_last": True}, hyper, dataset)
        self.assertEqual([len(y) for _, y in historical.ldr_train], [450, 50])
        self.assertEqual([len(y) for _, y in explicit.ldr_train], [450])

    def test_both_reset_options_preserve_updates(self):
        torch.manual_seed(73)
        template = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8),
                                 nn.ReLU(), nn.Dropout(0.2), nn.Linear(8, 3))
        images = torch.randn(32, 4)
        labels = torch.arange(32) % 3
        for option in (1, 2):
            plain = copy.deepcopy(template).train()
            measured = copy.deepcopy(template).train()
            a = Sls(plain.parameters(), init_step_size=0.1, reset_option=option)
            b = Sls(measured.parameters(), init_step_size=0.1, reset_option=option)
            generator = torch.Generator().manual_seed(100)
            loader = DataLoader(TensorDataset(images, labels), batch_size=13,
                                generator=generator)
            owner = SimpleNamespace(args={"device": "cpu"})

            def closure(model, optimizer):
                optimizer.zero_grad()
                return nn.functional.cross_entropy(model(images), labels)

            def reference():
                state = torch.get_rng_state().clone()
                value = LocalUpdate_Sls._reference_loss(owner, measured, loader)
                self.assertTrue(torch.equal(state, torch.get_rng_state()))
                return value

            for step in range(20):
                a.step_with_seed(lambda: closure(plain, a), 500 + step)
                b.step_with_kappa(lambda: closure(measured, b), reference,
                                  closure_seed=500 + step)
                for key, value in plain.state_dict().items():
                    self.assertTrue(torch.equal(value, measured.state_dict()[key]),
                                    f"option={option} step={step} key={key}")

    def test_client_fraction_is_stochastic_for_each_client(self):
        args = {"bs": 50, "device": "cpu", "cp": 1,
                "sls_drop_last": True, "sls_batch_fraction": 0.99}
        hyper = {"eta_l": 0.1, "use_augmentation": False,
                 "use_gradient_clipping": False, "max_norm": 10,
                 "weight_decay": 0, "reset_option": 2, "eta_lmax": 0.1}
        for n in (69, 258, 392, 500):
            dataset = TensorDataset(torch.zeros(n, 4), torch.zeros(n, dtype=torch.long))
            local = LocalUpdate_Sls(args, hyper, dataset)
            sizes = [len(y) for _, y in local.ldr_train]
            self.assertEqual(sizes, [int(n * 0.99)])
            self.assertLess(sizes[0], n)

    def test_full_batch_reference_agrees_at_both_iterates(self):
        torch.manual_seed(91)
        model = nn.Linear(4, 3)
        images = torch.randn(32, 4)
        labels = torch.arange(32) % 3
        optimizer = Sls(model.parameters(), init_step_size=0.1, reset_option=2)
        loader = DataLoader(TensorDataset(images, labels), batch_size=13,
                            generator=torch.Generator().manual_seed(2))
        owner = SimpleNamespace(args={"device": "cpu"})

        def closure():
            optimizer.zero_grad()
            return nn.functional.cross_entropy(model(images), labels)

        def reference():
            return LocalUpdate_Sls._reference_loss(owner, model, loader)

        _, _, failed, raw = optimizer.step_with_kappa(closure, reference, closure_seed=3)
        self.assertFalse(failed)
        self.assertAlmostEqual(raw["loss_prev_batch"], raw["f_ref_prev"], places=6)
        self.assertAlmostEqual(raw["loss_curr_batch"], raw["f_ref_curr"], places=6)


if __name__ == "__main__":
    unittest.main()
