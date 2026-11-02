import copy
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from SLS.sls import Sls
from scripts.kappa_conditional_core import (
    batch_trials, conditional_rows, reference_loss, summarize_rows,
)


class ConditionalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.model = nn.Linear(2, 2)
        self.x = torch.randn(5, 2)
        self.y = torch.tensor([0, 1, 0, 1, 0])

    def test_exact_enumeration_matches_manual_branches(self):
        optimizer = Sls(self.model.parameters(), init_step_size=0.01, reset_option=2)
        original = copy.deepcopy(self.model.state_dict())
        rows = list(conditional_rows(self.model, optimizer, self.x, self.y, 4, 128, 10))
        self.assertEqual([r["excluded_index"] for r in rows], list(range(5)))
        self.assertEqual(len({r["anchor_hash"] for r in rows}), 1)
        for omitted, row in enumerate(rows):
            manual = nn.Linear(2, 2)
            manual.load_state_dict(original)
            opt = Sls(manual.parameters(), init_step_size=0.01, reset_option=2)
            indices = torch.tensor([i for i in range(5) if i != omitted])
            before = reference_loss(manual, self.x, self.y)
            prev = F.cross_entropy(manual(self.x[indices]), self.y[indices]).item()

            def closure():
                opt.zero_grad()
                return F.cross_entropy(manual(self.x[indices]), self.y[indices])

            opt.step_with_seed(closure, 13)
            curr = F.cross_entropy(manual(self.x[indices]), self.y[indices]).item()
            after = reference_loss(manual, self.x, self.y)
            self.assertAlmostEqual(row["abs_fdiff_prev"], abs(prev - before), places=6)
            self.assertAlmostEqual(row["abs_fdiff_curr"], abs(curr - after), places=6)
        for key, value in original.items():
            self.assertTrue(torch.equal(self.model.state_dict()[key], value))

    def test_restores_state_and_rng_and_does_not_chain_trials(self):
        optimizer = Sls(self.model.parameters(), init_step_size=0.01, reset_option=1)
        optimizer.state["step"] = 19
        optimizer.state["step_size"] = 0.003
        frozen = copy.deepcopy(self.model.state_dict())
        opt_frozen = copy.deepcopy(optimizer.state_dict())
        rng = torch.get_rng_state().clone()
        np_before = np.random.get_state()

        def observe(trial, model, opt):
            for key, value in frozen.items():
                self.assertTrue(torch.equal(model.state_dict()[key], value))
            self.assertEqual(opt.state_dict(), opt_frozen)

        list(conditional_rows(self.model, optimizer, self.x, self.y, 2, 16, 2,
                              observer=observe))
        self.assertEqual(optimizer.state_dict(), opt_frozen)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(np.array_equal(np_before[1], np.random.get_state()[1]))

    def test_failed_fallback_is_retained_and_evaluated(self):
        model = nn.Linear(2, 2).double()
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
        x = torch.ones(4, 2, dtype=torch.double)
        y = torch.zeros(4, dtype=torch.long)
        # c>1 forces Armijo failure for this convex loss, only in this test.
        optimizer = Sls(model.parameters(), init_step_size=0.01, reset_option=2, c=2)
        rows = list(conditional_rows(model, optimizer, x, y, 3, 128, 9))
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["line_search_failed"] for row in rows))
        self.assertTrue(all(row["finite"] for row in rows))
        self.assertTrue(all(row["eta_returned"] == 1e-6 for row in rows))
        self.assertEqual(summarize_rows(rows, 4, 5)["failed_searches"], 4)

    def test_full_batch_and_zero_denominator(self):
        optimizer = Sls(self.model.parameters(), init_step_size=0.01, reset_option=2)
        rows = list(conditional_rows(self.model, optimizer, self.x, self.y, 5, 128, 5))
        self.assertEqual(len(rows), 1)
        self.assertLess(rows[0]["abs_fdiff_prev"], 1e-6)
        self.assertLess(rows[0]["abs_fdiff_curr"], 1e-6)
        model = nn.Linear(2, 1)
        optimizer = Sls(model.parameters(), init_step_size=0.01, reset_option=2)
        rows = list(conditional_rows(model, optimizer, self.x, torch.zeros(5, dtype=torch.long), 5, 1, 5))
        self.assertEqual(summarize_rows(rows, 1, 5)["status"], "degenerate_exact")

    def test_sampling_is_reproducible_but_not_repeated_same_batch(self):
        a = [indices.tolist() for _, _, indices in batch_trials(5, 2, 16, 13)]
        b = [indices.tolist() for _, _, indices in batch_trials(5, 2, 16, 13)]
        self.assertEqual(a, b)
        self.assertGreater(len({tuple(v) for v in a}), 1)
        self.assertTrue(all(len(v) == len(set(v)) == 2 and v == sorted(v) for v in a))


if __name__ == "__main__":
    unittest.main()
