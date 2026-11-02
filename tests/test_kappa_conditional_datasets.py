import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_kappa_conditional import (
    effective_batch, expected_trials, make_model, make_optimizer, natural_cohort_indices,
)
from scripts.summarize_kappa_conditional import summarize


class DatasetProtocolTests(unittest.TestCase):
    def test_variable_client_sizes_and_counts(self):
        for n in (69, 108, 392, 500):
            self.assertEqual(effective_batch("loo", n), n - 1)
            self.assertEqual(effective_batch("full", n), n)
            self.assertEqual(effective_batch(50, n), 50)
            self.assertEqual(expected_trials(n - 1, n), n)
            self.assertEqual(expected_trials(n, n), 1)
            self.assertEqual(expected_trials(50, n), 128)

    def test_femnist_model_and_cap(self):
        config = {"model": "LOGISTIC_REGRESSION", "num_classes": 62,
                  "initial_step": 0.1, "armijo_c": 0.1}
        model = make_model("cpu", config=config)
        optimizer = make_optimizer(model, config)
        self.assertEqual(model.linear.out_features, 62)
        self.assertEqual(optimizer.param_groups[0]["init_step_size"], 0.1)
        self.assertEqual(optimizer.param_groups[0]["reset_option"], 2)

    def test_cifar100_classifier_and_cap(self):
        config = {"model": "CNN", "num_classes": 100, "initial_step": 0.1}
        model = make_model("cpu", config=config)
        optimizer = make_optimizer(model, config)
        self.assertEqual(model.fc3.out_features, 100)
        self.assertEqual(optimizer.param_groups[0]["init_step_size"], 0.1)

    def test_shakespeare_natural_cohort_and_lstm(self):
        self.assertEqual(natural_cohort_indices([2, 40, 52, 492, 501, 33044], 2, 500), [2, 3])
        with self.assertRaises(ValueError):
            natural_cohort_indices([2, 40, 52], 2, 500)
        import torch
        model = make_model("cpu", config={"model": "LSTM", "num_classes": 68})
        inputs = torch.randint(0, 68, (3, 80))
        output = model(inputs)
        self.assertEqual(output.shape, (3, 68))
        self.assertEqual(model.lstm.dropout, 0)

    def test_report_groups_variable_batches_and_uses_actual_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "cells").mkdir()
            (root / "manifest.json").write_text(json.dumps({
                "batches": [50, "loo", "full"], "bound": 0.5,
                "exact_batches": ["loo", "full"], "expected_cells_per_batch": 2,
            }))
            for i, n in enumerate((69, 108)):
                value = {"batch_setting": "loo", "batch_size": n - 1,
                         "client_id": i, "R_a": 0.1, "R_b": 0.2,
                         "status": "enumerated_below", "failed_searches": 0,
                         "trials": n, "zero_denominator_trials": 0,
                         "positive_denominator_below_1e_8": 0,
                         "both_point_estimates_below": True}
                (root / "cells" / f"cell{i}.json").write_text(json.dumps(value))
            rows = summarize(root, plots=False)
            loo = next(row for row in rows if row["batch_setting"] == "loo")
            self.assertEqual(loo["completed_states"], 2)
            self.assertEqual(loo["actual_batch_min"], 68)
            self.assertEqual(loo["actual_batch_max"], 107)
            self.assertEqual(loo["bound"], 0.5)
            self.assertAlmostEqual(loo["candidate_common_kappa_on_tested_states"], 0.35)


if __name__ == "__main__":
    unittest.main()
