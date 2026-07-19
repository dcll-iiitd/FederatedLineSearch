### Requirements

Requirements can be found in the requirements.txt file.

### Instructions

- An example command to run an experiment is as follows:

```bash
python main.py --seed 0 --algorithm "fedavg" --dataset "CIFAR10" --model "resnet18" --num_clients 100 --num_participating_clients 5 --num_rounds 500 --alpha 0.5                 
```

Explanation of arguments:
1. `seed`: Choice of seed to fix randomness in experiment.
2. `algorithm`: Choice of algorithm. Possible options are `fedavg`, `fedexp`,`scaffold`, `scaffold(exp)`, `fedprox`, `fedprox(exp)`, `fedadam`, `fedadagrad`, `fedavgm`, `fedavgm(exp)`,`fedsls`,`fedexpsls`.
   
3. `dataset`: Choice of dataset. Possible options are `CIFAR10`,`CIFAR100`,`femnist`.

4. `model`: Choice of neural network model. Possible options are `resnet18`, `CNN` (for CIFAR dataset), `LOGISTIC_REGRESSION` (for femnist dataset),`LSTM`(for Shakespeare dataset).

5. `num_clients`: Total number of clients in FL system.

6. `num_participating_clients`: Number of clients that participate in each round of FL training.

7. `num_rounds`: Number of rounds of FL training

8. `alpha`: Choice of alpha parameter for the Dirichlet distribution used to create heterogeneity in the client datasets for CIFAR and CINIC datasets.


### ToyProblem

The Toy_problem folder contains code for comparing FedAvg, FedExp, FedSls and FedExpSls algorithms on a 2D toy problem.
### SLS reset policy

FedSLS and FedExpSLS accept two optional line-search arguments:

- `--reset-option {0,1,2}` (default `1`). Option 0 reuses the previous accepted step, option 1 grows the previous step before backtracking, and option 2 restarts from `init_step_size` every local step.
- `--eta-lmax FLOAT` (default `1.0`). This is passed to `Sls` as `init_step_size`. Under reset option 2, `init_step_size` is the analysis's `eta_lmax`.

These defaults reproduce the behavior that existed before the flags were added. Values selected for a paper experiment must be supplied on the command line; the default remains 1.0.

### Kappa-f measurement

Kappa measurement is available only for `fedsls` and `fedexpsls`, and requires reset option 2:

```bash
python -u main.py \
  --seed 0 \
  --algorithm fedsls \
  --dataset CIFAR10 \
  --model resnet18 \
  --num_clients 100 \
  --num_participating_clients 20 \
  --num_rounds 500 \
  --alpha 0.3 \
  --reset-option 2 \
  --eta-lmax 0.01 \
  --measure-kappa \
  --kappa-measure-every 10 \
  --deterministic-sls-seed
```

The new flags are:

- `--measure-kappa`: disabled by default. When absent, no reference sets are constructed and the original SLS step path is used.
- `--kappa-measure-every N`: measure rounds divisible by `N`; default `10`. Round numbers are zero-based, matching the training logs.
- `--deterministic-sls-seed`: disabled by default. When enabled, each SLS closure seed is derived from `(run_seed, round, client_id, local_step)` instead of wall-clock time. Use this for reproducible verification and kappa sweeps.

For clients with at most 1024 examples, the reference set is the complete local training dataset. CIFAR-10's equal partition has 500 examples per client, so its reference loss is the exact mean over that client's local data. Larger clients use one fixed 1024-example subset selected by a private generator seeded from the run seed and client ID.

Each run writes a cap-specific CSV such as
`results/kappa_measurements_seed0_eta0p01.csv` and a matching JSON metadata file.
The CSV contains:

- `round`, `client_id`, `local_step`, and `seed`: zero-based identifiers.
- `eta_returned`: the step size returned by Armijo.
- `loss_prev_batch`, `loss_curr_batch`: raw same-minibatch losses at the pre-step and accepted post-step iterates.
- `f_ref_prev`, `f_ref_curr`: reference-set mean losses at those two iterates.
- `grad_sq_norm`: squared norm of the same-minibatch pre-step gradient.
- `line_search_failed`: whether the 100-trial search failed. The post-step batch loss is NaN for these rows because the fallback step was not evaluated by the original line search.

The same minibatch is intentionally reused to compute the gradient, choose the
Armijo step, produce the new iterate, and record both batch-loss estimates. In
particular, the post-step estimate is intentionally biased because the new iterate
depends on that minibatch; do not replace it with a fresh or held-out batch.

For each seed and communication round, the plotter excludes failed searches and computes:

```text
R_a = sum(abs(loss_prev_batch - f_ref_prev))
      / sum(eta_returned^2 * grad_sq_norm)

R_b = sum(abs(loss_curr_batch - f_ref_curr))
      / sum(eta_returned^2 * grad_sq_norm)
```

This is a ratio of sums, not a mean of per-step ratios. Generate the combined plot with:

```bash
python scripts/plot_kappa.py
```

The script reads the actual Armijo `c` and `eta_lmax` from each JSON sidecar,
fails if either is unavailable, overlays all measured caps, prints per-round
line-search failure rates, and writes `results/kappa_ratio.png`.
