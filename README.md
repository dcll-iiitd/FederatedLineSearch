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
- `--armijo-c FLOAT` (default `0.1`). This overrides the Armijo sufficient-decrease constant for SLS runs; omitting it preserves historical behavior.

These defaults reproduce the behavior that existed before the flags were added. Values selected for a paper experiment must be supplied on the command line; the default remains 1.0.

### Kappa-f measurement

Kappa measurement is available only for `fedsls` and `fedexpsls`, with reset option 1 or 2:

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
- `--no-augmentation`: disabled by default and available only for SLS algorithms.
  Use it explicitly to study sampling error for an unaugmented local empirical
  objective. The reference evaluator already uses unaugmented local examples.
- `--sls-drop-last`: disabled by default and available only for SLS algorithms.
  The screening jobs with nominal CIFAR batch 450 enable it explicitly so all
  optimization batches contain 450 examples, rather than alternating 450/50.
  Every measurement CSV now records `batch_size_actual` to make tail batches
  visible. Other jobs keep the historical incomplete-batch behavior.
- `--sls-batch-fraction FLOAT`: disabled by default and SLS-only. When supplied,
  it overrides the nominal `--batch-size` separately for each client with
  `max(1, floor(fraction * n_i))`, where `0 < fraction < 1` and `n_i >= 2`.
  Pair with `--sls-drop-last` for fixed near-full stochastic batches on unequal
  FEMNIST client sizes. Reference sets still contain the full client data in the
  current screening partitions. Metadata records the fraction and effective sizes.

For reset option 2, `--eta-lmax` is the trial-step cap. For option 1 it is only the
initial step. A fresh optimizer is created for each client's 20-step local solve,
so an upper envelope is `max(init_step_size, 1e-6) * gamma**(20/B)`, where `gamma`
and `B=n_batches_per_epoch` come from the actual optimizer defaults. The `1e-6`
accounts for the optimizer's existing failed-search fallback. The metadata records
this envelope as `effective_eta_lmax`; the plotter uses
`c/(2*effective_eta_lmax)` for the bound. With initial step 0.1, the current defaults
give option 1 a bound of approximately 0.486327 and option 2 a bound of 0.5.
No reset, Armijo, or fallback behavior is changed by measurement.

The 2026 screening launcher is `scripts/run_kappa_screen.py`. It creates an
explicit manifest, separate working directories and CSV/log files per job, and
one detached queue per available GPU. Results are written under the main
repository's `results/kappa_screen_*` directory. Screens use 100 rounds, both reset
options, full-client reference losses, no augmentation, and deterministic closure
seeds without enabling `torch.use_deterministic_algorithms` or setting
`CUBLAS_WORKSPACE_CONFIG`. `scripts/summarize_kappa_screen.py --root PATH` reports
all jobs, failure rates, ratios, zero/tiny denominator counts, and the fraction of
measured rounds satisfying both inequalities. Full-local-batch runs are positive
controls; they do not establish the condition for stochastic minibatches. These
training-trajectory diagnostics do not establish the global assumption or a uniform
bound across every client and iterate. Promising stochastic settings need longer
runs and independent seed/checkpoint validation.

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

### Fixed-state conditional accuracy studies

The later conditional studies use `scripts/run_kappa_conditional.py` and
`scripts/kappa_conditional_core.py`. At a frozen client/model/optimizer state,
they run independent one-step branches and form separate numerator and
denominator means. They do not pool changing client states as a substitute for
the conditional expectations. Batch-50 measurements use Monte Carlo trials;
leave-one-out batches enumerate every omitted example; full batches are controls.
The driver preserves the existing main training paths and Armijo/reset rules.

The [completed experiment handout](docs/kappa_accuracy_handoff_20261029/README.md)
documents CIFAR-10, CIFAR-100, FEMNIST and Shakespeare, including all settings,
numerical policies, limitations and rebuttal drafts. Compact
[configuration results](docs/kappa_accuracy_handoff_20261029/conditional_results.csv)
and [state breakdowns](docs/kappa_accuracy_handoff_20261029/conditional_state_breakdown.csv)
are versioned alongside it. Raw trial files, checkpoints, datasets and logs
remain in the experiment folders on the server.

These studies support a restrictive, state-dependent sufficient condition, not
uniform stochastic validation. Full-batch satisfaction is expected by
construction; the tested stochastic configurations were not supported at every
tested state. The completed conditional studies use reset option 2 only. The
[larger protocol proposal](docs/kappa_conditional_experiment_protocol.md) is not
a record of additional completed experiments.

Run the focused CPU checks without launching training:

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg \
  python -m unittest discover -s tests -v
```
