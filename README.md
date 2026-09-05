# Federated Line Search

This repository contains implementations and experiment utilities for federated optimization with client-side stochastic Armijo line search, together with the fixed-learning-rate and adaptive-server baselines used in the accompanying experiments.

## Repository layout

- `main.py`: experiment entry point and server-side algorithm updates.
- `utils_general.py`: client-local solvers.
- `utils_data.py`: dataset loading and client partitioning.
- `utils_models.py`: model definitions.
- `SLS/`: stochastic Armijo line-search optimizer.
- `scripts/plotting/`: log parsing, seed averaging, and plotting utilities.
- `Toy_problem/`: two-dimensional illustrative experiments.

Generated logs, CSV files, plots, checkpoints, and local launch helpers are intentionally excluded from version control.

## Requirements

The tested Python dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The listed PyTorch build targets CUDA 11.6. If that exact wheel is unavailable for the local platform, install a compatible PyTorch build first and then install the remaining requirements.

## Experiments

The main experiments use the following datasets and models:

| Dataset | CLI value | Model |
|---|---|---|
| CIFAR-10 | `CIFAR10` | `resnet18` |
| CIFAR-100 | `CIFAR100` | `resnet18` |
| FEMNIST | `femnist` | `LOGISTIC_REGRESSION` |
| Shakespeare | `Shakespeare` | `LSTM` |

Implemented algorithm identifiers include:

```text
fedavg, fedexp, scaffold, feddyn,
fedprox, fedprox(exp), fedexprox,
fedadamsls, fedadamexpsls-scaled-regularized,
fedsls-regularized, fedexpsls-regularized,
fedadam, fedsls, fedexpsls
```

`fedsls` and `fedexpsls` use the stochastic Armijo line-search optimizer in `SLS/sls.py` for client-local optimization.

## Running an experiment

Run commands from the repository root. For example, the following command runs FedSLS on CIFAR-10 with the main experimental configuration:

```bash
mkdir -p runs/cifar10

CUDA_VISIBLE_DEVICES=0 nohup python -u main.py \
  --seed 0 \
  --algorithm fedsls \
  --dataset CIFAR10 \
  --model resnet18 \
  --num_clients 100 \
  --num_participating_clients 20 \
  --num_rounds 1000 \
  --alpha 0.3 \
  > runs/cifar10/fedsls_seed0.log 2>&1 &
```

Use a distinct log filename for every algorithm and seed. The recommended naming scheme is:

```text
runs/<dataset>/<algorithm>_seed<seed>.log
```

### Command-line arguments

- `--seed`: random seed for the run.
- `--algorithm`: algorithm identifier from the list above.
- `--dataset`: dataset identifier; values are case-sensitive.
- `--model`: model identifier associated with the selected dataset.
- `--num_clients`: total number of federated clients requested by the experiment.
- `--num_participating_clients`: clients sampled per communication round.
- `--num_rounds`: number of communication rounds.
- `--alpha`: Dirichlet concentration used for CIFAR client partitioning.
- `--fedexprox-alpha`: constant FedExProx server extrapolation factor; used only when `--algorithm fedexprox` is selected.

## FedExProx with constant extrapolation

Use `--algorithm fedexprox` to run the constant-extrapolation FedExProx variant. Selected clients approximately optimize the proximal local objective, and the server applies `--fedexprox-alpha` to the sample-weighted mean client update.

Setting

```bash
--fedexprox-alpha 1.0
```

recovers the ordinary FedProx aggregation step, while a value greater than one performs server extrapolation. The paper experiments use the value stated in the experimental configuration rather than relying on an implicit default.

## Averaging and plotting results

The repository includes utilities for extracting and averaging results from the text logs:

- `scripts/plotting/plot_fl_loss_acc.py`: extracts per-round training loss and test accuracy.
- `scripts/plotting/avg_seeds_metrics.py`: computes the per-round mean and standard deviation across seed CSV files.
- `scripts/plotting/dump_wallclock_csv.py`: extracts wall-clock metrics from a run log.
- `scripts/plotting/avg_wallclock.py`: averages wall-clock curves on a common time grid.
- `scripts/plotting/plot_avg_rounds.py`: plots averaged metrics against communication rounds.
- `scripts/plotting/plot_avg_time.py`: plots averaged metrics against wall-clock time.
- `scripts/plotting/plot_avg_line_search.py`: plots the five-seed mean and standard deviation of the average Armijo trials per local step for FedSLS and FedExpSLS.

For a fair wall-clock comparison, use the same time grid and a common horizon at which every displayed algorithm still has the required number of completed seeds.

## Generated outputs

Experiment logs are written under `runs/` by convention. Plotting utilities write derived CSV and figure files under `analysis/`. Both directories are local-only and ignored by Git.

## Toy problem

The `Toy_problem` directory contains the two-dimensional comparison of FedAvg, FedExp, FedSLS, and FedExpSLS.
