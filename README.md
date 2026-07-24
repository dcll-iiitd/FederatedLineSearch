### Requirements

Requirements can be found in the requirements.txt file.

### Instructions

- An example command to run an experiment is as follows:

```bash
python main.py --seed 0 --algorithm "fedavg" --dataset "CIFAR10" --model "resnet18" --num_clients 100 --num_participating_clients 5 --num_rounds 500 --alpha 0.5                 
```

Explanation of arguments:
1. `seed`: Choice of seed to fix randomness in experiment.
2. `algorithm`: Choice of algorithm. Possible options are `fedavg`, `fedexp`,`scaffold`, `scaffold(exp)`, `fedprox`, `fedprox(exp)`, `fedexprox`, `fedadam`, `fedadagrad`, `fedavgm`, `fedavgm(exp)`,`fedsls`,`fedexpsls`.
   
3. `dataset`: Choice of dataset. Possible options are `CIFAR10`,`CIFAR100`,`femnist`.

4. `model`: Choice of neural network model. Possible options are `resnet18`, `CNN` (for CIFAR dataset), `LOGISTIC_REGRESSION` (for femnist dataset),`LSTM`(for Shakespeare dataset).

5. `num_clients`: Total number of clients in FL system.

6. `num_participating_clients`: Number of clients that participate in each round of FL training.

7. `num_rounds`: Number of rounds of FL training

8. `alpha`: Choice of alpha parameter for the Dirichlet distribution used to create heterogeneity in the client datasets for CIFAR and CINIC datasets.


### FedExProx with constant extrapolation

Use `--algorithm fedexprox` to run inexact FedExProx with constant server extrapolation. Each selected client approximately solves the same proximal local objective used by FedProx, and the server applies `--fedexprox-alpha` (default `1.0`) to the same sample-weighted mean client update used by FedProx. Setting `--fedexprox-alpha 1.0` recovers ordinary FedProx aggregation; values greater than one extrapolate.

### ToyProblem

The Toy_problem folder contains code for comparing FedAvg, FedExp, FedSls and FedExpSls algorithms on a 2D toy problem.