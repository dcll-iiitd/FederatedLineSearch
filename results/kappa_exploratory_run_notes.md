# Exploratory kappa_f measurement runs

All runs use FedSLS with Armijo `c=0.1`, `eta_lmax=0.1`, reset option 2,
`--measure-kappa`, and `--kappa-measure-every 10`. Measurement is opt-in and
the repository defaults remain unchanged. These exploratory jobs intentionally
do not enable deterministic algorithms or `CUBLAS_WORKSPACE_CONFIG`.

Runs:

- CIFAR-10 / ResNet-18 / batch 50 / seed 0
- CIFAR-10 / ResNet-18 / batch 128 / seed 0
- CIFAR-10 / ResNet-18 / batch 256 / seed 0
- FEMNIST / logistic regression / batch 50 (the repository default) / seed 0
- FEMNIST / logistic regression / batch 50 (the repository default) / seed 1

Each concurrent job writes a distinct CSV under `results/`.

The CIFAR-10 sweep tests the prediction that minibatch estimation error scales
approximately as `1/sqrt(batch_size)`, so increasing the batch size should lower
the measured ratios without changing the theoretical bound. FEMNIST with logistic
regression is the convex setting assumed by the theory and is therefore expected
to satisfy kappa_f-accuracy more readily than non-convex ResNet-18.

For every seed and communication round, plotting aggregates numerator and
denominator separately over clients and local steps, excludes
`line_search_failed` rows, and divides once. It never averages per-step ratios.
The bound for every run is `c / (2 * eta_lmax) = 0.5`.
