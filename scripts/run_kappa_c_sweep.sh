#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/kappa_runs

while true; do
  for gpu in 0 1 2 3; do
    active_pids="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"
    if [[ -z "$active_pids" ]]; then
      selected_gpu="$gpu"
      break 2
    fi
  done
  sleep 30
done

echo "Using physical GPU ${selected_gpu}" | tee results/kappa_runs/fedsls_eta01_c_sweep.status

export CUDA_VISIBLE_DEVICES="$selected_gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

for armijo_c in 0.3 0.5; do
  c_token="${armijo_c/./}"
  echo "Starting c=${armijo_c} at $(date --iso-8601=seconds)" | tee -a results/kappa_runs/fedsls_eta01_c_sweep.status
  python -u -c 'import runpy, torch; torch.use_deterministic_algorithms(True); runpy.run_path("main.py", run_name="__main__")' \
    --seed 0 \
    --algorithm fedsls \
    --dataset CIFAR10 \
    --model resnet18 \
    --num_clients 100 \
    --num_participating_clients 20 \
    --num_rounds 500 \
    --alpha 0.3 \
    --reset-option 2 \
    --eta-lmax 0.1 \
    --armijo-c "$armijo_c" \
    --measure-kappa \
    --kappa-measure-every 10 \
    --deterministic-sls-seed \
    > "results/kappa_runs/fedsls_eta01_c${c_token}_seed0.log" 2>&1
  echo "Finished c=${armijo_c} at $(date --iso-8601=seconds)" | tee -a results/kappa_runs/fedsls_eta01_c_sweep.status
done

python scripts/plot_kappa.py \
  --input-glob 'results/kappa_measurements_seed0_eta0p1*.csv' \
  --output results/kappa_ratio_eta01_c_sweep.png \
  > results/kappa_runs/plot_kappa_eta01_c_sweep.log 2>&1

echo "Plot complete at $(date --iso-8601=seconds)" | tee -a results/kappa_runs/fedsls_eta01_c_sweep.status
