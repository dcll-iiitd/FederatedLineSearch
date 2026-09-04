#!/usr/bin/env bash

# Run five canonical FedAdam seeds sequentially for one dataset.
# Launch one copy of this script per available GPU. It does not start by itself.
#
# Examples:
#   nohup bash fedadam.sh cifar10 0 > runs/cifar10/fedadam_canonical_queue.log 2>&1 &
#   nohup bash fedadam.sh cifar100 1 > runs/cifar100/fedadam_canonical_queue.log 2>&1 &
#   nohup bash fedadam.sh femnist 2 > runs/femnist/fedadam_canonical_queue.log 2>&1 &
#   nohup bash fedadam.sh shakespeare 3 > runs/shakespeare/fedadam_canonical_queue.log 2>&1 &
#
# beta1=0.9 and beta2=0.99 are defined in main.py. Rates and tau below are
# the FedAdam values reported in Tables 8 and 9 of Reddi et al., Adaptive
# Federated Optimization (ICLR 2021). Client rates remain constant.
# The surrounding protocol remains this repository's protocol: 100 clients,
# 20 participants, K=20, batch size 50, alpha=0.3, and 1000 rounds.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash fedadam.sh {cifar10|cifar100|femnist|shakespeare} GPU_ID" >&2
  exit 2
fi

RUN_KEY=$1
GPU_ID=$2

case "$RUN_KEY" in
  cifar10)
    DATASET=CIFAR10; MODEL=resnet18; RUN_DIR=runs/cifar10
    ETA_L=0.03162277660168379; ETA_G=0.01; TAU=0.001
    ;;
  cifar100)
    DATASET=CIFAR100; MODEL=resnet18; RUN_DIR=runs/cifar100
    ETA_L=0.03162277660168379; ETA_G=1.0; TAU=0.1
    ;;
  femnist)
    DATASET=femnist; MODEL=LOGISTIC_REGRESSION; RUN_DIR=runs/femnist
    ETA_L=0.03162277660168379; ETA_G=0.0031622776601683794; TAU=0.0001
    ;;
  shakespeare)
    DATASET=shakespeare; MODEL=LSTM; RUN_DIR=runs/shakespeare
    ETA_L=1.0; ETA_G=0.01; TAU=0.001
    ;;
  *)
    echo "Unknown dataset key: $RUN_KEY" >&2
    exit 2
    ;;
esac

mkdir -p "$RUN_DIR"
echo "Canonical FedAdam queue"
echo "dataset=$DATASET model=$MODEL gpu=$GPU_ID eta_l=$ETA_L eta_g=$ETA_G tau=$TAU"

for SEED in 0 1 2 3 4; do
  LOG_PATH="$RUN_DIR/fedadam_canonical_seed${SEED}.log"
  if [[ -e "$LOG_PATH" ]]; then
    echo "Refusing to overwrite existing log: $LOG_PATH" >&2
    exit 1
  fi

  echo "Starting seed=$SEED log=$LOG_PATH"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python -u main.py \
    --seed "$SEED" \
    --algorithm fedadam \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --num_clients 100 \
    --num_participating_clients 20 \
    --num_rounds 1000 \
    --alpha 0.3 \
    --fedadam-eta-l "$ETA_L" \
    --fedadam-constant-client-lr \
    --eta-g "$ETA_G" \
    --fedadam-tau "$TAU" \
    --fedadam-tau-outside-sqrt \
    > "$LOG_PATH" 2>&1
  echo "Completed seed=$SEED"
done

echo "Completed all five seeds for $RUN_KEY"
