#!/usr/bin/env bash
set -euo pipefail

# ---------------- CONFIG ----------------
GPUS=(0 1 2 3)                      # <-- put your GPU ids here, e.g. (0 1 2 3)
PYTHON=python                   # or python3
MAIN=main.py
LOGDIR="logs"
LOCKDIR=".gpu_locks"

mkdir -p "$LOGDIR" "$LOCKDIR"

# ---------------- GPU LOCKING ----------------
pick_gpu_lock() {
  # Blocks until a GPU lock is available, then prints: "<gpu_id> <fd>"
  while true; do
    for g in "${GPUS[@]}"; do
      local lockfile="$LOCKDIR/gpu${g}.lock"
      exec {fd}>"$lockfile"
      if flock -n "$fd"; then
        echo "$g $fd"
        return 0
      fi
    done
    sleep 10
  done
}

run_one() {
  local alg="$1"
  local seed="$2"
  local dataset="$3"
  local model="$4"
  local num_clients="$5"
  local num_participating="$6"
  local num_rounds="$7"
  local alpha="$8"
  local logfile="$9"

  read -r gpu fd < <(pick_gpu_lock)

  echo "[QUEUE] START gpu=${gpu} alg=${alg} seed=${seed} dataset=${dataset} model=${model} rounds=${num_rounds} -> ${logfile}"

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    # NOTE: nohup is not needed here because the queue script itself will be nohup'd.
    "$PYTHON" "$MAIN" \
      --seed "$seed" \
      --algorithm "$alg" \
      --dataset "$dataset" \
      --model "$model" \
      --num_clients "$num_clients" \
      --num_participating_clients "$num_participating" \
      --num_rounds "$num_rounds" \
      --alpha "$alpha" \
      > "$logfile" 2>&1
  )

  # Release GPU lock (close FD)
  exec {fd}>&-

  echo "[QUEUE] DONE  gpu=${gpu} alg=${alg} seed=${seed} -> ${logfile}"
}

# ---------------- EXPERIMENT GRID ----------------
# Edit these lists:
DATASET="shakespeare"
MODEL="LSTM"
NUM_CLIENTS=100
NUM_PARTICIPATING=20
NUM_ROUNDS=500
ALPHA=0.3

# Algorithms to run (quote ones with parentheses)
ALGS=("fedprox(exp)" "fedavg" "fedsls" "fedexp" "fedadam") #add "fedexpsls" if you want to run that too for other dataset
SEEDS=(1 2 3 4)

# ---------------- LAUNCH ALL JOBS ----------------
pids=()

for alg in "${ALGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    # log naming: match your style
    # Example you gave: logs/fedexprox_shakespeare_seed4.log
    # We'll follow: logs/<alg>_<dataset>_seed<seed>.log (and you can customize below)
    safe_alg="${alg//[^A-Za-z0-9._-]/}"  # strip weird chars for filenames
    logfile="${LOGDIR}/${safe_alg}_${DATASET}_seed${seed}.log"

    run_one "$alg" "$seed" "$DATASET" "$MODEL" "$NUM_CLIENTS" "$NUM_PARTICIPATING" "$NUM_ROUNDS" "$ALPHA" "$logfile" &
    pids+=($!)
    sleep 1
  done
done

# Wait for all jobs
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "[QUEUE] ALL EXPERIMENTS FINISHED"