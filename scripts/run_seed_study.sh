#!/usr/bin/env bash
# 10-seed study driver: fills in missing seeds (see job lists below) for every
# architecture at its final training protocol, evaluating each run when done.
# Three lanes share the GPU: a pool of small jobs (4 concurrent), one vit lane
# and one swin lane (sequential each). Safe to interrupt; finished runs keep.
set -u
PY=${PY:-python}
cd "$(dirname "$0")/.."
LOG=data/seed_study_logs
mkdir -p "$LOG"

train_eval() {  # model seed extra-args...
  local model=$1 seed=$2; shift 2
  local log="$LOG/${model}_s${seed}.log"
  if grep -q "^EVAL_DONE" "$log" 2>/dev/null; then
    echo "skip ${model} s${seed} (already done)"; return 0
  fi
  echo "[$(date +%H:%M:%S)] start ${model} seed ${seed}"
  if ! $PY scripts/03_train.py --model "$model" --seed "$seed" "$@" > "$log" 2>&1; then
    echo "TRAIN FAILED: ${model} s${seed} (see $log)"; return 1
  fi
  local run_dir
  run_dir=$(grep -m1 "^run dir: " "$log" | sed 's/^run dir: //; s/ (lr.*//')
  if $PY scripts/04_evaluate.py --run_dir "$run_dir" >> "$log" 2>&1; then
    echo "EVAL_DONE $run_dir" >> "$log"
    echo "[$(date +%H:%M:%S)] done ${model} s${seed}: $(grep 'test CL_WC' "$log" | tail -1)"
  else
    echo "EVAL FAILED: ${model} s${seed}"
  fi
}
export -f train_eval
export PY LOG

small_jobs() {
  for s in 1 2 3 4 5 6 7 8 9; do
    echo "resnet3d_18 $s --set train.epochs=50"
    echo "resnet3d_34 $s --set train.epochs=50"
  done
  for s in 3 4 5 6 7 8 9; do
    echo "resnet_transformer_hybrid $s --set train.num_workers=4"
    echo "cnn_transformer_hybrid $s --set train.num_workers=4"
    echo "cnn_conv_control $s --set train.num_workers=4"
  done
}

# lane 1: small-job pool, 4 concurrent
small_jobs | xargs -P 4 -I{} bash -c 'train_eval {}' &
POOL=$!

# lane 2: vit sequential
(for s in 1 2 3 4 5 6 7 8 9; do
   train_eval vit3d_small "$s" --set train.epochs=150 --set train.num_workers=4
 done) &
VIT=$!

# lane 3: swin sequential
(for s in 1 2 3 4 5 6 7 8 9; do
   train_eval swinunetr_encoder "$s" --set train.epochs=150 --set train.num_workers=6
 done) &
SWIN=$!

wait $POOL $VIT $SWIN
echo "SEED STUDY COMPLETE"
