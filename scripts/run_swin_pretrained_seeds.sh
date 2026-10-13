#!/usr/bin/env bash
# Phase-B extension: after the main seed study finishes, train + evaluate
# swinunetr_encoder WITH the CT SSL pretrained weights for seeds 1..9
# (seed 0 already exists), then regenerate the seed-study summary.
# Resumable: completed seeds are skipped via EVAL_DONE markers.
set -u
PY=${PY:-python}
cd "$(dirname "$0")/.."
LOG=data/seed_study_logs
mkdir -p "$LOG"
PRE=data/pretrained/model_swinvit.pt

# wait until the main study's trainings are gone (no 03_train.py processes)
while pgrep -f "03_train.py" > /dev/null; do sleep 300; done
echo "[$(date)] main seed study finished; starting swin pretrained seeds"

for s in 1 2 3 4 5 6 7 8 9; do
  log="$LOG/swin_pretrained_s${s}.log"
  if grep -q "^EVAL_DONE" "$log" 2>/dev/null; then
    echo "skip swin-pretrained s${s}"; continue
  fi
  echo "[$(date +%H:%M:%S)] start swin-pretrained seed ${s}"
  if ! $PY scripts/03_train.py --model swinunetr_encoder --seed "$s" \
        --pretrained_path "$PRE" --set train.epochs=150 \
        --set train.num_workers=6 > "$log" 2>&1; then
    echo "TRAIN FAILED: swin-pretrained s${s}"; continue
  fi
  run_dir=$(grep -m1 "^run dir: " "$log" | sed 's/^run dir: //; s/ (lr.*//')
  if $PY scripts/04_evaluate.py --run_dir "$run_dir" >> "$log" 2>&1; then
    echo "EVAL_DONE $run_dir" >> "$log"
    echo "[$(date +%H:%M:%S)] done s${s}: $(grep 'test CL_WC' "$log" | tail -1)"
  fi
done

$PY scripts/16_seed_study_summary.py > /dev/null && echo "summary regenerated"
echo "SWIN PRETRAINED SEEDS COMPLETE"
