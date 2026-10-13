#!/usr/bin/env bash
# Reproduce steps 1-4 of the preliminary pipeline in order.
# Usage: PY=/path/to/python bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python}"

echo "== Step 0: build label table and PET file list =="
"$PY" scripts/00_build_labels.py

echo "== Step 1: manifest, QC, subject-level split =="
"$PY" scripts/01_make_manifest.py

echo "== Step 2: preprocess and cache =="
"$PY" scripts/02_preprocess.py

echo "== Unit tests =="
"$PY" -m pytest tests -q

echo "== Step 3: train resnet3d_18 seed 0 =="
"$PY" scripts/03_train.py --model resnet3d_18 --seed 0

echo "== Step 4: evaluate the newest run =="
RUN_DIR=$(ls -dt runs/resnet3d_18_0_* | head -1)
"$PY" scripts/04_evaluate.py --run_dir "$RUN_DIR"

echo "== Summary table =="
"$PY" scripts/05_summarize.py
