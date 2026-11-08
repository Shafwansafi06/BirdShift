#!/bin/bash
# Step 1 + 2 runs. Launched 2026-11-08.
cd "$(dirname "$0")/../.."
export PYTHONPATH=src
PY=.venv/bin/python
$PY scripts/train.py --mode full --epochs 30 --imbalance-ratio 10 --balance la --run-name longtail_la \
    > results/longtail/logs/longtail_la.log 2>&1 &
LA=$!
for m in full crop birdshift; do
  $PY scripts/train.py --mode $m --epochs 30 --image-size 448 --amp --workers 8 --run-name ${m/full/baseline}_448 \
      > results/logs/${m}_448.log 2>&1 || echo "$m failed" >> results/logs/run_la_448.status
done
wait $LA
$PY scripts/evaluate_longtail.py --split val > results/logs/evaluate_longtail.log 2>&1
$PY scripts/evaluate_shift.py --split val --workers 8 \
    --full-checkpoint checkpoints/baseline_448/best.pth --crop-checkpoint checkpoints/crop_448/best.pth \
    --birdshift-checkpoint checkpoints/birdshift_448/best.pth --output results/background_shift_448 \
    > results/logs/evaluate_shift_448.log 2>&1
echo done >> results/logs/run_la_448.status
