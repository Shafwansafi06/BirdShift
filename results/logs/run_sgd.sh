#!/bin/bash
# Recipe search on the full-image model at 448 px. Launched 2026-11-08.
cd "$(dirname "$0")/../.."
export PYTHONPATH=src
COMMON="--mode full --epochs 50 --image-size 448 --amp --workers 8 --train-aug fg --scheduler cosine"
.venv/bin/python scripts/train.py $COMMON --optimizer sgd --lr 0.01 --head-lr-mult 10 \
    --run-name recipe_sgd > results/logs/recipe_sgd.log 2>&1
.venv/bin/python scripts/train.py $COMMON --optimizer sgd --lr 0.01 --head-lr-mult 10 --weights resnet50.tv_in1k \
    --run-name recipe_sgd_tv > results/logs/recipe_sgd_tv.log 2>&1
.venv/bin/python scripts/train.py $COMMON --optimizer adamw --lr 1e-4 \
    --run-name recipe_adamw > results/logs/recipe_adamw.log 2>&1
echo done > results/logs/run_sgd.status
