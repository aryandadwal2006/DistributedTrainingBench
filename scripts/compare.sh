#!/usr/bin/env bash
set -euo pipefail

DEVICE=cuda
STEPS=50
WARMUP=10
COMMON="--device $DEVICE --steps $STEPS --warmup $WARMUP --batch-size 8 --seq-len 128 --hidden-size 256 --layers 4 --heads 4"

python -m bench.train $COMMON --kernel torch
python -m bench.train $COMMON --kernel torch --compile
python -m bench.train $COMMON --kernel triton
