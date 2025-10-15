# !/bin/bash
set -e

for target in "${targets[@]}"; do
    python train_gat.py $target
done