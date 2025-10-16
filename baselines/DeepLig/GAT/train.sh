# !/bin/bash
set -e

targets=(AChE MAOB D2R _5HT2A D3R)

for target in "${targets[@]}"; do
    python train_gat.py $target
done