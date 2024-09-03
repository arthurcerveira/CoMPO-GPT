#!/bin/sh

for epochs in 20 50 70 100;
do
    # echo "Running pipeline for $epochs epochs"
    # eval "$(conda shell.bash hook)"
    # conda activate aidd

    # bash scripts/03-single-target-generation.sh $epochs
    # python scripts/04-multi_target_generation.py $epochs

    eval "$(conda shell.bash hook)"
    conda activate chemprop

    # python scripts/07-predict_activity_chemprop.py $epochs
    python scripts/08-predict_activity_multitarget.py $epochs
done
