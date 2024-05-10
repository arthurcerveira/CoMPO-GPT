#!bin/bash

# Make sure we are in parent directory of script directory
cd "$(dirname "$0")"/..
pwd

python3 main.py --batch_size 512 --mode finetune \
                --path models_cMolGPT/base.h5 --loadmodel \
                --path_ft models_cMolGPT/finetune.h5 \
                --finetune_dataset data/active_compounds.smi