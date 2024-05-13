#!bin/bash

# Make sure we are in parent directory of script directory
cd "$(dirname "$0")"/..
pwd

# For pre-training
# python cMolGPT/main.py --mode train --batch_size 256 --epoch 100

echo "python3 cMolGPT/main.py --batch_size 256 --mode finetune \
                --path models_cMolGPT/base.h5 --loadmodel \
                --path_ft models_cMolGPT/finetune.h5 \
                --finetune_dataset data/active_compounds.smi"

python3 cMolGPT/main.py --batch_size 256 --mode finetune \
                --path models_cMolGPT/base.h5 --loadmodel \
                --path_ft models_cMolGPT/finetune.h5 \
                --finetune_dataset data/active_compounds.smi