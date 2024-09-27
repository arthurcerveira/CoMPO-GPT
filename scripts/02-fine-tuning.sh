#!bin/bash

# Make sure we are in parent directory of script directory
cd "$(dirname "$0")"/..
pwd

# We must pre-train first if there are architectural modifications
echo "python cMolGPT/main.py --mode train --batch_size 256 --epoch 100 \
                       --path models_cMolGPT/base.h5 \
                       --finetune_dataset data/excape_all_active_compounds.smi"

python cMolGPT/main.py --mode train --batch_size 256 --epoch 100 \
                       --path models_cMolGPT/base.h5 \
                       --finetune_dataset data/excape_all_active_compounds.smi

# For fine-tuning
echo "python3 cMolGPT/main.py --batch_size 256 --mode finetune \
                --path models_cMolGPT/base.h5 --loadmodel \
                --path_ft models_cMolGPT/finetune.h5 \
                --finetune_dataset data/excape_all_active_compounds.smi \
                --epoch 100"

python3 cMolGPT/main.py --batch_size 256 --mode finetune \
                --path models_cMolGPT/base.h5 --loadmodel \
                --path_ft models_cMolGPT/finetune.h5 \
                --finetune_dataset data/excape_all_active_compounds.smi \
                --epoch 100