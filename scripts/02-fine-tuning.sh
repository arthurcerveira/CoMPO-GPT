#!bin/bash

# Make sure we are in parent directory of script directory
cd "$(dirname "$0")"/..
pwd

# We must pre-train first if there are architectural modifications
echo "python CoMPO-GPT/main.py --mode train --batch_size 256 --epoch 100 \
                       --path weights/base.h5 \
                       --d_model 1024 \
                       --finetune_dataset data/excape_all_active_compounds.smi"

python CoMPO-GPT/main.py --mode train --batch_size 256 --epoch 100 \
                       --path weights/base.h5 \
                       --d_model 1024 \
                       --finetune_dataset data/excape_all_active_compounds.smi

# For fine-tuning
echo "python3 CoMPO-GPT/main.py --batch_size 256 --mode finetune \
                --path weights/base.h5 --loadmodel \
                --path_ft weights/finetune.h5 \
                --d_model 1024 \
                --finetune_dataset data/excape_all_active_compounds.smi \
                --epoch 100"

python3 CoMPO-GPT/main.py --batch_size 256 --mode finetune \
                --path weights/base.h5 --loadmodel \
                --path_ft weights/finetune.h5 \
                --d_model 1024 \
                --finetune_dataset data/excape_all_active_compounds.smi \
                --epoch 100