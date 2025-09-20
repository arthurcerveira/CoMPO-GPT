#!bin/bash

# Make sure we are in parent directory of script directory
cd "$(dirname "$0")"/..
pwd

# We must pre-train first if there are architectural modifications
# echo "python CoMPO-GPT/main.py --mode train --batch_size 256 --epoch 100 \
#                        --path weights/base_mpo.h5 \
#                        --d_model 1024 \
#                        --finetune_dataset data/active_compounds_mpo.smi"

# python CoMPO-GPT/main.py --mode train --batch_size 256 --epoch 100 \
#                        --path weights/base_mpo.h5 \
#                        --d_model 1024 \
#                        --finetune_dataset data/active_compounds_mpo.smi

# For fine-tuning
echo "python3 CoMPO-GPT/main.py --batch_size 256 --mode finetune \
                --path weights/base_mpo.h5 --loadmodel \
                --path_ft weights/finetune_mpo.h5 \
                --d_model 1024 \
                --finetune_dataset data/active_compounds_mpo.smi \
                --epoch 100"

python3 CoMPO-GPT/main.py --batch_size 256 --mode finetune \
                --path weights/base_mpo.h5 --loadmodel \
                --path_ft weights/finetune_mpo.h5 \
                --d_model 1024 \
                --finetune_dataset data/active_compounds_mpo.smi \
                --epoch 100