# !/bin/bash
set -e

cd gail_multitarget/
python pretrain.py

disease_list=(schizophrenia alzheimer parkinson)

for disease in ${disease_list[@]}; do
    python finetune.py --config-path=config --config-name=config_${disease}
    python finetune_generation.py --config-path=config --config-name=config_${disease}
    echo "Finished processing for ${disease}"
done
