# !/bin/bash
# pip install -e .
set -e  # exit on error

cd "$(dirname "$0")"
cd ./polygon

# First, train the ligand efficiency model
# cd utils
# python train_ligand_binding_model.py
# cd ..

# polygon train \
#   --train_data ../data/guacamol_v1_train.smiles  --log_file log.txt  --save_frequency 50  \
#   --model_save ./checkpoints/model.pt  --n_epoch 200  --n_batch 1024  --debug  --d_dropout 0.2  --device cuda

disease_list=(schizophrenia alzheimer parkinson)

output_dir="../../../generated_molecules/POLYGON"
# mkdir -p ${output_dir}

# Iterate over the list (molecular activity)
# for disease in ${disease_list[@]}; do
#     (
#         polygon generate \
#           --model_path ./checkpoints/model.pt --scoring_definition ../data/${disease}.csv --max_len 100 \
#           --n_epochs 200 --mols_to_sample 8192 --optimize_batch_size 512  --optimize_n_epochs 2 --keep_top 4096 \
#           --opti gauss --outF ./checkpoints/${disease}_activity --device cuda  --n_jobs 1 --debug  # --save_payloads

#         polygon sample \
#           --model_path ./checkpoints/${disease}_activity/GDM_200.pt --scoring_definition ../data/${disease}.csv \
#           --n_molecules 10000 --output ${output_dir}/${disease}.csv --device cuda --debug --batch_size 1000
#     )  # &
# done

# wait

# Brain diseases case study
for disease in ${disease_list[@]}; do
    (
        polygon generate \
          --model_path ./checkpoints/model.pt --scoring_definition ../data/${disease}-mpo.csv --max_len 100 \
          --n_epochs 200 --mols_to_sample 8192 --optimize_batch_size 512  --optimize_n_epochs 2 --keep_top 4096 \
          --opti gauss --outF ./checkpoints/${disease}_mpo --device cuda  --n_jobs 1 --debug  # --save_payloads

        polygon sample \
          --model_path ./checkpoints/${disease}_mpo/GDM_200.pt --scoring_definition ../data/${disease}.csv \
          --n_molecules 10000 --output ${output_dir}/${disease}-mpo.csv --device cuda --debug --batch_size 1000
    )  # &
done
