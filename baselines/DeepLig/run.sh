# !/bin/bash
set -e

cd GAT
bash train.sh
cd ..

python pretrain.py

disease_list=(schizophrenia alzheimer parkinson schizophrenia_mpo alzheimer_mpo parkinson_mpo)

for disease in ${disease_list[@]}; do
    python rl_pipeline.py $disease &
done

# Wait for all background processes to finish
wait

echo "Finished processing for all diseases"