#!bin/bash

# Make sure we are in parent directory of script directory
cd "$(dirname "$0")"/..
pwd

# list of target names
targets=("Unconditional" "AChE" "D2R" "D3R" "_5HT2A" "MAOB")

# number of epochs comes from command line argument
epochs=$1

# check if epochs is provided
if [ -z "$epochs" ]
then
    echo "Please provide number of epochs as argument"
    exit 1
fi

# create directory to store generated molecules
mkdir -p generated_molecules/${epochs}-epoch

# for each target, generate molecules based on index
for i in {0..5}
do
    model_path="weights/finetune.h5_${epochs}"

    # print target name
    echo "Generating molecules for target: ${targets[i]}"
    #print command
    echo "python3 CoMPO-GPT/main.py --mode infer --target ${i} --path ${model_path} \
                          --num_molecules 10000 --output_path generated_molecules/${epochs}-epoch/${targets[i]}.csv"

    python3 CoMPO-GPT/main.py --mode infer --target ${i} --path ${model_path} \
                    --num_molecules 10000 --output_path generated_molecules/${epochs}-epoch/${targets[i]}.csv  &
done

# wait for all background processes to finish
wait
echo "Done generating molecules with ${epochs} epochs model"