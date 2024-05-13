## Setup

**Create `aidd` conda environment**

1. `conda env create -f environment_a6000.yml`
2. `pip install lightgbm ipykernel ipdb`

**Create `chemprop` conda environment**

1. `cd chemprop`
2. `conda env create -f environment.yml`
3. `conda activate chemprop`
4. `pip install -e .`


## Instructions

**`conda activate aidd`**

1. Generate datasets: `scripts/01-Split-Fine-Tuning-QSAR-Datasets.ipynb`
2. Fine-tune the model: `scripts/02-fine-tuning.sh`
3. Single-target molecule generation: `scripts/03-single-target-generation.sh`
4. Multi-target molecule generation: `scripts/04-multi_target_generation.py`

**`conda activate chemprop`**

5. Train chemprop models: 
- `scripts/05-train_chemprop.py`
- `scripts/06-train_chemprop_pIC50.py`
6. Predict chemprop models: `scripts/07-predict_activity_chemprop.py`

# cMolGPT 

Implementation of ["cMolGPT: A Conditional Generative Pre-Trained Transformer for Target-Specific De Novo Molecular Generation"](https://pubmed.ncbi.nlm.nih.gov/37298906/).
Enforcing target embeddings as queries and keys.

Please feel free to open an issue or email wenlu.wang.1@gmail.com and ye.wang@biogen.com if you have any questions. We will respond as soon as we can.

## Dependencies

environment_v100.yml tested on NVIDIA V100

environment_a6000.yml tested on RTX A6000

[Create env from yml file](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

[Arthur] $ conda env create -f environment_a6000.yml
[Arthur] $ pip install lightgbm ipykernel ipdb

## Data

### [Mol_target_dataloader](https://github.com/alfredyewang/Mol_target_dataloader)
Please download this repo and put the folder in the root directory.
If you would like to finetune with your own target data, please replace 'target.smi'.

## How to run

*unzip train.sim.zip

### Train
```
  python3 main.py --batch_size 512 --mode train \
                  --path model_base.h5 
```
### Fine-tune
```
  python3 main.py --batch_size 512 --mode finetune \
                  --path model_base.h5 --loadmodel
```
*In the case of fine-tuning, the base model will be overwritten in place.

*You can change the number of targets in [model_auto.py](https://github.com/VV123/cMolGPT/blob/f0eba15dbf53b47a35afc305674c997354472590/model_auto.py#L58C66-L58C107).

### Infer/Generate
```
  python3 main.py --mode infer --target [0/1/2/3] --path model_finetune.h5
```

No target
```
  python3 main.py --mode infer --target 0 --path model_finetune.h5
```

Target 2
```
  python3 main.py --mode infer --target 2 --path model_finetune.h5
```

