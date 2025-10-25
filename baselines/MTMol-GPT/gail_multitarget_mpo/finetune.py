import sys

sys.path.append("..")
import os
import hydra
import torch
import logging
import wandb
import pickle
import pandas as pd
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from utils import set_random_seed, predict_model, tokens_struct, selfies_tokens_struct
from model import SmilesGAILModel
from rewardfunc import RewardFunction
from gail_trainerloop import GAILTrainerLoop

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.create(OmegaConf.to_yaml(cfg))
    output_path = os.getcwd()
    original_path = hydra.utils.get_original_cwd()
    root_path = '/'.join(original_path.split('/')[:-1])
    task_config = config.train.gail
    # set cuda
    if torch.cuda.is_available():
        dvc_id = 0
        device = f'cuda:{dvc_id}'
    else:
        device = 'cpu'
    set_random_seed(task_config.random_seed)
    config['device'] = device

    task_config.use_selfies = False

    vocab = tokens_struct()
    
    # Load all target datasets
    t1_path = os.path.join(root_path, config.target.target1.train_path)
    t2_path = os.path.join(root_path, config.target.target2.train_path)
    t3_path = os.path.join(root_path, config.target.target3.train_path)
    t4_path = os.path.join(root_path, config.target.target4.train_path)
    t5_path = os.path.join(root_path, config.target.target5.train_path)
    model_name = 'Prior_DLGN.ckpt'

    # Load and filter sequences by length (max 140 tokens)
    def filter_by_length(df, max_length=140):
        """Filter sequences that are too long for the pre-trained model"""
        filtered_data = []
        for idx, row in df.iterrows():
            sequence = str(row[0])  # Assuming SMILES is in first column
            if len(sequence) <= max_length:
                filtered_data.append(row)
        return pd.DataFrame(filtered_data) if filtered_data else pd.DataFrame()

    t1_train_data = filter_by_length(pd.read_csv(t1_path, index_col=False, header=None))
    t2_train_data = filter_by_length(pd.read_csv(t2_path, index_col=False, header=None))
    t3_train_data = filter_by_length(pd.read_csv(t3_path, index_col=False, header=None))
    t4_train_data = filter_by_length(pd.read_csv(t4_path, index_col=False, header=None))
    t5_train_data = filter_by_length(pd.read_csv(t5_path, index_col=False, header=None))

    t1_valid_path = os.path.join(root_path, config.target.target1.valid_path)
    t2_valid_path = os.path.join(root_path, config.target.target2.valid_path)
    t3_valid_path = os.path.join(root_path, config.target.target3.valid_path)
    t4_valid_path = os.path.join(root_path, config.target.target4.valid_path)
    t5_valid_path = os.path.join(root_path, config.target.target5.valid_path)
    
    t1_valid_data = filter_by_length(pd.read_csv(t1_valid_path, index_col=False, header=None))
    t2_valid_data = filter_by_length(pd.read_csv(t2_valid_path, index_col=False, header=None))
    t3_valid_data = filter_by_length(pd.read_csv(t3_valid_path, index_col=False, header=None))
    t4_valid_data = filter_by_length(pd.read_csv(t4_valid_path, index_col=False, header=None))
    t5_valid_data = filter_by_length(pd.read_csv(t5_valid_path, index_col=False, header=None))

    print(f"Loaded datasets:")
    print(f"  Target1: {len(t1_train_data)} train, {len(t1_valid_data)} valid")
    print(f"  Target2: {len(t2_train_data)} train, {len(t2_valid_data)} valid")
    print(f"  Target3: {len(t3_train_data)} train, {len(t3_valid_data)} valid")
    print(f"  Target4: {len(t4_train_data)} train, {len(t4_valid_data)} valid")
    print(f"  Target5: {len(t5_train_data)} train, {len(t5_valid_data)} valid")

    # Apparently not used in fine-tuning (SVM activity prediction)
    # drd2 = predict_model(os.path.join(root_path, f'data/DLGN/SVM/best_drd2_svm.m'))
    # htr1a = predict_model(os.path.join(root_path, f'data/DLGN/SVM/best_htr1a_svm.m'))
    num_objectives = 5
    model = SmilesGAILModel(config, vocab, num_objectives=num_objectives)
    generator_path = os.path.join(original_path, task_config.model_path + f'{model_name}')
    model.generator.load_model(generator_path, device)
    model.generator.train()
    reward_func = RewardFunction(vocab, model.discriminator, device, num_objectives=num_objectives)
    task_config.use_wandb = False
    if task_config.use_wandb:
        wandb.init(project="multitarget", name='drd2_htr1a_aug_Da_Dlr1e5_Glr3e5_sf')
        wandb.log({'t1_dataset_num': len(t1_train_data), 't2_dataset_num': len(t2_train_data)})
    trainer = GAILTrainerLoop(task_config, vocab, model=model,
                              reward_func=reward_func, device=device,
                              dataset1=t1_train_data, dataset2=t2_train_data,
                              dataset3=t3_train_data, dataset4=t4_train_data, dataset5=t5_train_data,
                              valid_dataset1=t1_valid_data, valid_dataset2=t2_valid_data,
                              valid_dataset3=t3_valid_data, valid_dataset4=t4_valid_data, valid_dataset5=t5_valid_data)

    # trainer.train(os.path.join(output_path, task_config.save_path), t1, t2)
    scenario = config.target.name
    trainer.train(scenario, None, None)


if __name__ == "__main__":
    main()
