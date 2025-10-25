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
    t1_path = os.path.join(root_path, config.target.target1.train_path)
    t2_path = os.path.join(root_path, config.target.target2.train_path)
    model_name = 'Prior_DLGN.ckpt'

    t1_train_data = pd.read_csv(t1_path, index_col=False, header=None)
    t2_train_data = pd.read_csv(t2_path, index_col=False, header=None)

    t1_valid_path = os.path.join(root_path, config.target.target1.valid_path)
    t2_valid_path = os.path.join(root_path, config.target.target2.valid_path)
    t1_valid_data = pd.read_csv(t1_valid_path, index_col=False, header=None)
    t2_valid_data = pd.read_csv(t2_valid_path, index_col=False, header=None)

    # Apparently not used in fine-tuning (SVM activity prediction)
    # drd2 = predict_model(os.path.join(root_path, f'data/DLGN/SVM/best_drd2_svm.m'))
    # htr1a = predict_model(os.path.join(root_path, f'data/DLGN/SVM/best_htr1a_svm.m'))
    model = SmilesGAILModel(config, vocab)
    generator_path = os.path.join(original_path, task_config.model_path + f'{model_name}')
    model.generator.load_model(generator_path, device)
    model.generator.train()
    reward_func = RewardFunction(vocab, model.discriminator, device)
    task_config.use_wandb = False
    if task_config.use_wandb:
        wandb.init(project="multitarget", name='drd2_htr1a_aug_Da_Dlr1e5_Glr3e5_sf')
        wandb.log({'t1_dataset_num': len(t1_train_data), 't2_dataset_num': len(t2_train_data)})
    trainer = GAILTrainerLoop(task_config, vocab, model=model,
                              reward_func=reward_func, device=device,
                              dataset1=t1_train_data, dataset2=t2_train_data,
                              valid_dataset1=t1_valid_data, valid_dataset2=t2_valid_data)

    # trainer.train(os.path.join(output_path, task_config.save_path), t1, t2)
    scenario = config.target.name
    trainer.train(scenario, None, None)


if __name__ == "__main__":
    main()
