import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import pickle
import hydra
import torch
import logging
import numpy as np
# import moses
import selfies as sf

from omegaconf import DictConfig, OmegaConf
from utils import set_random_seed, tokens_struct, selfies_tokens_struct
from model import Generator
from tqdm import tqdm
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
log = logging.getLogger(__name__)

current_dir = Path(__file__).resolve().parent
generated_mols_dir = current_dir / ".." / ".." / ".." / "generated_molecules" / "MTMol-GPT"
generated_mols_dir.mkdir(parents=True, exist_ok=True)

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.create(OmegaConf.to_yaml(cfg))
    output_path = os.getcwd()
    original_path = hydra.utils.get_original_cwd()
    root_path = '/'.join(original_path.split('/')[:-1])
    model_comfig = config.model.transformers
    task_config = config.generation.gail
    # set cuda
    if torch.cuda.is_available():
        dvc_id = 0
        device_set = f'cuda:{dvc_id}'
    else:
        device_set = 'cpu'

    set_random_seed(task_config.random_seed)
    config.train.gail.use_selfies = False
    if config.train.gail.use_selfies:
        vocab = selfies_tokens_struct()
    else:
        vocab = tokens_struct()

    task_config.save_path = generated_mols_dir
    save_path_smiles = task_config.save_path / f"{config.target.name}.csv"
    
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)

    iteration = task_config.generate_num // task_config.batch_size
    model_root = current_dir / "checkpoints" / config.target.name

    # for e in range(0, 1001, 500):
    # for e in (0, 500, 995):
    e = 1000  # Hardcoded to last epoch
    generator = Generator(model_comfig, vocab, device=device_set)

    model_path = model_root / f'fine_tuning_generator_{e}.pt'
    generator.load_model(model_path, device_set)
    generator.model.eval()
    loop = tqdm(range(iteration + 1), desc=f"Generating with fine_tuning_generator_{e}.pt")
    gen_samples = []

    with torch.no_grad():
        for i in loop:
            sequences = generator.generate(task_config.batch_size, task_config.max_length)
            if config.train.gail.use_selfies:
                sm = []
                for sequence in sequences:
                    sm.append(sf.decoder(sequence))
                gen_samples += sm
            else:
                gen_samples += sequences
    
    np.savetxt(save_path_smiles, gen_samples, fmt='%s')


if __name__ == "__main__":
    main()
