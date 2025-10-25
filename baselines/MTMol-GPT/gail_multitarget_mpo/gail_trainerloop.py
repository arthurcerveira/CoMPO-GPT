import os
import sys

sys.path.append("..")
import pandas as pd
import numpy as np
import torch
import random
import wandb
from tqdm import tqdm, trange
import selfies as sf
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import logging
from replayBuffer import MolRLReplayBuffer
from rdkit.Chem import AllChem as Chem
from utils import canonical_smiles
from moses.metrics import FCDMetric
from pathlib import Path

logger = logging.getLogger(__name__)


class GAILTrainerLoop:
    def __init__(self, config, vocab, model, reward_func, device='cuda', dataset1=None, dataset2=None,
                 dataset3=None, dataset4=None, dataset5=None,
                 valid_dataset1=None, valid_dataset2=None, valid_dataset3=None, valid_dataset4=None,
                 valid_dataset5=None):
        self.config = config
        self.vocab = vocab
        self.model = model
        self.reward_func = reward_func
        self.n_gail = config.n_gail
        self.ppo_buffer_size = config.ppo_buffer_size
        self.ppo_mini_batch_size = config.ppo_mini_batch_size
        self.ppo_epsilon = config.ppo_epsilon
        self.ppo_iteration = config.ppo_iteration
        self.dis_nums = config.dis_nums
        self.mix_demo_ratio = config.mix_demo_ratio
        self.replay_buffer = MolRLReplayBuffer(self.ppo_buffer_size, shuffle=True)

        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.epoch_step = config.epoch_step
        self.warmup_step = config.warmup_step
        self.use_aug = config.use_augmentation
        self.use_sf = config.use_selfies

        self.num_objectives = 5
        dataset_args = [dataset1, dataset2, dataset3, dataset4, dataset5]
        if any(ds is None for ds in dataset_args):
            raise ValueError("All five objective datasets must be provided.")
        self.org_datasets = [ds.copy() if isinstance(ds, pd.DataFrame) else pd.DataFrame(ds) for ds in dataset_args]
        self.datasets = [df.copy() for df in self.org_datasets]
        self.dataset_aug = [pd.DataFrame() for _ in range(self.num_objectives)]
        self.valid_datasets = [valid_dataset1, valid_dataset2, valid_dataset3, valid_dataset4, valid_dataset5]
        self.dataset_train_loaders = [None] * self.num_objectives

        self.frac = 5
        self.device = device
        self.tmp_vars = {}

    def train(self, scenario, predictor1, predictor2):
        # Training begins
        # self.output_path = output_path
        # if not os.path.exists(output_path):
        #     os.makedirs(output_path)
        self.output_path = Path(__file__).parent / "checkpoints" / scenario
        self.output_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.epochs + 1):
            logger.info(f'train epoch {epoch}')

            with torch.no_grad():
                self.model.generator.eval()

                if self.config.use_selfies:
                    smiles = self.model.generator.generate(100)
                    sequence = []
                    for sequence_i in smiles:
                        try:
                            sequence.append(sf.decoder(sequence_i))
                        except sf.DecoderError:
                            pass  # sf.encoder error!
                else:
                    sequence = self.model.generator.generate(100)
            # new_smiles, valid_vec, valid_smiles = canonical_smiles(sequence)
            # drd2_pre = predictor1(sequence)
            # drd2_seq = [sequence[i] for i in list(np.array(np.where(drd2_pre >= 0.5)[0]))]
            # htr1a_pre = predictor2(sequence)
            # htr1a_seq = [sequence[i] for i in list(np.array(np.where(htr1a_pre >= 0.5)[0]))]
            if self.use_aug:
                logger.warning("Data augmentation is not enabled for the MPO multi-objective run; skipping.")

            self.dataset_train_loaders = [
                DataLoader(list(dataset[0]), shuffle=True, pin_memory=True,
                           batch_size=self.batch_size, drop_last=True)
                for dataset in self.datasets
            ]
            # mean_drd2 = np.mean(drd2_pre)
            # mean_htr1a = np.mean(htr1a_pre)
            # num_drd2 = np.sum(drd2_pre >= 0.5)
            # num_htr1a = np.sum(htr1a_pre >= 0.5)
            # both_sum = np.sum((htr1a_pre >= 0.5) & (drd2_pre >= 0.5))
            # unique = len(np.unique(sequence))
            # if self.config.use_wandb:
            #     # log the loss
            #     wandb.log(
            #         {'test_vaild': len(valid_vec), 'test_drd2_mean_pre': mean_drd2, 'test_htr1a_mean_pre': mean_htr1a,
            #          'test_drd2_num': num_drd2, 'drd2_dataset_num': len(self.dataset1),
            #          'htr1a_dataset_num': len(self.dataset2),
            #          'test_htr1a_num': num_htr1a, 'both_target_num': both_sum, 'test_unique': unique})
            
            if epoch % 200 == 0:
                self.model.generator.save_model(
                    os.path.join(self.output_path, f'fine_tuning_generator_{epoch}.pt'))
                self.model.discriminator.save_model(
                    os.path.join(self.output_path, f'fine_tuning_discriminator_{epoch}.pt'))
            self.train_epoch(epoch)
            self.replay_buffer.clear()
            torch.cuda.empty_cache()

    def train_epoch(self, epoch):
        iter_train_dataloaders = [iter(loader) for loader in self.dataset_train_loaders]
        for step in range(999):
            '''Buliding buffer'''
            buffer_count = 0
            while buffer_count < self.ppo_buffer_size:
                try:
                    batches = [next(iterator) for iterator in iter_train_dataloaders]
                except StopIteration:
                    return
                self.collect_samples(batches)
                buffer_count += len(batches[0])
            '''Rewardnet Training'''
            gail_batch_losses = []
            for i in range(self.n_gail):
                self.replay_buffer.restart()
                for mini_batch in self.replay_buffer.iterate_sample(self.ppo_mini_batch_size):
                    if not isinstance(mini_batch, tuple):
                        mini_batch = (mini_batch,)
                    state_tuples, actions, action_log_probs, rewards = zip(*mini_batch)
                    demo_smiles_list = [list(items) for items in zip(*state_tuples)]
                    gail_loss = self.reward_func.fit(demo_smiles_list, actions)
                    gail_batch_losses.append(gail_loss)
                    if self.config.use_wandb:
                        # log the loss
                        wandb.log({'gail_loss': gail_loss})
                    logger.info(
                        f"rewardnet --mini-batch gail_loss:{gail_loss}")
                    torch.cuda.empty_cache()

            '''Generator Training'''
            for i in range(self.ppo_iteration):
                torch.cuda.empty_cache()
                mini_batch_gen_loss = []
                self.replay_buffer.restart()
                for mini_batch in self.replay_buffer.iterate_sample(self.ppo_mini_batch_size):
                    ppo_batch = {}
                    if not isinstance(mini_batch, tuple):
                        mini_batch = (mini_batch,)
                    state_tuples, actions, action_log_probs, rewards = zip(*mini_batch)
                    ppo_batch['generated_seq'] = {}
                    batch_idx = [torch.tensor(self.vocab.encode(string, add_bos=True, add_eos=True), dtype=torch.long,
                                              device=self.device) for string in actions]
                    ppo_batch['generated_seq']['index'] = pad_sequence([t[:-1] for t in batch_idx], batch_first=True,
                                                                       padding_value=self.vocab.pad)
                    ppo_batch['generated_seq']['label'] = pad_sequence([t[1:] for t in batch_idx], batch_first=True,
                                                                       padding_value=self.vocab.pad)
                    ppo_batch['generated_seq']['lens'] = torch.tensor([len(t) - 1 for t in batch_idx],
                                                                      dtype=torch.long, device=self.device)
                    ppo_batch["rewards"] = torch.FloatTensor(rewards).to(self.device)
                    ppo_batch["old_log_probs"] = torch.FloatTensor(action_log_probs).to(self.device)

                    gen_dict = self.train_generator_step(ppo_batch)
                    mini_batch_gen_loss.append(gen_dict["loss"])

                    if self.config.use_wandb:
                        wandb.log({'generator_loss': gen_dict["loss"]})
                    logger.info(
                        f'Generator --Epoch:{epoch}---loacl_step:{step}---loss:{gen_dict["loss"]}')

                generator_mean_loss = np.mean(mini_batch_gen_loss)
                if self.config.use_wandb:
                    wandb.log({'mini_batch_generator_mean_loss': generator_mean_loss,
                               'gail_batch_losses': np.mean(gail_batch_losses),
                               'lr': self.model.generator.optimizer.defaults['lr']})
                logger.info(
                    f"generator --mini-batch loss:{generator_mean_loss}")
            # self.mix_demo_ratio -= 0.001
            self.replay_buffer.clear()

    def train_generator_step(self, batch):
        self.model.generator.train()
        results = self.model.generator.compute_log_probs(batch['generated_seq'])
        # old log probilities
        log_probs = torch.nan_to_num(results["log_probs"], nan=0.0, neginf=-1e9, posinf=1e9)
        old_log_probs = torch.nan_to_num(batch["old_log_probs"], nan=0.0, neginf=-1e9, posinf=1e9)
        # # advantage
        advantages = torch.nan_to_num(batch["rewards"], nan=0.0, neginf=0.0, posinf=0.0)
        valid_mask = torch.isfinite(log_probs) & torch.isfinite(old_log_probs) & torch.isfinite(advantages)
        if not valid_mask.any():
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "clip_frac": 0.0,
                "approx_kl": 0.0,
                "ratio": 0.0,
                "advantages": 0.0,
            }
        log_probs = log_probs[valid_mask]
        old_log_probs = old_log_probs[valid_mask]
        advantages = advantages[valid_mask]
        # Policy Loss
        # shape: (batch)
        ratio_delta = (log_probs - old_log_probs).clamp(min=-20.0, max=20.0)
        ratio = ratio_delta.exp()
        # ratio = log_probs
        ## shape: (batch)
        policy_loss1 = -advantages * ratio
        ## shape: (batch)
        policy_loss2 = -advantages * ratio.clamp(1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon)
        ## shape: (batch)
        policy_loss = torch.max(policy_loss1, policy_loss2).mean()
        # loss = policy_loss
        loss = policy_loss

        # loss = -torch.mean(log_probs * advantages)
        # Backward Loss
        self.model.generator.optimizer.zero_grad()
        loss.backward()
        self.model.generator.optimizer.step()
        # self.model.generator.scheduler.step()
        with torch.no_grad():
            clip_frac = ((ratio - 1.0).abs() > self.ppo_epsilon).float().mean()
            approx_kl = (log_probs - old_log_probs).pow(2).mean()

        log_dict = {}
        log_dict["loss"] = loss.item()
        log_dict["policy_loss"] = policy_loss.item()
        log_dict["clip_frac"] = clip_frac.item()
        log_dict["approx_kl"] = approx_kl.item()
        log_dict["ratio"] = ratio.mean().item()
        log_dict["advantages"] = advantages.mean().item()
        if self.config.use_wandb:
            wandb.log(log_dict)
        return log_dict

    @torch.no_grad()
    def collect_samples(self, batch_targets):
        batch_targets = [list(batch) for batch in batch_targets]
        batch_size = len(batch_targets[0])
        num_demos = int(batch_size * self.mix_demo_ratio)

        if num_demos > 0:
            demo_batches = [batch[:num_demos] for batch in batch_targets]
            for obj_idx in range(self.num_objectives):
                log_probs = self._compute_log_probs(demo_batches[obj_idx])
                if not log_probs:
                    continue
                rewards = (np.ones(len(log_probs)) * 2.0).tolist()
                demo_states = [states[:len(log_probs)] for states in demo_batches]
                self.replay_buffer.update_batch(
                    states_list=demo_states,
                    actions=demo_batches[obj_idx][:len(log_probs)],
                    action_log_probs=log_probs,
                    rewards=rewards,
                )
        else:
            num_demos = 0

        self.model.discriminator.eval()
        actual_sample_size = len(batch_targets[0]) - num_demos
        if actual_sample_size <= 0:
            return

        select_batch_targets = [batch[num_demos:num_demos + actual_sample_size] for batch in batch_targets]

        sequences = []
        seqs_log_p_sum = torch.empty(0, 1, device=self.device)
        while len(sequences) < actual_sample_size:
            torch.cuda.empty_cache()
            sequence, seq_log_p, seq_log_p_sum = self.model.generator.generate_prob(actual_sample_size * 2)
            seq_log_p_sum = seq_log_p_sum.view(-1, 1)
            index = -1
            while len(sequences) < actual_sample_size:
                index += 1
                if index >= len(sequence):
                    break
                sm = sequence[index]
                try:
                    sm2 = sf.decoder(sm)
                    mol = Chem.MolFromSmiles(sm2)
                except Exception:
                    continue
                if sm in sequences:
                    continue
                sequences.append(sm)
                seqs_log_p_sum = torch.cat([seqs_log_p_sum, seq_log_p_sum[index].view(1, 1)], dim=0)
        seqs_log_p_sum = torch.nan_to_num(seqs_log_p_sum, nan=0.0, neginf=-1e9, posinf=1e9)
        finite_mask = torch.isfinite(seqs_log_p_sum.view(-1))
        if finite_mask.sum().item() == 0:
            return
        mask_list = finite_mask.cpu().tolist()
        sequences = [seq for seq, keep in zip(sequences, mask_list) if keep]
        seqs_log_p_sum = seqs_log_p_sum[finite_mask]
        select_batch_targets = [
            [seq for seq, keep in zip(targets, mask_list) if keep]
            for targets in select_batch_targets
        ]
        rewards = self.reward_func.get_reward(select_batch_targets, sequences)
        rewards = torch.nan_to_num(rewards, nan=0.0, neginf=0.0, posinf=0.0).tolist()
        log_prob_list = seqs_log_p_sum.view(-1).tolist()
        self.replay_buffer.update_batch(
            states_list=select_batch_targets,
            actions=sequences,
            action_log_probs=log_prob_list,
            rewards=rewards,
        )

    def _compute_log_probs(self, sequences):
        if not sequences:
            return []
        batch_idx = [
            torch.tensor(self.vocab.encode(string, add_bos=True, add_eos=True), dtype=torch.long, device=self.device)
            for string in sequences
        ]
        batch = {
            'index': pad_sequence([t[:-1] for t in batch_idx], batch_first=True, padding_value=self.vocab.pad),
            'label': pad_sequence([t[1:] for t in batch_idx], batch_first=True, padding_value=self.vocab.pad),
            'lens': torch.tensor([len(t) - 1 for t in batch_idx], dtype=torch.long, device=self.device),
            'prop': [],
        }
        log_probs = self.model.generator.compute_log_probs(batch)["log_probs"]
        log_probs = torch.nan_to_num(log_probs, nan=0.0, neginf=-1e9, posinf=1e9)
        return log_probs.tolist()
