import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim import SGD, Adam
from torch.nn import MSELoss, L1Loss
from torch.nn.init import xavier_uniform_
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import sys
from model_auto import ConditionalTransformer, PositionalEncoding, generate_square_subsequent_mask, create_mask
from utils import top_k_top_p_filtering, open_file, read_csv_file, load_sets
import vocabulary as mv
import dataset as md
import torch.utils.data as tud
from utils import read_delimited_file
import os.path
import glob
import math
import torch
import torch.nn as nn
from collections import Counter
from torch import Tensor
import io
import time
from tqdm import tqdm
import pandas as pd

torch.manual_seed(0)

def topk_filter(scores: torch.FloatTensor, top_k: int = 0, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
    top_k = min(max(top_k, min_tokens_to_keep), scores.size(-1))  # Safety check
     # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
    scores = scores.masked_fill(indices_to_remove, filter_value)
    return scores


def evaluate(model, valid_iter):
    model.eval()
    losses = 0
    for idx, _tgt in (enumerate(valid_iter)):
        _target = None
        if type(_tgt) is tuple:
            _tgt, _target = _tgt
            _target = torch.LongTensor(_target).to(device)
        tgt = _tgt.transpose(0, 1).to(device)
        tgt_input = tgt[:-1, :]

        tgt_mask, tgt_padding_mask = create_mask(tgt_input)

        if _target is None:
            target = torch.zeros((tgt_input.size()[-1]), dtype=torch.int32).to(device)
        else:
            target = _target
        #target = torch.zeros((tgt_input.size()[-1]), dtype=torch.int32).to(device)
        logits = model(tgt_input, tgt_mask, tgt_padding_mask, target)
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(valid_iter)


def train_epoch(model, train_iter, optimizer):
    model.train()
    losses = 0
    for idx, _tgt in enumerate(train_iter):
        _target = None
        if type(_tgt) is tuple:
            _tgt, _target = _tgt
            _target = torch.LongTensor(_target).to(device)
        #print(type(_tgt) is tuple)
        tgt = _tgt.transpose(0, 1).to(device)
        # remove encoder
        tgt_input = tgt[:-1, :]

        tgt_mask, tgt_padding_mask = create_mask(tgt_input)
        if _target is None:
            target = torch.zeros((tgt_input.size()[-1]), dtype=torch.int32).to(device)
        else:
            target = _target
        
        logits = model(tgt_input, tgt_mask, tgt_padding_mask, target)
      
        optimizer.zero_grad()
      
        tgt_out = tgt[1:,:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        loss.backward()

        optimizer.step()
        if idx % 100 == 0:
            print('Train Epoch: {}\t Loss: {:.6f}'.format(epoch, loss.item()))     
        losses += loss.item()

    print('====> Epoch: {0} total loss: {1:.4f}.'.format(epoch, losses))
    return losses / len(train_iter)

def greedy_decode(model, max_len, start_symbol, target, exclude_target=None):
    #memory = torch.zeros(40, 512, 512).to('cuda')
    #memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        #s, b = ys.size()
        # batch_size = 1
        b = 1
        s = max_len
        FFD = 512
        if target == 0:
            _target = torch.zeros((b), dtype=torch.int32).to(device)
        else:
            _target = (torch.ones((b), dtype=torch.int32)*target).to(device)

        # breakpoint()
        #memory = torch.zeros(s, b, FFD).to('cuda')
        #memory = memory.to(device)
        #memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                                    .type(torch.bool)).to(device)
        if exclude_target is None:
            out = model.decode(ys, tgt_mask, _target)
        else:
            _exclude_target = (torch.ones((b), dtype=torch.int32)*exclude_target).to(device)
            out = model.decode_exclude(ys, tgt_mask, _target, _exclude_target)
        
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1]) #[b, vocab_size]
        pred_proba_t = topk_filter(prob, top_k=30) #[b, vocab_size]
        probs = pred_proba_t.softmax(dim=1) #[b, vocab_size]
        next_word = torch.multinomial(probs, 1)
        #_, next_word = torch.max(prob, dim = 1)
        next_word = next_word.item()
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(ys.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
          break
    return ys


def greedy_decode_multitarget(model, max_len, start_symbol, targets, aggregate_function):
    #memory = torch.zeros(40, 512, 512).to('cuda')
    #memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        #s, b = ys.size()
        # batch_size = 1
        b = 1
        s = max_len
        FFD = 512

        _targets = torch.tensor(targets, dtype=torch.int32).to(device)

        # breakpoint()

        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                                    .type(torch.bool)).to(device)
        out = model.decode_multitarget(ys, tgt_mask, _targets, aggregate_function)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1]) #[b, vocab_size]
        pred_proba_t = topk_filter(prob, top_k=30) #[b, vocab_size]
        probs = pred_proba_t.softmax(dim=1) #[b, vocab_size]
        next_word = torch.multinomial(probs, 1)
        #_, next_word = torch.max(prob, dim = 1)
        next_word = next_word.item()
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(ys.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
          break

    return ys


def generate_n_sequences(model, n, max_len, start_symbol, target, file_path, 
                         multi_target=False, exclude_target=None, aggregate_function='mean'):
    """
    Generate n sequences of SMILES and save them to a file
    """
    smiles = []
    for _ in tqdm(range(n)):
        if multi_target:
            ybar = greedy_decode_multitarget(
                model, max_len, start_symbol, target, aggregate_function
            ).flatten()
        else:
            ybar = greedy_decode(model, max_len, start_symbol, target, exclude_target).flatten()

        ybar = mv.SMILESTokenizer().untokenize(vocabulary.decode(ybar.to('cpu').data.numpy()))
        smiles.append(ybar)

    smiles_df = pd.DataFrame(smiles, columns=['SMILES'])
    smiles_df.to_csv(file_path, index=False, header=True)
    
    print(f"Generated {n} sequences and saved to {file_path}")


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--mode', choices=['train', 'infer', 'baseline', 'finetune'],\
        default='train',help='Run mode')
    arg_parser.add_argument('--device', choices=['cuda:0', 'cpu'],\
        default='cuda:0',help='Device')
    arg_parser.add_argument('--epoch', default='30', type=int)
    arg_parser.add_argument('--batch_size', default='512', type=int)
    arg_parser.add_argument('--layer', default=3, type=int)
    arg_parser.add_argument('--path', default='model_chem.h5', type=str)
    arg_parser.add_argument('--path_ft', default='model_chem_finetune.h5', type=str)
    arg_parser.add_argument('--datamode', default=1, type=int)
    arg_parser.add_argument('--target', default=1, type=int)
    arg_parser.add_argument('--finetune_dataset', default='data/excape_all_active_compounds.smi', type=str)

    # List of targets for inference
    arg_parser.add_argument('--infer_targets', nargs='+', type=int)
    # Multivariative function to aggregate target embeddings
    arg_parser.add_argument('--multivariate', default='mean', type=str)
    # Number of molecules to generate for inference
    arg_parser.add_argument('--num_molecules', default=100, type=int)
    # Path to save generated molecules
    arg_parser.add_argument('--output_path', default='generated_molecules.csv', type=str)
    # Exclude target from decoder input
    arg_parser.add_argument('--exclude_target', type=int)

    arg_parser.add_argument('--d_model', default=1024, type=int)
    arg_parser.add_argument('--nhead', default=8, type=int)
    arg_parser.add_argument('--embedding_size', default=200, type=int)
    arg_parser.add_argument('--loadmodel', default=False, action="store_true")
    arg_parser.add_argument("--loaddata", default=False, action="store_true")
    args = arg_parser.parse_args()

    print('==========  Transformer x->x ==============')

    #scaffold_list, decoration_list = zip(*read_csv_file('zinc/zinc.smi', num_fields=2))
    #vocabulary = mv.DecoratorVocabulary.from_lists(scaffold_list, decoration_list)
    #training_sets = load_sets('zinc/zinc.smi')
    #dataset = md.DecoratorDataset(training_sets, vocabulary=vocabulary)

    mol_list0_train = list(read_delimited_file('data/train.smi'))
    mol_list0_test = list(read_delimited_file('data/test.smi'))
    
    # mol_list1, target_list = zip(*read_csv_file('mol_target_dataloader/target.smi', num_fields=2))
    mol_list1, target_list = zip(*read_csv_file(args.finetune_dataset, num_fields=2))
    mol_list = mol_list0_train
    mol_list.extend(mol_list0_test) 
    mol_list.extend(mol_list1)
    vocabulary = mv.create_vocabulary(smiles_list=mol_list, tokenizer=mv.SMILESTokenizer())
    
    train_data = md.Dataset(mol_list0_train, vocabulary, mv.SMILESTokenizer())
    test_data = md.Dataset(mol_list0_test, vocabulary, mv.SMILESTokenizer())

    BATCH_SIZE = args.batch_size
    SRC_VOCAB_SIZE = len(vocabulary)
    TGT_VOCAB_SIZE = len(vocabulary)

    EMB_SIZE = args.d_model
    NHEAD = args.nhead
    FFN_HID_DIM = 512

    NUM_ENCODER_LAYERS = args.layer
    NUM_DECODER_LAYERS = args.layer
    NUM_EPOCHS = args.epoch
    PAD_IDX = 0
    BOS_IDX = 1
    EOS_IDX = 2
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = args.device

    emb_len = max([int(t) for t in target_list]) + 1
    print(f'# of targets: {emb_len}')
    transformer = ConditionalTransformer(
        NUM_ENCODER_LAYERS, EMB_SIZE, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM, args=args, emb_input_size=emb_len
    )

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    #num_train= int(len(dataset)*0.8)
    #num_test= len(dataset) -num_train
    #train_data, test_data = torch.utils.data.random_split(dataset, [num_train, num_test])
    train_iter = tud.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_data.collate_fn, drop_last=True)
    test_iter = tud.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=test_data.collate_fn, drop_last=True)
    valid_iter = test_iter

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # If fine-tuning, set different optimizers for the embeddings and the rest of the model
    if args.mode == 'finetune':
        optimizer = torch.optim.Adam([
            {'params': transformer.params["conditional"].parameters(), 'lr': 3e-4},
            {'params': transformer.params["generation"].parameters(), 'lr': 1e-4}
        ], betas=(0.9, 0.98), eps=1e-9)
    else:
        optimizer = torch.optim.Adam(
            transformer.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9
        )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95) 
    if args.mode == 'train':
        transformer = transformer.to(DEVICE)

        if args.loadmodel:
            transformer.load_state_dict(torch.load(args.path))

        min_loss, val_loss = float('inf'), float('inf')
        for epoch in range(1, NUM_EPOCHS+1):
            start_time = time.time()
            train_loss = train_epoch(transformer, train_iter, optimizer)
            scheduler.step()
            end_time = time.time()
            torch.cuda.empty_cache()

            if (epoch+1)%25==0:
                torch.save(transformer.state_dict(), args.path+'_'+str(epoch+1))
                print('Model saved every 25 epoches.') 
            
            if (epoch+1)%1==0:
                val_loss = evaluate(transformer, valid_iter)
                if val_loss < min_loss:
                    min_loss = val_loss
                    torch.save(transformer.state_dict(), args.path)
                    print('Model saved!')
                torch.cuda.empty_cache()

            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
                f"Epoch time = {(end_time - start_time):.3f}s"))
    
    elif args.mode == 'finetune':
        from mol_target_dataloader.utils import read_csv_file
        import mol_target_dataloader.dataset as md

        mol_list1, target_list = zip(*read_csv_file(args.finetune_dataset, num_fields=2))
        #vocabulary = mv.create_vocabulary(smiles_list=mol_list, tokenizer=mv.SMILESTokenizer())
        finetune_dataset = md.Dataset(mol_list1, target_list, vocabulary, mv.SMILESTokenizer())
        num_train= int(len(finetune_dataset)*0.8)
        num_test= len(finetune_dataset) -num_train
        train_data, val_data = torch.utils.data.random_split(finetune_dataset, [num_train, num_test])

        train_iter = tud.DataLoader(train_data, args.batch_size, collate_fn=finetune_dataset.collate_fn, shuffle=True)
        val_iter = tud.DataLoader(val_data, args.batch_size, collate_fn=finetune_dataset.collate_fn, shuffle=True)
        transformer = transformer.to(DEVICE)
        transformer.load_state_dict(torch.load(args.path))

        min_loss, val_loss = float('inf'), float('inf')
        for epoch in range(1, NUM_EPOCHS+1):
            start_time = time.time()
            train_loss = train_epoch(transformer, train_iter, optimizer)
            scheduler.step()
            end_time = time.time()
            torch.cuda.empty_cache()

            if (epoch+1)%25==0:
                torch.save(transformer.state_dict(), args.path_ft+'_'+str(epoch+1))
                print('Fine-tunned model saved every 25 epoches.')

            if (epoch+1)%1==0:
                val_loss = evaluate(transformer, val_iter)
                if val_loss < min_loss:
                    min_loss = val_loss
                    torch.save(transformer.state_dict(), args.path_ft)
                    print('Model saved!')
                torch.cuda.empty_cache()

            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
                f"Epoch time = {(end_time - start_time):.3f}s"))

    elif args.mode == 'infer':
        if args.device == 'cpu':
            transformer.load_state_dict(torch.load(args.path,  map_location=torch.device('cpu')))
        else:
            transformer.load_state_dict(torch.load(args.path))
        device = args.device
        transformer.to(device)
        transformer.eval()

        if args.infer_targets:
            _targets = args.infer_targets
            
            print('multi-target: {0}'.format(_targets))
            print('aggregate-function: {0}'.format(args.multivariate))

            generate_n_sequences(
                transformer,
                args.num_molecules, 
                100, BOS_IDX, _targets, 
                args.output_path,
                multi_target=True,
                aggregate_function=args.multivariate
            )
        else:
            _target = args.target

            print('single-target: {0}'.format(_target))
            if args.exclude_target is not None:
                print('exclude-target: {0}'.format(args.exclude_target))

            generate_n_sequences(
                transformer,
                args.num_molecules, 
                100, BOS_IDX, _target, 
                args.output_path,
                exclude_target=args.exclude_target
            )
