from utils import *
from model import *
#from gensim.models import Word2Vec
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cosine
#import gensim.downloader
#import spacy
#spacy.cli.download('en_core_web_lg')
from scipy.spatial.distance import cosine
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import torch.utils.data as data
from functools import partial
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
from torch.optim import AdamW


def load_data(train_fold, valid_fold, test_fold):
    train_raw = read_file(train_fold)
    dev_raw = read_file(valid_fold)
    test_raw = read_file(test_fold)

    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])

    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=256, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=256, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    
    return vocab, lang, train_loader, dev_loader, test_loader


def eval_ppl(lang, train_loader, dev_loader, test_loader, call):
    hid_size = 250
    emb_size = 400
    
    device = 'cuda:0'
    vocab_len = len(lang.word2id)
    
    model = LM_LSTM(emb_size, hid_size, vocab_len, call, pad_index=lang.word2id["<pad>"]).to('cuda:0')
    model.apply(init_weights)
    
    if call == 0 or call == 1:
        lr = 0.5
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        lr = 0.0001
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01, eps=1e-6)
    
    clip = 5
    
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    n_epochs = 100
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1,n_epochs))
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)

        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)
            if  ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1

            if patience <= 0:
                break 

    best_model.to('cuda:0')
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
    if call == 0:
        torch.save(best_model.state_dict(), 'bin/replace_rnn_lstm.bin')
        print("Replace RNN and Changing Learning Rate: ")
    elif call == 1:
        torch.save(best_model.state_dict(), 'bin/dropout_lstm.bin')
        print("Adding dropout layers: ")
    else:
        torch.save(best_model.state_dict(), 'bin/replace_sgd_adamw_lstm.bin')
        print("Replace SGD with AdamW: ")
    print('Test ppl: ', final_ppl, '\n')
