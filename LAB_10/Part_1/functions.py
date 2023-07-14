import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from collections import Counter
import os
import json
from pprint import pprint
import random
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import sys
#from conll import evaluate
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.utils.data as data
#from transformers import BertModel, TFBertModel
#from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling1D
from sklearn.utils.class_weight import compute_class_weight
from utils import *
from model import *
import main


def load_data(train_fold, test_fold):
    device = 'cuda:0' # cuda:0 means we are using the GPU with id 0, if you have multiple GPU
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
    
    tmp_train_raw = read_file(train_fold)
    test_raw = read_file(test_fold)
    print('Train samples:', len(tmp_train_raw))
    print('Test samples:', len(test_raw))

    portion = round(((len(tmp_train_raw) + len(test_raw)) * 0.10)/(len(tmp_train_raw)),2)

    intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
    count_y = Counter(intents)

    Y = []
    X = []
    mini_Train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: # If some intents occure once only, we put them in training
            X.append(tmp_train_raw[id_y])
            Y.append(y)
        else:
            mini_Train.append(tmp_train_raw[id_y])
    # Random Stratify
    X_train, X_dev, y_train, y_dev = train_test_split(X, Y, test_size=portion,
                                                        random_state=42,
                                                        shuffle=True,
                                                        stratify=Y)
    X_train.extend(mini_Train)
    train_raw = X_train
    dev_raw = X_dev

    y_test = [x['intent'] for x in test_raw]
    
    return PAD_TOKEN, train_raw, dev_raw, test_raw, y_train, y_dev, y_test


def get_lang(train_raw, dev_raw, test_raw, y_train, y_dev, y_test, PAD_TOKEN):
    print('Train:')
    pprint({k:round(v/len(y_train),3)*100 for k, v in sorted(Counter(y_train).items())})
    print('Dev:'),
    pprint({k:round(v/len(y_dev),3)*100 for k, v in sorted(Counter(y_dev).items())})
    print('Test:')
    pprint({k:round(v/len(y_test),3)*100 for k, v in sorted(Counter(y_test).items())})
    print('='*89)
    # Dataset size
    print('TRAIN size:', len(train_raw))
    print('DEV size:', len(dev_raw))
    print('TEST size:', len(test_raw))

    w2id = {'pad':PAD_TOKEN, 'unk': 1}
    slot2id = {'pad':PAD_TOKEN}
    intent2id = {}

    for example in train_raw:
        for w in example['utterance'].split():
            if w not in w2id:
                w2id[w] = len(w2id)
        for slot in example['slots'].split():
            if slot not in slot2id:
                slot2id[slot] = len(slot2id)
        if example['intent'] not in intent2id:
            intent2id[example['intent']] = len(intent2id)

    for example in dev_raw:
        for slot in example['slots'].split():
            if slot not in slot2id:
                slot2id[slot] = len(slot2id)
        if example['intent'] not in intent2id:
            intent2id[example['intent']] = len(intent2id)

    for example in test_raw:
        for slot in example['slots'].split():
            if slot not in slot2id:
                slot2id[slot] = len(slot2id)
        if example['intent'] not in intent2id:
            intent2id[example['intent']] = len(intent2id)

    print('# Vocab:', len(w2id)-2)
    print('# Slots:', len(slot2id)-1)
    print('# Intent:', len(intent2id))

    words = sum([x['utterance'].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])
    lang = Lang(words, intents, slots, PAD_TOKEN, cutoff=0)
    
    return lang


def get_dataset(train_raw, dev_raw, test_raw, lang):
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)
    
    return train_dataset, dev_dataset, test_dataset


def get_dataload(train_dataset, dev_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    return train_loader, dev_loader, test_loader


def eval_f1_acc(lang, train_loader, dev_loader, test_loader, PAD_TOKEN,call):
    hid_size = 200
    emb_size = 300

    lr = 0.0001 # learning rate
    clip = 5 # Clip the gradient

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    runs = 5
    slot_f1s, intent_acc = [], []
    for x in tqdm(range(0, runs)):
        if call==0:
            model = ModelIAS_bidir(hid_size, out_slot, out_int, emb_size,
                        vocab_len, pad_index=PAD_TOKEN).to('cuda:0')
        else:
            model = ModelIAS_dropout(hid_size, out_slot, out_int, emb_size,
                        vocab_len, pad_index=PAD_TOKEN).to('cuda:0')
            
        model.apply(init_weights)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()

        n_epochs = 200
        patience = 3
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0
        for x in tqdm(range(1,n_epochs)):
            loss = train_loop(train_loader, optimizer, criterion_slots,
                            criterion_intents, model)
            if x % 5 == 0:
                sampled_epochs.append(x)
                losses_train.append(np.asarray(loss).mean())
                results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots,
                                                            criterion_intents, model, lang)
                losses_dev.append(np.asarray(loss_dev).mean())
                f1 = results_dev['total']['f']

                if f1 > best_f1:
                    best_f1 = f1
                else:
                    patience -= 1
                if patience <= 0: # Early stoping with patient
                    break # Not nice but it keeps the code clean

        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots,
                                                criterion_intents, model, lang)
        intent_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])
    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)
    f1_mean = round(slot_f1s.mean(),3)
    f1_std = round(slot_f1s.std(),3)
    acc_mean = round(intent_acc.mean(), 3)
    acc_std = round(slot_f1s.std(), 3)
    print('Slot F1', f1_mean, '+-', f1_std)
    print('Intent Acc', acc_mean, '+-', acc_std)
    
    return f1_mean, acc_mean
