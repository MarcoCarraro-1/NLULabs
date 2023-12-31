# -*- coding: utf-8 -*-
"""# Lab Exercise 1
Modify the baseline LM_RNN (the idea is to add a set of improvements and see how these affect the performance). Furthremore, you have to play with the hyperparameters to minimise the PPL and thus print the results achieved with the best configuration. Here are the links to the state-of-the-art papers which uses vanilla RNN paper1, paper2.

* Replace RNN with LSTM (output the PPL)
* Add two dropout layers: (output the PPL)
  * one on embeddings,
  * one on the output
* Replace SGD with AdamW (output the PPL)

## Install & Importing
"""

from gensim.models import Word2Vec
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cosine
import gensim.downloader
import spacy
spacy.cli.download('en_core_web_lg')
from scipy.spatial.distance import cosine
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
#from google.colab import drive
#drive.mount('/content/drive')
import torch.utils.data as data
from functools import partial
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
from torch.optim import AdamW

"""## Functions"""

def cosine_similarity(v, w):
    return np.dot(v,w)/(norm(v)*norm(w))


def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line + eos_token)
    return output

def get_vocab(corpus, special_tokens=[]):
    output = {}
    i = 0
    for st in special_tokens:
        output[st] = i
        i += 1
    for sentence in corpus:
        for w in sentence.split():
            if w not in output:
                output[w] = i
                i += 1
    return output


class Lang():
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}

    def get_vocab(self, corpus, special_tokens=[]):
        output = {}
        i = 0
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output



class PennTreeBank (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []

        for sentence in corpus:
            self.source.append(sentence.split()[0:-1]) # We get from the first token till the second-last token
            self.target.append(sentence.split()[1:]) # We get from the second token till the last token
            # See example in section 6.2

        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)


    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src= torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample

    # Auxiliary methods

    def mapping_seq(self, data, lang): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found!')
                    print('You have to deal with that') # PennTreeBank doesn't have OOV but "Trust is good, control is better!"
                    break
            res.append(tmp_seq)
        return res


def collate_fn(data, pad_token):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    # Sort data by seq lengths

    data.sort(key=lambda x: len(x["source"]), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])

    new_item["source"] = source.to(device)
    new_item["target"] = target.to(device)
    new_item["number_tokens"] = sum(lengths)
    return new_item


def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []

    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # Update the weights

    return sum(loss_array)/sum(number_of_tokens)

def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return

def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

"""## Replace RNN with LSTM"""

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_LSTM, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False)
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        lstm_out, _  = self.lstm(emb)
        output = self.output(lstm_out).permute(0,2,1)
        return output
    def get_word_embedding(self, token):
        return self.embedding(token).squeeze(0).detach().cpu().numpy()

    def get_most_similar(self, vector, top_k=10):
        embs = self.embedding.weight.detach().cpu().numpy()
        #Our function that we used before
        scores = []
        for i, x in enumerate(embs):
            if i != self.pad_token:
                scores.append(cosine_similarity(x, vector))
        # Take ids of the most similar tokens
        scores = np.asarray(scores)
        indexes = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[indexes]
        return (indexes, top_scores)

train_raw = read_file("dataset/ptb.train.txt")
dev_raw = read_file("dataset/ptb.valid.txt")
test_raw = read_file("dataset/ptb.test.txt")

vocab = get_vocab(train_raw, ["<pad>", "<eos>"])

lang = Lang(train_raw, ["<pad>", "<eos>"])

train_dataset = PennTreeBank(train_raw, lang)
dev_dataset = PennTreeBank(dev_raw, lang)
test_dataset = PennTreeBank(test_raw, lang)

train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=256, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
test_loader = DataLoader(test_dataset, batch_size=256, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))



"""### Just Replace"""
"""
# Experiment also with a smaller or bigger model by changing hid and emb sizes
# A large model tends to overfit
hid_size = 200
emb_size = 200

# Don't forget to experiment with a lower training batch size

# With SGD try with an higer learning rate
lr = 0.0001 # This is definitely not good for SGD
clip = 5 # Clip the gradient
device = 'cuda:0'

vocab_len = len(lang.word2id)

model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
model.apply(init_weights)

optimizer = optim.SGD(model.parameters(), lr=lr)
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
        if  ppl_dev < best_ppl: # the lower, the better
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to('cpu')
            patience = 3
        else:
            patience -= 1

        if patience <= 0: # Early stopping with patience
            break # Not nice but it keeps the code clean

best_model.to(device)
final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
print("Replace RNN with LSTM: ")
print()
print('Test ppl: ', final_ppl)
'''Test ppl:  8927.317317345098'''
"""

"""### Changing Learning Rate"""
"""
hid_size = 250
emb_size = 400


# With SGD try with an higer learning rate
lr = 0.5
clip = 5 # Clip the gradient
device = 'cuda:0'

vocab_len = len(lang.word2id)

model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
model.apply(init_weights)

optimizer = optim.SGD(model.parameters(), lr=lr)
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
        if  ppl_dev < best_ppl: # the lower, the better
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to('cpu')
            patience = 3
        else:
            patience -= 1

        if patience <= 0: # Early stopping with patience
            break # Not nice but it keeps the code clean

best_model.to(device)
final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
print("Changing Learning Rate: ")
print()
print('Test ppl: ', final_ppl)
'''Test ppl:  239.16614436083387'''
"""

## Add dropout layers
"""
class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.3, n_layers=1):
        super(LM_LSTM, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.dropout_emb = nn.Dropout(emb_dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False)
        self.dropout_out = nn.Dropout(out_dropout)
        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.dropout_emb(emb)
        lstm_out, _  = self.lstm(emb)
        lstm_out = self.dropout_out(lstm_out)
        output = self.output(lstm_out).permute(0,2,1)
        return output
    def get_word_embedding(self, token):
        return self.embedding(token).squeeze(0).detach().cpu().numpy()

    def get_most_similar(self, vector, top_k=10):
        embs = self.embedding.weight.detach().cpu().numpy()
        scores = []
        for i, x in enumerate(embs):
            if i != self.pad_token:
                scores.append(cosine_similarity(x, vector))
        # Take ids of the most similar tokens
        scores = np.asarray(scores)
        indexes = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[indexes]
        return (indexes, top_scores)

hid_size = 250
emb_size = 400


# With SGD try with an higer learning rate
lr = 1
clip = 5 # Clip the gradient
device = 'cuda:0'

vocab_len = len(lang.word2id)

model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
model.apply(init_weights)

optimizer = optim.SGD(model.parameters(), lr=lr)
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
        if  ppl_dev < best_ppl: # the lower, the better
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to('cpu')
            patience = 3
        else:
            patience -= 1

        if patience <= 0: # Early stopping with patience
            break # Not nice but it keeps the code clean

best_model.to(device)
final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
print("Adding dropout layers: ")
print()
print('Test ppl: ', final_ppl)
'''Test ppl:  234.78091642129814'''
"""
## Substitute SGD with AdamW
"""
hid_size = 250
emb_size = 400


# With SGD try with an higer learning rate
lr = 0.0001
clip = 5 # Clip the gradient
device = 'cuda:0'

vocab_len = len(lang.word2id)

model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
model.apply(init_weights)

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01, eps=1e-6)
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
        if  ppl_dev < best_ppl: # the lower, the better
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to('cpu')
            patience = 3
        else:
            patience -= 1

        if patience <= 0: # Early stopping with patience
            break # Not nice but it keeps the code clean

best_model.to(device)
final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
print("Replace SGD with AdamW: ")
print()
print('Test ppl: ', final_ppl)
'''Test ppl:  187.16777831184564'''
"""

## Weight Tying

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.4,
                 emb_dropout=0.1, n_layers=1, tie_weights=False):
        super(LM_LSTM, self).__init__()
        self.encoder = nn.Embedding(output_size, emb_size)
        self.decoder = nn.Linear(hidden_size, output_size)
        if tie_weights:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        #self.dropout_emb = nn.Dropout(emb_dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False)
        #self.dropout_out = nn.Dropout(out_dropout)
        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)
        self.tie_weights = tie_weights

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        #emb = self.dropout_emb(emb)
        lstm_out, _  = self.lstm(emb)
        #lstm_out = self.dropout_out(lstm_out)
        output = self.output(lstm_out).permute(0,2,1)
        return output
    def get_word_embedding(self, token):
        return self.embedding(token).squeeze(0).detach().cpu().numpy()

    def get_most_similar(self, vector, top_k=10):
        embs = self.embedding.weight.detach().cpu().numpy()
        scores = []
        for i, x in enumerate(embs):
            if i != self.pad_token:
                scores.append(cosine_similarity(x, vector))
        # Take ids of the most similar tokens
        scores = np.asarray(scores)
        indexes = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[indexes]
        return (indexes, top_scores)


train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

"""
hid_size = 250
emb_size = 400
torch.cuda.empty_cache()

# With SGD try with an higer learning rate
lr = 0.0001
#lr = 30
clip = 0.25 # Clip the gradient
device = 'cuda:0'
decay = 1.2e-6
mu = 0.9
epsilon = 1e-8

vocab_len = len(lang.word2id)

model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"], tie_weights=True).to(device)
model.apply(init_weights)

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01, eps=1e-6)
#optimizer = optim.SGD(model.parameters(), lr=lr)
#optimizer = torch.optim.ASGD(model.parameters(), lr=lr, t0=0.5, lambd=0.5, weight_decay=decay)
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
        if  ppl_dev < best_ppl: # the lower, the better
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to('cpu')
            patience = 3
        else:
            patience -= 1
        
        if patience <= 0: # Early stopping with patience
            break # Not nice but it keeps the code clean

    '''
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            param_state = optimizer.state[p]
            if 'momentum_buffer' not in param_state:
                buf = param_state['momentum_buffer'] = torch.clone(p.grad).detach()
            else:
                buf = param_state['momentum_buffer']
                buf.mul_(mu).add_(p.grad, alpha=1 - mu)
                p.grad = buf

            p.add_(p.grad, alpha=-group['lr'])
            p.add_(torch.sign(p) * epsilon)
    '''

best_model.to(device)
final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
print("Weight Tying: ")
print()
print('Test ppl: ', final_ppl)
"""
'''Test ppl:  178.24676073494174'''

## Variational dropout
"""
class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.3, n_layers=1, tie_weights=False):
        super(LM_LSTM, self).__init__()
        self.encoder = nn.Embedding(output_size, emb_size)
        self.decoder = nn.Linear(hidden_size, output_size)
        if tie_weights:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        #self.dropout_emb = nn.Dropout(emb_dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False)
        #self.dropout_out = nn.Dropout(out_dropout)
        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)
        self.tie_weights = tie_weights

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        #emb = self.dropout_emb(emb)
        lstm_out, _  = self.lstm(emb)
        #lstm_out = self.dropout_out(lstm_out)
        output = self.output(lstm_out).permute(0,2,1)
        return output
    def get_word_embedding(self, token):
        return self.embedding(token).squeeze(0).detach().cpu().numpy()

    def get_most_similar(self, vector, top_k=10):
        embs = self.embedding.weight.detach().cpu().numpy()
        scores = []
        for i, x in enumerate(embs):
            if i != self.pad_token:
                scores.append(cosine_similarity(x, vector))
        # Take ids of the most similar tokens
        scores = np.asarray(scores)
        indexes = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[indexes]
        return (indexes, top_scores)


hid_size = 250
emb_size = 400
torch.cuda.empty_cache()


# With SGD try with an higer learning rate
lr = 0.0001
clip = 5 # Clip the gradient
device = 'cuda:0'

vocab_len = len(lang.word2id)

model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"], tie_weights=True).to(device)
model.apply(init_weights)

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01, eps=1e-6)
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
        if  ppl_dev < best_ppl: # the lower, the better
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to('cpu')
            patience = 3
        else:
            patience -= 1

        if patience <= 0: # Early stopping with patience
            break # Not nice but it keeps the code clean

best_model.to(device)
final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
print("Variational dropout: ")
print()
print('Test ppl: ', final_ppl)
'''Test ppl:  176.82281834176788'''
"""


## Non monotonically Triggered AvSGD

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.3, n_layers=1, tie_weights=False):
        super(LM_LSTM, self).__init__()
        self.encoder = nn.Embedding(output_size, emb_size)
        self.decoder = nn.Linear(hidden_size, output_size)
        if tie_weights:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        #self.dropout_emb = nn.Dropout(emb_dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False)
        #self.dropout_out = nn.Dropout(out_dropout)
        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)
        self.tie_weights = tie_weights

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        #emb = self.dropout_emb(emb)
        lstm_out, _  = self.lstm(emb)
        #lstm_out = self.dropout_out(lstm_out)
        output = self.output(lstm_out).permute(0,2,1)
        return output
    def get_word_embedding(self, token):
        return self.embedding(token).squeeze(0).detach().cpu().numpy()

    def get_most_similar(self, vector, top_k=10):
        embs = self.embedding.weight.detach().cpu().numpy()
        scores = []
        for i, x in enumerate(embs):
            if i != self.pad_token:
                scores.append(cosine_similarity(x, vector))
        # Take ids of the most similar tokens
        scores = np.asarray(scores)
        indexes = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[indexes]
        return (indexes, top_scores)

hid_size = 250
emb_size = 400


# With SGD try with an higer learning rate
lr = 30
clip = 0.25 # Clip the gradient
device = 'cuda:0'

vocab_len = len(lang.word2id)

model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"], tie_weights=True).to(device)
model.apply(init_weights)

optimizer = torch.optim.ASGD(model.parameters(), lr=lr, t0=0, lambd=0., weight_decay=1.2e-6)
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
        if  ppl_dev < best_ppl: # the lower, the better
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to('cpu')
            patience = 3
        else:
            patience -= 1

        if patience <= 0: # Early stopping with patience
            break # Not nice but it keeps the code clean

    """
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            param_state = optimizer.state[p]
            if 'momentum_buffer' not in param_state:
                buf = param_state['momentum_buffer'] = torch.clone(p.grad).detach()
            else:
                buf = param_state['momentum_buffer']
                buf.mul_(mu).add_(p.grad, alpha=1 - mu)
                p.grad = buf

            p.add_(p.grad, alpha=-group['lr'])
            p.add_(torch.sign(p) * epsilon)
    """

best_model.to(device)
final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
print("Non monotonically Triggered AvSGD: ")
print()
print('Test ppl: ', final_ppl)
'''Test ppl:  204.66631850765194'''
