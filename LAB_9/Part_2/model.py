from functions import *
from utils import *
import torch
import torch.nn as nn
import numpy as np



class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.4,
                 emb_dropout=0.1, n_layers=1, tie_weights=False):
        super(LM_LSTM, self).__init__()
        self.encoder = nn.Embedding(output_size, emb_size)
        self.decoder = nn.Linear(hidden_size, output_size)
        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False)
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
    
    
class VarDrop(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x):
        if not self.training:
            return x

        if self.mask is None:
            self.mask = x.new_empty(1, x.size(1), requires_grad=False).bernoulli_(1 - self.p)

        print(x.size())
        
        return x * self.mask.div_(1 - self.p)
    
    
class LM_LSTM_VarDrop(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1, tie_weights=False):
        super(LM_LSTM_VarDrop, self).__init__()
        '''
        embedding diventa encoder
        aggiungo self.decoder
        aggiungo if tie_weights
        aggiungo self.embedding
        aggiungo self.tie_weights
        '''
                     
        #self.encoder = nn.Embedding(output_size, emb_size)
        #self.decoder = nn.Linear(hidden_size, output_size)
        if tie_weights:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.emb_dropout = VarDrop(emb_dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False)    
        self.out_dropout = VarDrop(out_dropout)
        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)
        #self.tie_weights = tie_weights
        self.output.weight = self.embedding.weight  # Weight tying
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.emb_dropout(emb)
        lstm_out, _  = self.lstm(emb)
        lstm_out = self.out_dropout(lstm_out)
        output = self.output(lstm_out).permute(0,2,1)
        return output
