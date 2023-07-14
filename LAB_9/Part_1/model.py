from functions import *
from utils import *
import torch
import torch.nn as nn

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, call, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_LSTM, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False)
        self.call = call
        if  call==1:
            self.dropout_out = nn.Dropout(out_dropout)        
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        if self.call==1:
            emb = self.dropout_emb(emb)
        lstm_out, _  = self.lstm(emb)
        if self.call==1:
            lstm_out = self.dropout_out(lstm_out)
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