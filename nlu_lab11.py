# -*- coding: utf-8 -*-
"""# Lab Exercise1
Create a pipeline model for Subjectivity and Polarity detection tasks. The pipeline has to be composed of two different models:

1. The first model predicts if a sentence is subjective or objective;
2. The second model performs the polarity detection of a document after removing the objective sentences predicted by the first model;

You have to report the results of the first and the second models. For the second model, you have to report the resutls achieved with and without the removal of the objective sentences to see if the pipeline can actually improve the performance.

**The type of model**: You have to choose a Neutral Network in PyTorch (e.g. MLP or RNN ) or a pre-trained language model (e.g. BERT or T5).

**Datasets**:

- NLTK: subjectivity (Subjectivity task)
- NLTK: movie reviews (Polarity task)

**Evaluation**:

Use a K-fold evaluation for both tasks where with K = 10

## Install & import
"""

#!pip install transformers

import nltk
nltk.download("subjectivity")
from nltk.corpus import movie_reviews
from nltk.corpus import subjectivity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from nltk.sentiment.vader import SentimentIntensityAnalyzer, VaderConstants
import torch
from transformers import BertTokenizer, BertForSequenceClassification
nltk.download('punkt')
import torch.nn as nn
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
nltk.download('stopwords')
nltk.download('wordnet')
import numpy as np
from nltk.sentiment import SentimentAnalyzer
import pandas as pd
from nltk import NaiveBayesClassifier
from nltk.sentiment.util import mark_negation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from tqdm import tqdm

"""## Load Data"""

n_instances = 10000

subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]

len(subj_docs), len(obj_docs)
print(subj_docs[0])

"""## Preprocessing Data"""

def preprocess_text(text):
    tokens = word_tokenize(text)  # Tokenizzazione
    tokens = [token.lower() for token in tokens if token not in string.punctuation]  # Rimozione della punteggiatura
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]  # Rimozione delle stop word
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatizzazione
    return tokens

subj_prep = []
obj_prep = []
all_prep = []

for sent, label in subj_docs:
    preprocessed_sentence = preprocess_text(' '.join(sent))
    subj_prep.append([preprocessed_sentence, label])

for sent, label in obj_docs:
    preprocessed_sentence = preprocess_text(' '.join(sent))
    obj_prep.append([preprocessed_sentence, label])

all_prep = subj_prep + obj_prep

print(len(subj_prep))
print(len(obj_prep))
print(len(all_prep))
print(subj_prep[0])
print(subj_prep[1])

"""## Split Data"""

train_subj = subj_prep[:4000]
test_subj = subj_prep[4000:5000]
train_obj = obj_prep[:4000]
test_obj = obj_prep[4000:5000]
train_all = train_subj+train_obj
test_all = test_subj+test_obj

sentim_analyzer = SentimentAnalyzer()
all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in train_all])

X_train = []
y_train = []
X_test = []
y_test = []

for sent, label in train_all:
  X_train.append(' '.join(sent))
  y_train.append(label)

for sent, label in test_all:
  X_test.append(' '.join(sent))
  y_test.append(label)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

"""## MLP Model"""

model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam')
model.fit(X_train_vectorized, y_train)

# Valutazione del modello
X_test_vectorized = vectorizer.transform(X_test)
predictions = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

"""## BERT Model"""

device = torch.device("cuda")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenizzazione e codifica dei dati di addestramento
X_train_encoded = tokenizer.batch_encode_plus(X_train, padding=True, truncation=True, max_length=128, return_tensors='pt')
X_train_input_ids = X_train_encoded['input_ids']
X_train_attention_mask = X_train_encoded['attention_mask']

model.to(device)
X_train_input_ids = X_train_input_ids.to(device)
X_train_attention_mask = X_train_attention_mask.to(device)
y_train = y_train.to(device)

# Addestramento del modello BERT
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
num_epochs = 5

for epoch in tqdm(range(num_epochs)):
    outputs = model(input_ids=X_train_input_ids, attention_mask=X_train_attention_mask, labels=y_train)
    loss = outputs.loss
    logits = outputs.logits

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Tokenizzazione e codifica dei dati di test
X_test_encoded = tokenizer.batch_encode_plus(X_test, padding=True, truncation=True, max_length=128, return_tensors='pt')
X_test_input_ids = X_test_encoded['input_ids']
X_test_attention_mask = X_test_encoded['attention_mask']

# Valutazione del modello BERT
model.eval()
with torch.no_grad():
    outputs = model(input_ids=X_test_input_ids, attention_mask=X_test_attention_mask)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1)

# Calcolo dell'accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
