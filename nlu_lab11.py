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
nltk.download("polarity")
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
from nltk.corpus import movie_reviews
nltk.download('movie_reviews')
nltk.download('vader_lexicon')
'''
"""## Load Data"""

n_instances = 10000

subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]

len(subj_docs), len(obj_docs)
print(subj_docs[0])
'''

"""## Preprocessing Data"""

def preprocess_text(text):
    tokens = word_tokenize(text)  # Tokenizzazione
    tokens = [token.lower() for token in tokens if token not in string.punctuation]  # Rimozione della punteggiatura
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]  # Rimozione delle stop word
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatizzazione
    return tokens
'''
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

"""## Subjectivity - MLP Model"""

model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam')
model.fit(X_train_vectorized, y_train)

# Valutazione del modello
X_test_vectorized = vectorizer.transform(X_test)
predictions = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

"""## Polarity - MLP Model"""

x = X_train + X_test
y = y_train + y_test

vectorizer = CountVectorizer()

clf = MLPClassifier()

skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

accuracies = []

for train_index, test_index in tqdm(skf.split(x, y)):
    x_train_fold, x_test_fold = [x[i] for i in train_index], [x[i] for i in test_index]
    y_train_fold, y_test_fold = [y[i] for i in train_index], [y[i] for i in test_index]

    train_features = vectorizer.fit_transform(x_train_fold)
    test_features = vectorizer.transform(x_test_fold)

    clf.fit(train_features, y_train_fold)

    predictions = clf.predict(test_features)

    accuracy = accuracy_score(y_test_fold, predictions)
    accuracies.append(accuracy)

mean_accuracy = sum(accuracies) / len(accuracies)

print("Mean Accuracy:", mean_accuracy)


X_train_subj = []
y_train_subj = []
X_test_subj = []
y_test_subj = []

for sent, label in train_subj:
  X_train_subj.append(' '.join(sent))
  y_train_subj.append(label)

for sent, label in test_subj:
  X_test_subj.append(' '.join(sent))
  y_test_subj.append(label)

vectorizer = CountVectorizer()
X_train__subj_vectorized = vectorizer.fit_transform(X_train_subj)

x = X_train_subj + X_test_subj
y = y_train_subj + y_test_subj

vectorizer = CountVectorizer()

clf = MLPClassifier()

skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

accuracies = []

for train_index, test_index in tqdm(skf.split(x, y)):
    x_train_fold, x_test_fold = [x[i] for i in train_index], [x[i] for i in test_index]
    y_train_fold, y_test_fold = [y[i] for i in train_index], [y[i] for i in test_index]

    train_features = vectorizer.fit_transform(x_train_fold)
    test_features = vectorizer.transform(x_test_fold)

    clf.fit(train_features, y_train_fold)

    predictions = clf.predict(test_features)

    accuracy = accuracy_score(y_test_fold, predictions)
    accuracies.append(accuracy)

mean_accuracy = sum(accuracies) / len(accuracies)

print("Mean Accuracy:", mean_accuracy)
'''

movie_sentences = movie_reviews.sents()
polarity_all = []
sid = SentimentIntensityAnalyzer()

for sentence in tqdm(movie_sentences):
  sentence_str = ' '.join(sentence)
  ss = sid.polarity_scores(sentence_str)
  polarity_all.append(ss)
  #print(sentence_str, ss)
  #print()

print(type(movie_sentences))
print(type(polarity_all))

##[POLARITY] Preprocessing Data

movie_prep = []

for sent in tqdm(movie_sentences):
    preprocessed_sentence = preprocess_text(' '.join(sent))
    movie_prep.append(preprocessed_sentence)

print(len(movie_prep))
print(movie_prep[0])

##[POLARITY] Split Data

movie_train, movie_test = train_test_split(movie_prep, test_size=0.2, random_state=42)

print(len(movie_train), len(movie_test))

polarity_train = []
polarity_test = []

for sentence in tqdm(movie_train):
  sentence_str = ' '.join(sentence)
  ss = sid.polarity_scores(sentence_str)
  if(ss['compound'] == 0):
    pol = 'Neu'
  elif(ss['compound'] > 0):
    pol = 'Pos'
  else:
    pol = 'Neg'
  polarity_train.append(pol)

for sentence in tqdm(movie_test):
  sentence_str = ' '.join(sentence)
  ss = sid.polarity_scores(sentence_str)
  if(ss['compound'] == 0):
    pol = 'Neu'
  elif(ss['compound'] > 0):
    pol = 'Pos'
  else:
    pol = 'Neg'
  polarity_test.append(pol)

print()
print("TRAINING VALUES:")
print(len(polarity_train))
values = pd.Series(polarity_train).value_counts()
print(values)
print()
print("TEST VALUES:")
print(len(polarity_test))
values = pd.Series(polarity_test).value_counts()
print(values)

X_train = []
y_train = []
X_test = []
y_test = []

for sent, polarity in zip(movie_train, polarity_train):
  X_train.append(' '.join(sent))
  y_train.append(polarity)

for sent, polarity in zip(movie_test, polarity_test):
  X_test.append(' '.join(sent))
  y_test.append(polarity)

print(len(X_train), len(X_test))
print(len(y_test))

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

## [POLARITY] MLP Model

x = X_train + X_test
y = y_train + y_test

vectorizer = CountVectorizer()

clf = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam')

skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

accuracies = []

for train_index, test_index in tqdm(skf.split(x, y)):
    x_train_fold, x_test_fold = [x[i] for i in train_index], [x[i] for i in test_index]
    y_train_fold, y_test_fold = [y[i] for i in train_index], [y[i] for i in test_index]

    train_features = vectorizer.fit_transform(x_train_fold)
    test_features = vectorizer.transform(x_test_fold)

    clf.fit(train_features, y_train_fold)

    predictions = clf.predict(test_features)

    accuracy = accuracy_score(y_test_fold, predictions)
    accuracies.append(accuracy)

mean_accuracy = sum(accuracies) / len(accuracies)

print("[POLARITY] Mean Accuracy - All sentences: ", mean_accuracy)
