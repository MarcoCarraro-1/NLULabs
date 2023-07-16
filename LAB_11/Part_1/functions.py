from utils import *
from model import *
import nltk
nltk.download("subjectivity")
nltk.download("polarity")
from nltk.corpus import movie_reviews
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
nltk.download('wordnet')
import numpy as np
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
import joblib


def eval_acc1(X_train_vectorized, X_test_vectorized, y_train, y_test):
    model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam')
    model.fit(X_train_vectorized, y_train)
    
    subj_predictions = model.predict(X_test_vectorized)
    subj_accuracy = accuracy_score(y_test, subj_predictions)
    print("[SUBJECTIVITY] Accuracy with no K-fold:", subj_accuracy)
    
    return subj_accuracy, model


def eval_acc2(X_train, X_test, y_train, y_test):
    x = X_train + X_test
    y = y_train + y_test

    vectorizer = CountVectorizer()
    clf = MLPClassifier()
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    accuracies = []
    best_accuracy = 0
    best_model = None

    for train_index, test_index in tqdm(skf.split(x, y)):
        x_train_fold, x_test_fold = [x[i] for i in train_index], [x[i] for i in test_index]
        y_train_fold, y_test_fold = [y[i] for i in train_index], [y[i] for i in test_index]

        train_features = vectorizer.fit_transform(x_train_fold)
        test_features = vectorizer.transform(x_test_fold)

        clf.fit(train_features, y_train_fold)
        predictions = clf.predict(test_features)

        accuracy = accuracy_score(y_test_fold, predictions)
        accuracies.append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = clf

    mean_accuracy = sum(accuracies) / len(accuracies)
    joblib.dump(best_model, 'bin/subjectivity_all_sents.bin')
    
    print("[SUBJECTIVITY] Mean Accuracy - All sentences: ", mean_accuracy)
    
    return mean_accuracy
    
    
def get_movie_data():
    movie_sentences = movie_reviews.sents()
    polarity_all = []
    sid = SentimentIntensityAnalyzer()

    for sentence in tqdm(movie_sentences):
        sentence_str = ' '.join(sentence)
        ss = sid.polarity_scores(sentence_str)
        polarity_all.append(ss)
        
    return movie_sentences, polarity_all


def get_prep_movie(movie_sentences):
    movie_prep = []

    for sent in tqdm(movie_sentences):
        preprocessed_sentence = preprocess_text(' '.join(sent))
        movie_prep.append(preprocessed_sentence)

    print("Length movie dataset: ", len(movie_prep))
    print("Example: ", movie_prep[0])
    
    return movie_prep


def split_movie_data(movie_prep):
    movie_train, movie_test = train_test_split(movie_prep, test_size=0.2, random_state=42)
    sid = SentimentIntensityAnalyzer()
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
    print("MOVIES TRAINING VALUES:")
    print(len(polarity_train))
    values = pd.Series(polarity_train).value_counts()
    print(values)
    print()
    print("MOVIES TEST VALUES:")
    print(len(polarity_test))
    values = pd.Series(polarity_test).value_counts()
    print(values)
    
    return movie_train, movie_test, polarity_train, polarity_test


def get_movie_x_y(movie_train, polarity_train, movie_test, polarity_test):
    X_train_movie = []
    y_train_movie = []
    X_test_movie = []
    y_test_movie = []

    for sent, polarity in zip(movie_train, polarity_train):
        X_train_movie.append(' '.join(sent))
        y_train_movie.append(polarity)

    for sent, polarity in zip(movie_test, polarity_test):
        X_test_movie.append(' '.join(sent))
        y_test_movie.append(polarity)


    #polarity_vectorizer = CountVectorizer()
    #X_train_vectorized = polarity_vectorizer.fit_transform(X_train_movie)

    x_movie = X_train_movie + X_test_movie
    y_movie = y_train_movie + y_test_movie
    
    return x_movie, y_movie


def eval_acc3(x_movie, y_movie):
    vectorizer = CountVectorizer()
    model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam')
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    accuracies = []
    best_accuracy = 0
    best_model = None

    for train_index, test_index in tqdm(skf.split(x_movie, y_movie), total=skf.get_n_splits()):
        x_train_fold, x_test_fold = [x_movie[i] for i in train_index], [x_movie[i] for i in test_index]
        y_train_fold, y_test_fold = [y_movie[i] for i in train_index], [y_movie[i] for i in test_index]

        train_features = vectorizer.fit_transform(x_train_fold)
        test_features = vectorizer.transform(x_test_fold)

        model.fit(train_features, y_train_fold)
        predictions = model.predict(test_features)

        accuracy = accuracy_score(y_test_fold, predictions)
        accuracies.append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    mean_accuracy = sum(accuracies) / len(accuracies)
    joblib.dump(best_model, 'bin/polarity_all_sents.bin')
    
    print("[POLARITY] Mean Accuracy - All sentences: ", mean_accuracy)

    return mean_accuracy, model, skf


def remove_obj_sents(movie_train, movie_test, subj_vectorizer,
                     model):
    #polarity_all = polarity_train + polarity_test
    movie_all = movie_train + movie_test
    movie_temp = []

    for sent in movie_all:
        movie_temp.append(' '.join(sent))

    movie_all = movie_temp
    movie_all_vectorized = subj_vectorizer.transform(movie_all)
    movie_subj_pred = model.predict(movie_all_vectorized)
    movie_no_obj = []

    for sent, label in zip(movie_all, movie_subj_pred):
        if label == 'subj':
            movie_no_obj.append(sent)

    print("Length of movie subjective sentences: ", len(movie_no_obj))
    movie_temp = []

    for sent in movie_no_obj:
        movie_temp.append(sent.split())

    movie_no_obj = movie_temp
    sid = SentimentIntensityAnalyzer()
    polarity_no_obj = []

    for sentence in tqdm(movie_no_obj):
        sentence_str = ' '.join(sentence)
        ss = sid.polarity_scores(sentence_str)
        if(ss['compound'] == 0):
            pol = 'Neu'
        elif(ss['compound'] > 0):
            pol = 'Pos'
        else:
            pol = 'Neg'
        polarity_no_obj.append(pol)

    print("Length of movie subjective polarity: ", len(polarity_no_obj))
    counts = pd.Series(polarity_no_obj).value_counts()
    print(counts)
    movie_temp = []

    for sent in movie_no_obj:
        movie_temp.append(' '.join(sent))

    movie_no_obj = movie_temp

    return movie_no_obj, polarity_no_obj


def eval_acc4(skf_polarity, movie_no_obj, polarity_no_obj, model_polarity):
    accuracies = []
    best_accuracy = 0
    best_model = None
    polarity_vectorizer = CountVectorizer()

    for train_index, test_index in tqdm(skf_polarity.split(movie_no_obj, polarity_no_obj), total=skf_polarity.get_n_splits()):
        x_train_fold, x_test_fold = [movie_no_obj[i] for i in train_index], [movie_no_obj[i] for i in test_index]
        y_train_fold, y_test_fold = [polarity_no_obj[i] for i in train_index], [polarity_no_obj[i] for i in test_index]

        train_features = polarity_vectorizer.fit_transform(x_train_fold)
        test_features = polarity_vectorizer.transform(x_test_fold)

        model_polarity.fit(train_features, y_train_fold)
        predictions = model_polarity.predict(test_features)

        accuracy = accuracy_score(y_test_fold, predictions)
        accuracies.append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model_polarity

    mean_accuracy = sum(accuracies) / len(accuracies)
    joblib.dump(best_model, 'bin/polarity_without_objective.bin')
    
    print("[POLARITY] Mean Accuracy - Without objective sentences: ", mean_accuracy)
    
    return mean_accuracy
