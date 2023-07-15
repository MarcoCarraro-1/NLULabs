from utils import *
from model import *
import nltk
nltk.download("subjectivity")
nltk.download("polarity")
from nltk.corpus import movie_reviews
from nltk.corpus import subjectivity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from nltk.sentiment.vader import SentimentIntensityAnalyzer, VaderConstants
import torch
nltk.download('punkt')
import torch.nn as nn
from sklearn.metrics import classification_report
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
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
from nltk.corpus import movie_reviews
nltk.download('movie_reviews')
nltk.download('vader_lexicon')


def eval_acc(tokenized_test_data, model, mlb, test_labels, vectorizer):
    test_tokenized_vec = vectorizer.transform([' '.join(tokens) for tokens in tokenized_test_data])
    test_tokenized_array = test_tokenized_vec.toarray()

    predictions = model.predict(test_tokenized_array)
    
    test_labels_bin = mlb.fit_transform(test_labels)
    accuracy = accuracy_score(test_labels_bin, predictions)
    
    report = classification_report(test_labels_bin, predictions)

    print("Accuracy: ", accuracy)
    print(report)
    
    