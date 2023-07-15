from functions import *
from model import *
import nltk
nltk.download("subjectivity")
from nltk.corpus import subjectivity
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from nltk.sentiment import SentimentAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

def get_data(subj, obj):
    n_instances = 10000

    subj_docs = [(sent, subj) for sent in subjectivity.sents(categories='subj')[:n_instances]]
    obj_docs = [(sent, obj) for sent in subjectivity.sents(categories='obj')[:n_instances]]

    len(subj_docs), len(obj_docs)
    print("Sentence of Subjectivity dataset: ", subj_docs[0])
    
    return subj_docs, obj_docs


def preprocess_text(text):
    tokens = word_tokenize(text)  # Tokenizzazione
    tokens = [token.lower() for token in tokens if token not in string.punctuation]  # Rimozione della punteggiatura
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]  # Rimozione delle stop word
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatizzazione
    return tokens


def get_prep_sents(subj_docs, obj_docs):
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

    print("Subjective sentences: ", len(subj_prep))
    print("Objective sentences: ", len(obj_prep))
    print("All sentences: ", len(all_prep))
    print(subj_prep[0])
    print(subj_prep[1])
    
    return subj_prep, obj_prep, all_prep


def split_data(subj_prep, obj_prep):
    train_subj = subj_prep[:4000]
    test_subj = subj_prep[4000:5000]
    train_obj = obj_prep[:4000]
    test_obj = obj_prep[4000:5000]
    train_all = train_subj+train_obj
    test_all = test_subj+test_obj

    sentim_analyzer = SentimentAnalyzer()
    #all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in train_all])

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
    
    return X_train, y_train, X_test, y_test


def vectorize(X_train, X_test):
    subj_vectorizer = CountVectorizer()
    X_train_vectorized = subj_vectorizer.fit_transform(X_train)
    X_test_vectorized = subj_vectorizer.transform(X_test)
    
    return X_train_vectorized, X_test_vectorized, subj_vectorizer