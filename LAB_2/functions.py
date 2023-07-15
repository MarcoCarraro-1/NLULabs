from sklearn.datasets import fetch_20newsgroups
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from tqdm import tqdm

def load_dataset():
    data = fetch_20newsgroups()
    
    return data
    
    
def split_data(data, n_splits):
    random_split = KFold(n_splits= n_splits, shuffle=True)
    split_train = []
    split_test = []
    
    for train_index, test_index in tqdm(random_split.split(data.data)):
        split_train.append([ v for _, v in sorted(Counter(list(data.target[train_index])).items())])
        split_test.append([ v for _, v in sorted(Counter(list(data.target[test_index])).items())])
    
    plot_bars(split_train, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], title='Random split Train')
    plot_bars(split_test, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], title='Random split Test')
    
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
    stratified_split = StratifiedShuffleSplit(n_splits=4, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test, stratified_split
    
    
def plot_bars(values, labels, width=0.35, title=""):
    x = np.arange(len(labels))  # the label locations
    fig, ax = plt.subplots(figsize=(12,5))
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Classes')
    ax.set_title(title)
    ax.set_xticks(x, labels)

    center = round(len(values)/2)
    for id_x, temp in enumerate(values):
        new_x = x + width/len(values) * (id_x-center)
        lab = 'split'+str(id_x+1)
        ax.bar(new_x, temp, width/len(values), label=lab)

    ax.legend(loc='lower right')
    plt.show()
    
    
def vectorize_count(X_train, X_test):
    vectorizer = CountVectorizer(binary=True)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    
    return X_train, X_test
    
    
def vectorize_cutoff(X_train, X_test):
    vectorizer = CountVectorizer(max_df=2000, min_df=5, lowercase=True)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    
    return X_train, X_test
    
    
def vectorize_stop_words(X_train, X_test):
    vectorizer = CountVectorizer(max_df=2000, min_df=5, lowercase=True, stop_words='english')
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    
    return X_train, X_test
    
    
def vectorize_no_lower(X_train, X_test):
    vectorizer = CountVectorizer(max_df=2000, min_df=5, lowercase=False, stop_words='english')
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    
    return X_train, X_test
    
    
def eval_f1(X_train, X_test, y_train, y_test, stratified_split):
    clf = LinearSVC(C=0.02, max_iter=10000, dual=True)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    refs = y_test
    
    f1 = f1_score(refs, y_pred, average='macro')
    print(f'Macro avg F1-score: {f1:.3f}\n')
    
    return f1