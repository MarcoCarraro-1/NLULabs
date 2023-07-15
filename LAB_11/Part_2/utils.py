from functions import *
import os
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neural_network import MLPClassifier


def load_data(train_fold, test_fold):
    file_path_train = os.path.join(os.path.dirname(__file__), train_fold)
    with open(file_path_train, 'r') as file1:
        train_data = file1.read()
    
    file_path_test = os.path.join(os.path.dirname(__file__), test_fold)
    with open(file_path_test, 'r', encoding='utf-8') as file2:
        test_data = file2.read()
    
    test_data = test_data.split('\n')
    train_data = train_data.split('\n')
    
    print("Length test data: ", len(test_data))
    print("Length train data: ", len(train_data))
    
    return train_data, test_data


def create_dict(train_data):
    aspect_terms = {}
    
    for sentence in train_data:
        words_tags = sentence.split("####")[-1].split()
        for word_tag in words_tags:
            try:
                word, tag = word_tag.split("=")
                if tag.startswith("T-POS"):
                    aspect_terms[word] = aspect_terms.get(word, 0) + 1
                elif tag.startswith("T-NEG"):
                    aspect_terms[word] = aspect_terms.get(word, 0) - 1
            except:
                print("Wrong Format")
                print(word_tag)
                
    print("Dictionary length: ", len(aspect_terms))
    
    return aspect_terms


def tokenize_sents(train_data, test_data, aspect_terms):
    train_data_temp = []
    train_data_no_tags = []
    aspect_labels = []
    test_labels = []
    test_data_temp = []
    test_data_no_tags = []


    for sentence in tqdm(train_data):
        no_tags = sentence.split("####")[0].split()
        train_data_temp.append(no_tags)

    for sentence in tqdm(train_data_temp):
        sent = ' '.join(sentence)
        train_data_no_tags.append(sent)

    tokenized_train_data = [word_tokenize(sentence) for sentence in train_data_no_tags]
    
    for sentence in tqdm(test_data):
        no_tags = sentence.split("####")[0].split()
        test_data_temp.append(no_tags)

    for sentence in tqdm(test_data_temp):
        sent = ' '.join(sentence)
        test_data_no_tags.append(sent)

    tokenized_test_data = [word_tokenize(sentence) for sentence in test_data_no_tags]
    
    for sentence in tqdm(tokenized_train_data):
        labels = []
        for word in sentence:
            if word in aspect_terms:
                labels.append(1)
            else:
                labels.append(0)
        aspect_labels.append(labels)
        
    for sentence in tqdm(tokenized_test_data):
        labels = []
        for word in sentence:
            if word in aspect_terms:
                labels.append(1)
            else:
                labels.append(0)
        test_labels.append(labels)
        
    return tokenized_train_data, tokenized_test_data, aspect_labels, test_labels


def extract_term_model(tokenized_train_data, aspect_labels):    
    #aspect_labels = np.array(aspect_labels)
    vectorizer = CountVectorizer()

    train_tokenized_vec = vectorizer.fit_transform([' '.join(tokens) for tokens in tokenized_train_data])
    train_tokenized_array = train_tokenized_vec.toarray()

    mlb = MultiLabelBinarizer()
    aspect_labels_bin = mlb.fit_transform(aspect_labels)

    model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam')
    model.fit(train_tokenized_array, aspect_labels_bin)

    torch.save(model.state_dict(), 'term_extraction_model.bin')
    
    return train_tokenized_array, aspect_labels_bin, model, vectorizer, mlb
