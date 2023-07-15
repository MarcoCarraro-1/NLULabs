from functions import *
from model import *
from utils import *

if __name__ == "__main__":
    train_data, test_data = load_data('dataset/laptop14_train.txt', 'dataset/laptop14_test.txt')
    aspect_terms = create_dict(train_data)
    
    tokenized_train_data, tokenized_test_data, aspect_labels, test_labels = tokenize_sents(
                                                        train_data, test_data, aspect_terms)
    
    tokenized_train_array, aspect_labels_bin, model, vectorizer, mlb = extract_term_model(
                                                        tokenized_train_data, aspect_labels)
    
    model_acc = eval_acc(tokenized_test_data, model, mlb, test_labels, vectorizer)