from functions import *

if __name__ == "__main__":
    
    train_perc = 80
    
    train_set, test_set = create_dataset(train_perc)
    
    ngram_tagger_eval(2, 2, train_set, test_set)
    ngram_tagger_eval(4, 1, train_set, test_set)
    ngram_tagger_eval(2, 1, train_set, test_set)
    ngram_tagger_eval(1, 1, train_set, test_set)
    ngram_tagger_eval(1, 5, train_set, test_set)
    ngram_tagger_eval(1, 4, train_set, test_set)
    ngram_tagger_eval(1, 3, train_set, test_set)
    ngram_tagger_eval(1, 2, train_set, test_set)
    
    mapping = define_mapping()
    
    spacy_acc, nltk_acc = eval_accuracy(test_set, mapping)