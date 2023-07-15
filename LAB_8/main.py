from functions import *

if __name__ == "__main__":    
    data_col, lbls = extract_data()
    extended_score = eval_score(data_col, lbls)
    
    bow_data, bow_lbls = extract_bow_data()
    test_features, bow_labels = bow_concat(bow_data, bow_lbls, data_col)
    bow_score = eval_bow_score(test_features, bow_labels)
    
    mapping = define_mapping()
    
    acc_original = eval_lesk(mapping, test_features, 'original')
    
    acc_similarity = eval_lesk(mapping, test_features, 'similarity')