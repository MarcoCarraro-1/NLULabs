from functions import *

if __name__ == "__main__":
    data = load_dataset()

    n_splits = 5
    X_train, X_test, y_train, y_test, stratified_split  = split_data(data, n_splits)
    
    print("COUNT VECTORIZER")
    X_tr1, X_te1 = vectorize_count(X_train, X_test)
    f1_count_vec = eval_f1(X_tr1, X_te1, y_train, y_test, stratified_split)
    
    print("MIN AND MAX CUTOFF")
    X_tr2, X_te2 = vectorize_cutoff(X_train, X_test)
    f1_cutoff = eval_f1(X_tr2, X_te2, y_train, y_test, stratified_split)
    
    print("WITHOUT STOPWORDS")
    X_tr3, X_te3 = vectorize_stop_words(X_train, X_test)
    f1_stop_words = eval_f1(X_tr3, X_te3, y_train, y_test, stratified_split)
    
    print("NO LOWERCASE")
    X_tr4, X_te4 = vectorize_no_lower(X_train, X_test)
    f1_no_lower = eval_f1(X_tr4, X_te4, y_train, y_test, stratified_split)