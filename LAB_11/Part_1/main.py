from functions import *
from model import *
from utils import *

if __name__ == "__main__":
    subj_docs, obj_docs = get_data('subj', 'obj')
    subj_prep, obj_prep, all_prep = get_prep_sents(subj_docs, obj_docs)
    
    X_train, y_train, X_test, y_test = split_data(subj_prep, obj_prep)
    X_train_vectorized, X_test_vectorized, subj_vectorizer = vectorize(X_train, X_test)
    
    acc_no_kfold, model = eval_acc1(X_train_vectorized, X_test_vectorized, y_train, y_test)
    #acc_all_sents = eval_acc2(X_train, X_test, y_train, y_test)
    
    movie_sentences, polarity_all = get_movie_data()
    
    movie_prep = get_prep_movie(movie_sentences)
    movie_train, movie_test, polarity_train, polarity_test = split_movie_data(movie_prep)
    x_movie, y_movie = get_movie_x_y(movie_train, polarity_train, movie_test, polarity_test)
    
    acc_all_polarity, model_polarity, skf_polarity = eval_acc3(x_movie, y_movie)
    
    movie_no_obj, polarity_no_obj = remove_obj_sents(movie_train, movie_test, subj_vectorizer, 
                                                     model)
    
    acc_no_obj_sents = eval_acc4(skf_polarity, movie_no_obj, polarity_no_obj, model_polarity)