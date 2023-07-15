from functions import *

if __name__ == "__main__":
    
    order = 2
    alpha = 0.4
    
    macbeth_oov_sents, lex, padded_ngrams_oov, flat_text_oov = tokenization_voc(
                                                        'shakespeare-macbeth.txt', order)
    
    nltk_model, train_data, test_data = train_backoff(order, alpha, lex, padded_ngrams_oov, 
                                                      macbeth_oov_sents)
    
    nltk_perplexity = eval_nltk_ppl(lex, test_data, nltk_model)
    
    personal_model = train_personal(order, alpha, lex, macbeth_oov_sents)
    
    personal_perplexity = eval_personal_ppl(lex, test_data, personal_model)