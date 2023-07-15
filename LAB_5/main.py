from functions import *

if __name__ == "__main__":
    sent1 = "Inter won the CoppaItalia final against Fiorentina"
    sent2 = "Last night the concert lasted two hours"
    n_sents = 10
    
    my_grammar, weighted_rules = define_grammar()
    
    viterbi_parse(my_grammar, sent1, sent2)
    
    inside_parse(my_grammar, sent1, sent2)
    
    random_parse(my_grammar, sent1, sent2)
    
    nltk_sents = generate_sents(weighted_rules, n_sents)
    
    pcfg_sents = generate_sents(weighted_rules, n_sents, grammar=my_grammar)