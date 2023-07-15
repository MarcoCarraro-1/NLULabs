from pcfg import PCFG
import nltk
from tkinter.constants import N
from nltk.parse.generate import generate


def define_grammar():
    weighted_rules = [
        'S    -> NP VP [0.5]',              
        'S    -> TP NP [0.5]',              
        'NP   -> Det N N PP [0.25]',        
        'NP   -> Det N VP [0.25]',          
        'NP   -> PRON [0.25]',              
        'NP   -> Num N [0.25]',             
        'VP   -> V NP [1.0]',               
        'PP   -> P N [1.0]',                
        'TP   -> T N [1.0]',                
        'Det  -> "the" [1.0]',              
        'Num  -> "two" [1.0]',              
        'N    -> "CoppaItalia" [0.2]',      
        'N    -> "final" [0.2]',            
        'N    -> "Fiorentina" [0.2]',       
        'N    -> "night" [0.1]',            
        'N    -> "concert" [0.2]',          
        'N    -> "hours" [0.1]',            
        'PRON -> "Inter" [1.0]',            
        'V    -> "won" [0.5]',              
        'V    -> "lasted" [0.5]',           
        'P    -> "against" [1.0]',          
        'T    -> "Last" [1.0]'              
    ]

    my_grammar = nltk.PCFG.fromstring(weighted_rules)
    print(my_grammar)
    
    return my_grammar, weighted_rules


def print_tree(sent1, sent2, parser):
    for tree in parser.parse(sent1.split()):
        print(tree)

    for tree in parser.parse(sent2.split()):
        print(tree)
    
    
def viterbi_parse(my_grammar, sent1, sent2):
    print("\nVITERBI PARSER")
    parser = nltk.ViterbiParser(my_grammar)
    print_tree(sent1, sent2, parser)
    
    
def inside_parse(my_grammar, sent1, sent2):
    print("\nINSIDECHART PARSER")
    parser = nltk.InsideChartParser(my_grammar, beam_size=100)
    print_tree(sent1, sent2, parser)
    
    
def random_parse(my_grammar, sent1, sent2):
    print("\nRANDOMCHART PARSER")
    parser = nltk.RandomChartParser(my_grammar, beam_size=100)
    print_tree(sent1, sent2, parser)
    
    
def generate_sents(weighted_rules, n_sents, grammar=None):
    if grammar==None:
        grammar = PCFG.fromstring(weighted_rules)
        print("\n---- FIRST 10 SENTENCES: ----")
        for sent in grammar.generate(n_sents):
            print(sent)
    else:
        print("\n---- OPTIONAL 10 SENTENCES: ----")
        for sent in generate(grammar, n=n_sents):
            print(" ".join(sent))