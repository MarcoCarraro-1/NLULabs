from functions import *

if __name__ == "__main__":
    
    last_100_sents, spacy_dep_tags, stanza_dep_tags = extract_sents()
    
    compare_dep_tags(spacy_dep_tags, stanza_dep_tags)
    
    dp_graph_spacy = conv_dep_graph_spacy(last_100_sents)
    
    try:
        dp_graph_stanza = conv_dep_graph_stanza(last_100_sents)
    except:
        print("Error in converting in DependencyGraph with Stanza")