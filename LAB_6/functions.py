import spacy
from spacy.tokens import Token
from spacy.tokens import Doc
from spacy.pipeline import Pipe
import stanza
import nltk
stanza.download("en")
nltk.download('dependency_treebank')
from nltk.corpus import dependency_treebank
from nltk.parse.dependencygraph import DependencyGraph
from spacy.tokenizer import Tokenizer
from stanza.utils.conll import CoNLL
from nltk.parse import DependencyEvaluator
from tqdm import tqdm


nlp_spacy = spacy.load("en_core_web_sm")
nlp_stanza = stanza.Pipeline("en")


def extract_sents():
    treebank_sentences = dependency_treebank.sents()
    last_100_sentences = treebank_sentences[-100:]
    spacy_dep_tags = []
    stanza_dep_tags = []

    for sentence in tqdm(last_100_sentences):
        spacy_doc = nlp_spacy(" ".join(sentence))
        for token in spacy_doc:
            spacy_dep_tags.append(token.dep_)

        stanza_doc = nlp_stanza(" ".join(sentence))
        for sent in stanza_doc.sentences:
            for word in sent.words:
                stanza_dep_tags.append(word.deprel)
                
    return last_100_sentences, spacy_dep_tags, stanza_dep_tags


def compare_dep_tags(spacy_dep_tags, stanza_dep_tags):
    same = 0
    diff = 0

    for i in range(0, len(spacy_dep_tags)-1):
        if(spacy_dep_tags[i] == stanza_dep_tags[i]):
            same += 1
        else:
            diff += 1
    
    print()
    print("Same tags: ", same)
    print("Different tags: ", diff)

    
def conv_dep_graph_spacy(last_100_sents):
    config = {"ext_names": {"conll_pd": "pandas"},
            "conversion_maps": {"deprel": {"nsubj": "subj"}}}
    all_graph_spacy = []

    for sentence in tqdm(last_100_sents):
        try:
            nlp_spacy.add_pipe("conll_formatter", config=config, last=True)
        except:
            nlp_spacy.tokenizer = Tokenizer(nlp_spacy.vocab)
            spacy_doc = nlp_spacy(" ".join(sentence))
            conll_output_spacy = spacy_doc._.conll_str
            dp_graph_spacy = DependencyGraph(conll_output_spacy)
            all_graph_spacy.append(dp_graph_spacy)

    return all_graph_spacy