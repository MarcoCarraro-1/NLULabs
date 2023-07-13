import nltk
nltk.download('treebank')
from nltk.corpus import treebank
from nltk.tag import NgramTagger
import spacy
import en_core_web_sm
from nltk.metrics import accuracy
nlp = en_core_web_sm.load()
from nltk.metrics import accuracy as accu
from itertools import chain

def create_dataset(train_perc):
    tagged_sentences = treebank.tagged_sents()
    train_set_size = int(len(tagged_sentences) / 100 * train_perc)
    test_set_size = int(len(tagged_sentences) - train_set_size)
    train_set = tagged_sentences[:train_set_size]
    test_set = tagged_sentences[train_set_size:]

    print("Dataset size: ", len(tagged_sentences))
    print("Train set size: ", len(train_set))
    print("Test set size: ", len(test_set))
    
    return train_set, test_set
    
    
def ngram_tagger_eval(n, cutoff, train_set, test_set):
    print("Ngram Tagger with n=", n, " and cutoff=", cutoff)
    tagger = NgramTagger(n, train=train_set, cutoff=cutoff)

    accuracy = tagger.accuracy(test_set)
    print("Accuracy:", accuracy)
    
    
def define_mapping():
    mapping_spacy_to_NLTK = {
        "NN": "NN",
        "NNS": "NNS",
        "NNP": "NNP",
        "NNPS": "NNPS",
        "VB": "VB",
        "VBD": "VBD",
        "VBG": "VBG",
        "VBN": "VBN",
        "VBP": "VBP",
        "VBZ": "VBZ",
        "JJ": "JJ",
        "JJR": "JJR",
        "JJS": "JJS",
        "RB": "RB",
        "RBR": "RBR",
        "RBS": "RBS",
        "IN": "IN",
        "DT": "DT",
        "PDT": "PDT",
        "CC": "CC",
        "CD": "CD",
        ".": ".",
        ",": ",",
        ":": ":",
        ";": ":",
        "\"": ".",
        "'": ".",
        "-LRB-": "-LRB-",
        "-RRB-": "-RRB-",
        "-LSB-": "-LRB-",
        "-RSB-": "-RRB-",
        "-LCB-": "-LRB-",
        "-RCB-": "-RRB-"
    }
    
    return mapping_spacy_to_NLTK    
    
def get_spacy_pos_tags(sentences):
    pos_tags = []
    for sentence in sentences:
        tokens = [token for token, _ in sentence]
        doc = nlp(" ".join(tokens))
        tags = [token.tag_ for token in doc]
        pos_tags.append(list(zip(tokens, tags)))
    return pos_tags

def map_spacy_tags_to_nltk(tags, mapping):
    mapped_tags = []
    for sentence_tags in tags:
        for token, tag in sentence_tags:
          if tag in mapping:
            mapped_tags.append((token, mapping[tag]))
          else:
            mapped_tags.append((token, 'X'))
    return mapped_tags

def calculate_accuracy(true_tags, predicted_tags):
    correct = 0
    total = len(true_tags)
    for true_tag, predicted_tag in zip(true_tags, predicted_tags):
        #print("true: ", true_tag, " - pred: ", predicted_tag)
        if true_tag == predicted_tag:
            correct += 1
    accuracy = correct / total
    return accuracy
    
    
def eval_accuracy(test_set, mapping):
    spacy_pos_tags = get_spacy_pos_tags(test_set)
    nltk_pos_tags = map_spacy_tags_to_nltk(spacy_pos_tags, mapping)

    spacy_pos_tags = list(chain.from_iterable(spacy_pos_tags)) # Flatten the list of lists 

    true_tags = [tag for sentence in test_set for _, tag in sentence]

    spacy_predicted_tags = [tag for _, tag in spacy_pos_tags]
    nltk_predicted_tags = [tag for _, tag in nltk_pos_tags]

    spacy_accuracy = calculate_accuracy(true_tags, spacy_predicted_tags)
    print("Spacy Accuracy:", spacy_accuracy)

    nltk_accuracy = accu(true_tags, nltk_predicted_tags)
    print("NLTK Accuracy:", nltk_accuracy)
    
    return spacy_accuracy, nltk_accuracy