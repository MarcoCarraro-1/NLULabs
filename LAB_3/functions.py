import nltk
nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.lm import StupidBackoff
from nltk.corpus import gutenberg
from nltk.lm.preprocessing import flatten
from nltk.lm import Vocabulary
from nltk.lm.preprocessing import padded_everygram_pipeline
from itertools import chain
from nltk.lm import NgramCounter
import math


def tokenization_voc(text, order):
    macbeth_sents = [[w.lower() for w in sent] for sent in gutenberg.sents(text)]
    macbeth_words = flatten(macbeth_sents)

    lex = Vocabulary(macbeth_words, unk_cutoff=2)

    macbeth_oov_sents = [list(lex.lookup(sent)) for sent in macbeth_sents]
    padded_ngrams_oov, flat_text_oov = padded_everygram_pipeline(order, macbeth_oov_sents)
    
    return macbeth_oov_sents, lex, padded_ngrams_oov, flat_text_oov


def train_backoff(order, alpha, lex, padded_ngrams_oov, macbeth_oov_sents):
    train_data = macbeth_oov_sents[:int(0.8 * len(macbeth_oov_sents))]
    test_data = macbeth_oov_sents[int(0.8 * len(macbeth_oov_sents)):]

    model = StupidBackoff(order=order, alpha=alpha)
    model.fit(padded_ngrams_oov, vocabulary_text=lex)
    
    return model, train_data, test_data


def eval_nltk_ppl(lex, test_data, model):
    ngrams, flat_text = padded_everygram_pipeline(model.order, [lex.lookup(sent) for sent in test_data])
    ngrams = chain.from_iterable(ngrams)
    ppl =  model.perplexity([x for x in ngrams if len(x) == model.order])
    print('PPL Nltk:', ppl)

    ngrams, flat_text = padded_everygram_pipeline(model.order, [lex.lookup(sent) for sent in test_data])
    ngrams = chain.from_iterable(ngrams)
    cross_entropy =  model.entropy([x for x in ngrams if len(x) == model.order])
    print('\t PPL from Cross Entropy:', pow(2, cross_entropy))
    
    return ppl


class personal_StupidBackoffLM:
    def __init__(self, order, alpha):
        self.order = order
        self.alpha = alpha
        self.counter = None

    def fit(self, padded_ngrams):
      self.counter = NgramCounter(padded_ngrams)

    def personal_backoff(self, ngram):
      epsilon = 1e-10
      if len(ngram) == 1:
        return self.counter[ngram[0]] / sum(self.counter.unigrams.values())
      else:
        context = ngram[:-1]
        word = ngram[-1]
        if self.counter[context][word] > 0:
          counter_curr_ngram = self.counter[context][word]
          if len(context) > 1:
            counter_lower_ngram = self.counter[context[:-1]][context[-1]]
          else:
            counter_lower_ngram = self.counter[context[0]]
          return counter_curr_ngram / counter_lower_ngram
        else:
          return self.alpha * self.personal_backoff(ngram[1:]) + epsilon

    def calculate_perplexity(self, ngrams):
      return math.pow(2.0, self.entropy(ngrams))

    def entropy(self, ngrams):
      return -1 * sum([self.logscore(ngram[-1], ngram[:-1]) for ngram in ngrams]) / len(ngrams)

    def logscore(self, word, context):
      ngram = tuple(context[-self.order+1:] + tuple([word]))
      return math.log(self.personal_backoff(ngram), 2)
 
  

def train_personal(order, alpha, lex, macbeth_oov_sents):
    padded_ngrams, flat_text = padded_everygram_pipeline(order, [lex.lookup(sent) for sent in macbeth_oov_sents])
    model = personal_StupidBackoffLM(order=order, alpha=alpha)
    model.fit(padded_ngrams)
    
    return model


def eval_personal_ppl(lex, test_data, model):
    ngrams, flat_text = padded_everygram_pipeline(model.order, [lex.lookup(sent) for sent in test_data])
    ngrams = flatten(ngrams)
    
    ppl = model.calculate_perplexity([x for x in ngrams if len(x) == model.order])
    print('PPL MyStupidBackoff:', ppl)
    
    return ppl