import spacy
import nltk
nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter


def load_data(text):
    milton_chars = nltk.corpus.gutenberg.raw(text)
    milton_words = nltk.corpus.gutenberg.words(text)
    milton_sents = nltk.corpus.gutenberg.sents(text)
    
    return milton_chars, milton_words, milton_sents


def desc_stat(words, sents):
    chars_num = [len(word) for word in words]
    chars_tot = sum(chars_num)
    words_tot = len(words)
    chars_AVG = round(chars_tot / words_tot , 2)
    sents_tot = len(sents)
    words_AVG = round(words_tot / sents_tot , 2)
    chars_in_sent_AVG = round(chars_tot / sents_tot , 2)
    max_word = max(chars_num)
    min_word = min(chars_num)
    max_sentence = max([len(sent) for sent in sents])
    min_sentence = min([len(sent) for sent in sents])

    print("Average of chars per word: {:.2f}".format(chars_AVG))
    print("Average of words per sentence: {:.2f}".format(words_AVG))
    print("Average of characters per sentence: {:.2f}".format(chars_in_sent_AVG))
    print("Length of longest word: {:.2f}".format(max_word))
    print("Length of shortest word: {:.2f}".format(min_word))
    print("Length of longest sentence: {:.2f}".format(max_sentence))
    print("Length of shortest sentence: {:.2f}".format(min_sentence))

    return words_AVG, chars_AVG, chars_in_sent_AVG, max_sentence, max_word


def process_spacy(milton_chars, nlp):
    processed_text = nlp(milton_chars)
    words_spacy = [token.text for token in processed_text if not token.is_punct and not token.is_space and not token.is_left_punct]
    sents_spacy = list(processed_text.sents)
    
    return words_spacy, sents_spacy


def process_nltk(milton_chars):
    words_nltk = word_tokenize(milton_chars)
    sents_nltk = sent_tokenize(milton_chars)
    
    return words_nltk, sents_nltk


def lowercased_lexicon(milton_words, doc):
    print("--- REFERENCE ---")
    milton_words_lowercase = set([w.lower() for w in milton_words])
    milton_lexicon_lowercase = set(milton_words_lowercase)
    print(len(milton_lexicon_lowercase))

    print("--- SPACY ---")
    spacy_lexicon_lowercase = set([token.lower_ for token in doc])
    spacy_lexicon_length = len(spacy_lexicon_lowercase)
    print( spacy_lexicon_length)

    print("--- NLTK ---")
    nltk_lexicon_lowercase = set([word.lower() for word in milton_words])
    nltk_lexicon_length = len(nltk_lexicon_lowercase)
    print(nltk_lexicon_length)
    
    
def show_freq_dist(milton_words, doc, n):
    milton_words_lowercase = set([w.lower() for w in milton_words])
    milton_lexicon = set(milton_words)
    milton_lexicon_lowercase = set(milton_words_lowercase)
    milton_lowercase_freq_list = Counter(milton_words)

    #SPACY top N frequency
    spacy_freq_list = Counter([token.text.lower() for token in doc])
    spacy_most_common_no_cut = nbest(spacy_freq_list, n)
    print("[SPACY] Most common words: ", spacy_most_common_no_cut)
    print_cut_off(0, float('inf'), spacy_freq_list)
    print_cut_off(2, float('inf'), spacy_freq_list)
    print_cut_off(0, 100, spacy_freq_list)
    print_cut_off(2, 100, spacy_freq_list)

    #NLTK top N frequency
    nltk_most_common_no_cut = nbest(milton_lowercase_freq_list, n)
    print("\n[NLTK] Most common words: ", nltk_most_common_no_cut)
    print_cut_off(0, float('inf'), milton_lowercase_freq_list)
    print_cut_off(2, float('inf'), milton_lowercase_freq_list)
    print_cut_off(0, 100, milton_lowercase_freq_list)
    print_cut_off(2, 100, milton_lowercase_freq_list)
    
    
def nbest(d, n=5):
  return dict(sorted(d.items(), key=lambda item: item[1], reverse=True)[:n])


def cut_off(vocab, n_min=100, n_max=100):
    new_vocab = []
    for word, count in vocab.items():
        if count >= n_min and count <= n_max:
            new_vocab.append(word)
    return new_vocab


def print_cut_off(lower_bound, upper_bound, freq_list):
  lexicon_cut_off = len(cut_off(freq_list, n_min=lower_bound, n_max=upper_bound))
  print('CutOFF Min:', lower_bound, 'MAX:', upper_bound, '-----> Lexicon Size:', lexicon_cut_off)