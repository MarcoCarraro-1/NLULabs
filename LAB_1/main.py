from functions import *

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm",  disable=["tagger", "ner"])
    
    milton_chars, milton_words, milton_sents = load_data('milton-paradise.txt')
    doc = nlp(milton_chars)
    
    print("--- REFERENCE ---")
    word_per_sent, char_per_word, char_per_sent, longest_sent, longest_word = desc_stat(milton_words, milton_sents)
    
    words_spacy, sents_spacy = process_spacy(milton_chars, nlp)
    print("--- SPACY ---")
    word_per_sent_spacy, char_per_word_spacy, char_per_sent_spacy, longest_sent_spacy, longest_word_spacy = desc_stat(words_spacy, sents_spacy)
    
    words_nltk, sents_nltk = process_nltk(milton_chars)
    print("--- NLTK ---")
    word_per_sent_nltk, char_per_word_nltk, char_per_sent_nltk, longest_sent_nltk, longest_word_nltk = desc_stat(words_nltk, sents_nltk)

    print("\nLOWERCASE LEXICON")
    lowercased_lexicon(milton_words, doc)
    
    print("\nFREQUENCY DISTRIBUTION")
    n = 5
    show_freq_dist(milton_words, doc, n)