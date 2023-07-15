import spacy
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
import nltk
from nltk.corpus import conll2002
nltk.download('conll2002')
import es_core_news_sm
from spacy.tokenizer import Tokenizer
from conll import evaluate
import pandas as pd
from tqdm import tqdm


nlp = es_core_news_sm.load()
nlp.tokenizer = Tokenizer(nlp.vocab)

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, label in sent]


def sent2tokens(sent):
    return [token for token, label in sent]


def word2features(sent, i):
    word = sent[i][0]
    return {'bias': 1.0, 'word.lower()': word.lower()}


def load_data():
    train_data = [[(text, iob) for text, pos, iob in sent] for sent in conll2002.iob_sents('esp.train')]
    test_data = [[(text, iob) for text, pos, iob in sent] for sent in conll2002.iob_sents('esp.testa')]
    
    return train_data, test_data


def define_crf():
    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    
    return crf


def sent2spacy_baseline(sent):
    spacy_sent = nlp(" ".join(sent2tokens(sent)))
    feats = []
    
    for token in spacy_sent:
        token_feats = {
            'bias': 1.0,
            'word.lower()': token.lower_,
            'pos': token.pos_,
            'lemma': token.lemma_
        }
        feats.append(token_feats)
        
        
    return feats


def sent2spacy_suffix(sent):
    spacy_sent = nlp(" ".join(sent2tokens(sent)))
    feats = []
    
    for token in spacy_sent:
        token_feats = {
            'bias': 1.0,
            'word.lower()': token.text.lower(),
            'pos': token.pos_,
            'lemma': token.lemma_,
            'suffix': token.text[-3:]
        }
        feats.append(token_feats)
        
    return feats


def sent2spacy_tutorial(sent):
    sent_tokens = [token for token, _ in sent]
    sent_text = " ".join(sent_tokens)
    spacy_sent = nlp(sent_text)
    feats = []
    
    for token in spacy_sent:
        token_feats = {
            'bias': 1.0,
            'word.lower()': token.text.lower(),
            'word[-3:]': token.text[-3:],
            'word[-2:]': token.text[-2:],
            'word.isupper()': token.text.isupper(),
            'word.istitle()': token.text.istitle(),
            'word.isdigit()': token.text.isdigit(),
            'pos': token.pos_,
            'lemma': token.lemma_,
            'suffix': token.text[-3:]
        }
        feats.append(token_feats)
        
    return feats


def sent2spacy_window(sent, low, high):
    sent_tokens = [token for token, _ in sent]  # Estrai solo i token dalla lista 'sent'
    sent_text = " ".join(sent_tokens)  # Unisci i token in una stringa separata da spazi
    spacy_sent = nlp(sent_text)  # Esegui l'analisi spacy sulla stringa di token
    feats = []
    
    for i, token in enumerate(spacy_sent):
        token_feats = {
            'bias': 1.0,
            'word.lower()': token.text.lower(),
            'pos': token.pos_,
            'lemma': token.lemma_
        }
        for offset in range(low, high):
            if i + offset >= 0 and i + offset < len(spacy_sent):
                prev_token = spacy_sent[i + offset]
                token_feats[f'word.lower({offset})'] = prev_token.lower_
                token_feats[f'pos({offset})'] = prev_token.pos_
                token_feats[f'lemma({offset})'] = prev_token.lemma_
        feats.append(token_feats)
    
    return feats


def set_data(train_data, test_data, feature):
    if feature == 'baseline':
        trn_feats = [sent2spacy_baseline(s) for s in tqdm(train_data)]
        trn_label = [sent2labels(s) for s in tqdm(train_data)]
        tst_feats = [sent2spacy_baseline(s) for s in tqdm(test_data)]
    elif feature == 'suffix':
        trn_feats = [sent2spacy_suffix(s) for s in tqdm(train_data)]
        trn_label = [sent2labels(s) for s in tqdm(train_data)]
        tst_feats = [sent2spacy_suffix(s) for s in tqdm(test_data)]
    elif feature == 'tutorial':
        trn_feats = [sent2spacy_tutorial(s) for s in tqdm(train_data)]
        trn_label = [sent2labels(s) for s in tqdm(train_data)]
        tst_feats = [sent2spacy_tutorial(s) for s in tqdm(test_data)]
    elif feature == 'window1':
        trn_feats = [sent2spacy_window(s, -1, 2) for s in tqdm(train_data)]
        trn_label = [sent2labels(s) for s in tqdm(train_data)]
        tst_feats = [sent2spacy_window(s, -1, 2) for s in tqdm(test_data)]
    elif feature == 'window2':
        trn_feats = [sent2spacy_window(s, -2, 3) for s in tqdm(train_data)]
        trn_label = [sent2labels(s) for s in tqdm(train_data)]
        tst_feats = [sent2spacy_window(s, -2, 3) for s in tqdm(test_data)]
    
    return trn_feats, trn_label, tst_feats


def train_and_predict(crf, trn_feats, trn_label, tst_feats):
    try:
        crf.fit(trn_feats, trn_label)
    except AttributeError:
        pass

    pred = crf.predict(tst_feats)
    
    return pred


def show_result(tst_feats, pred, test_data):
    hyp = [[(tst_feats[i][j], t) for j, t in enumerate(tokens)] for i, tokens in enumerate(pred)]
    results = evaluate(test_data, hyp)

    pd_tbl = pd.DataFrame().from_dict(results, orient='index')
    pd_tbl.round(decimals=3)
    
    print(pd_tbl)
    print()