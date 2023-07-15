import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('senseval')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('universal_tagset')
from nltk.corpus import senseval
inst = senseval.instances('interest.pos')[0]
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
from nltk.metrics.scores import precision, recall, f_measure, accuracy
from collections import Counter
from tqdm import tqdm
    

def collocational_features(inst, n=2):
    context_words = [token[0] for token in inst.context]
    context_pos = [token[1] for token in inst.context]
    ngrams_within_window = list(ngrams(context_words, n))
    ngram_features = {}
    p = inst.position
    
    for i, ngram in enumerate(ngrams_within_window):
        feature_name = f"ngram_{i}"
        ngram_features[feature_name] = " ".join(ngram)
    
    features = {
        "w-2_word": 'NULL' if p < 2 else inst.context[p-2][0],
        "w-2_pos": 'NULL' if p < 2 else inst.context[p-2][1],
        "w-1_word": 'NULL' if p < 1 else inst.context[p-1][0],
        "w-1_pos": 'NULL' if p < 1 else inst.context[p-1][1],
        "w+1_word": 'NULL' if len(inst.context) - 1 < p+1 else inst.context[p+1][0],
        "w+1_pos": 'NULL' if len(inst.context) - 1 < p+1 else inst.context[p+1][1],
        "w+2_word": 'NULL' if len(inst.context) - 1 < p+2 else inst.context[p+2][0],
        "w+2_pos": 'NULL' if len(inst.context) - 1 < p+2 else inst.context[p+2][1]
    }
    features.update(ngram_features)
    
    return features


def extract_data():
    data_col = [collocational_features(inst) for inst in senseval.instances('interest.pos')]
    lbls = [inst.senses[0] for inst in senseval.instances('interest.pos')]
        
    return data_col, lbls


def extract_bow_data():
    bow_data = [" ".join([t[0] for t in inst.context]) for inst in senseval.instances('interest.pos')]
    bow_lbls = [inst.senses[0] for inst in senseval.instances('interest.pos')]
    
    return bow_data, bow_lbls


def eval_score(data_col, lbls):
    dvectorizer = DictVectorizer(sparse=False)
    dvectors = dvectorizer.fit_transform(data_col)
    classifier = MultinomialNB()
    lblencoder = LabelEncoder()

    lblencoder.fit(lbls)
    labels = lblencoder.transform(lbls)

    stratified_split = StratifiedKFold(n_splits=5, shuffle=True)
    scores = cross_validate(classifier, dvectors, labels, cv=stratified_split, scoring=['f1_micro'])

    print("Extend with N-grams:")
    res = sum(scores['test_f1_micro'])/len(scores['test_f1_micro'])
    print(res)
    
    return res


def bow_concat(bow_data, bow_lbls, data_col):
    dvectorizer = DictVectorizer(sparse=False)
    dvectors = dvectorizer.fit_transform(data_col)
    bow_vectorizer = CountVectorizer()
    bow_lblencoder = LabelEncoder()
    bow_vectors = bow_vectorizer.fit_transform(bow_data)

    bow_lblencoder.fit(bow_lbls)
    bow_labels = bow_lblencoder.transform(bow_lbls)    
    test_features = np.concatenate((bow_vectors.toarray(), dvectors), axis=1)
    
    return test_features, bow_labels

def eval_bow_score(test_features, bow_labels):
    bow_classifier = MultinomialNB()
    stratified_split = StratifiedKFold(n_splits=5, shuffle=True)
    scores = cross_validate(bow_classifier, test_features, bow_labels, cv=stratified_split, scoring=['f1_micro'])

    print("Concatenate BOW:")
    res = sum(scores['test_f1_micro'])/len(scores['test_f1_micro'])
    print(res)
    
    return res
    
    
def define_mapping():
    mapping = {
        'interest_1': 'interest.n.01',
        'interest_2': 'interest.n.03',
        'interest_3': 'pastime.n.01',
        'interest_4': 'sake.n.01',
        'interest_5': 'interest.n.05',
        'interest_6': 'interest.n.04',
    }
    
    return mapping


def eval_lesk(mapping, test_features, lesk_type):
    refs = {k: set() for k in mapping.values()}
    hyps = {k: set() for k in mapping.values()}
    refs_list = []
    hyps_list = []
    synsets = []
    
    for ss in tqdm(wordnet.synsets('interest', pos='n')):
        if ss.name() in mapping.values():
            defn = ss.definition()
            tags = preprocess(defn)
            toks = [l for w, l, p in tags]
            synsets.append((ss,toks))

    for i, inst in tqdm(enumerate(senseval.instances('interest.pos'))):
        txt = [t[0] for t in inst.context]
        raw_ref = inst.senses[0] # let's get first sense
        test_instance_features = test_features[i]

        if lesk_type=='original':
            hyp = original_lesk(txt, txt[inst.position], synsets=synsets, majority=True, features=test_instance_features).name()
        else:
            hyp = lesk_similarity(txt, txt[inst.position], synsets=synsets, majority=True).name()
        
        ref = mapping.get(raw_ref)
        refs[ref].add(i)
        hyps[hyp].add(i)
        refs_list.append(ref)
        hyps_list.append(hyp)

    if lesk_type=='original':
        print("Lesk original: ")
    else:
        print("Lesk similarity: ")
    
    print()
    acc = round(accuracy(refs_list, hyps_list), 3)
    print("Acc:", acc)

    for cls in hyps.keys():
        p = precision(refs[cls], hyps[cls])
        r = recall(refs[cls], hyps[cls])
        f = f_measure(refs[cls], hyps[cls], alpha=1)
        
        if(p is None):
            p= 0.0
        if(f is None):
            f = 0.0

        print("{:15s}: p={:.3f}; r={:.3f}; f={:.3f}; s={}".format(cls, p, r, f, len(refs[cls])))
    
    return acc


    
def preprocess(text):
    mapping = {"NOUN": wordnet.NOUN, "VERB": wordnet.VERB, "ADJ": wordnet.ADJ, "ADV": wordnet.ADV}
    sw_list = stopwords.words('english')
    lem = WordNetLemmatizer()

    tokens = nltk.word_tokenize(text) if type(text) is str else text
    tagged = nltk.pos_tag(tokens, tagset="universal")
    tagged = [(w.lower(), p) for w, p in tagged]
    tagged = [(w, p) for w, p in tagged if p in mapping]
    tagged = [(w, mapping.get(p, p)) for w, p in tagged]
    tagged = [(w, p) for w, p in tagged if w not in sw_list]
    tagged = [(w, lem.lemmatize(w, pos=p), p) for w, p in tagged]
    tagged = list(set(tagged))

    return tagged



def original_lesk(context_sentence, ambiguous_word, pos=None, synsets=None, majority=False, features=None):

    context_senses = get_sense_definitions(set(context_sentence)-set([ambiguous_word]))
    
    if synsets is None:
        synsets = get_sense_definitions(ambiguous_word)[0][1]

    if pos:
        synsets = [ss for ss in synsets if str(ss[0].pos()) == pos]

    if not synsets:
        return None
    
    scores = []
    
    for senses in context_senses:
        for sense in senses[1]:
            if features is not None:
                score = get_top_sense(sense[1] + features.tolist(), synsets)
            else:
                score = get_top_sense(sense[1], synsets)
            scores.append(score)

    if len(scores) == 0:
        return synsets[0][0]

    if majority:
        filtered_scores = [x[1] for x in scores if x[0] != 0]
        if len(filtered_scores) > 0:
            best_sense = Counter(filtered_scores).most_common(1)[0][0]
        else:
            best_sense = Counter([x[1] for x in scores]).most_common(1)[0][0]
    else:
        _, best_sense = max(scores)
    
    return best_sense


def get_sense_definitions(context):
    lemma_tags = preprocess(context)
    senses = [(w, wordnet.synsets(l, p)) for w, l, p in lemma_tags]
    definitions = []
    
    for raw_word, sense_list in senses:
        if len(sense_list) > 0:
            def_list = []
            for s in sense_list:
                defn = s.definition()
                tags = preprocess(defn)
                toks = [l for w, l, p in tags]
                def_list.append((s, toks))
            definitions.append((raw_word, def_list))
    
    return definitions


def get_top_sense(words, sense_list):
    val, sense = max((len(set(words).intersection(set(defn))), ss) for ss, defn in sense_list)
    
    return val, sense



def lesk_similarity(context_sentence, ambiguous_word, similarity="resnik", pos=None,
                    synsets=None, majority=True):
    context_senses = get_sense_definitions(set(context_sentence) - set([ambiguous_word]))

    if synsets is None:
        synsets = get_sense_definitions(ambiguous_word)[0][1]

    if pos:
        synsets = [ss for ss in synsets if str(ss[0].pos()) == pos]

    if not synsets:
        return None

    scores = []

    for senses in context_senses:
        for sense in senses[1]:
            scores.append(get_top_sense_sim(sense[0], synsets, similarity))

    if len(scores) == 0:
        return synsets[0][0]

    if majority:
        filtered_scores = [x[1] for x in scores if x[0] != 0]
        if len(filtered_scores) > 0:
            best_sense = Counter(filtered_scores).most_common(1)[0][0]
        else:
            best_sense = Counter([x[1] for x in scores]).most_common(1)[0][0]
    else:
        _, best_sense = max(scores)

    return best_sense


def get_top_sense_sim(context_sense, sense_list, similarity):
    scores = []
    for sense in sense_list:
        ss = sense[0]
        if similarity == "path":
            try:
                scores.append((context_sense.path_similarity(ss), ss))
            except:
                scores.append((0, ss))
        elif similarity == "lch":
            try:
                scores.append((context_sense.lch_similarity(ss), ss))
            except:
                scores.append((0, ss))
        elif similarity == "wup":
            try:
                scores.append((context_sense.wup_similarity(ss), ss))
            except:
                scores.append((0, ss))
        elif similarity == "resnik":
            try:
                scores.append((context_sense.res_similarity(ss, semcor_ic), ss))
            except:
                scores.append((0, ss))
        elif similarity == "lin":
            try:
                scores.append((context_sense.lin_similarity(ss, semcor_ic), ss))
            except:
                scores.append((0, ss))
        elif similarity == "jiang":
            try:
                scores.append((context_sense.jcn_similarity(ss, semcor_ic), ss))
            except:
                scores.append((0, ss))
        else:
            print("Similarity metric not found")
            return None
    
    val, sense = max(scores)
    
    return val, sense