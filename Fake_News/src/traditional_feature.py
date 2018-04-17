import nltk as nlp
import numpy as np
import re
from sklearn import feature_extraction
from nltk.stem.snowball import SnowballStemmer
from enum import Enum
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os


def normalize_word(word):
    _wnl = nlp.WordNetLemmatizer()
    return _wnl.lemmatize(word).lower()


def get_tokenized_lemmas(str):
    return [normalize_word(t) for t in nlp.word_tokenize(str)]


def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]


def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric

    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()


def get_cosine_similarity(headlines, bodies, vec):
    cosine_simi= []
    for head, body in tqdm(zip(headlines, bodies)):
        head_tfidf = vec.transform([head]).toarray()
        body_tfidf = vec.transform([body]).toarray()
        tfidf_cos = cosine_similarity(head_tfidf, body_tfidf)[0]
        cosine_simi.append(tfidf_cos)
    return cosine_simi


def gen_or_load_feats(feat_fn, headlines, bodies, feature_file, vec=None):
    if not os.path.isfile(feature_file):
        feats = []
        if not vec is None:
            feats = feat_fn(headlines, bodies, vec)
        else:
            feats = feat_fn(headlines, bodies)
        np.save(feature_file, feats)

    return np.load(feature_file)


def word_overlap_features(headlines, bodies):
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        clean_headline = get_tokenized_lemmas(clean_headline)
        clean_body = get_tokenized_lemmas(clean_body)
        features = [
            len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]
        X.append(features)
    return X


def refuting_features(headlines, bodies):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        # 'refute',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_headline = get_tokenized_lemmas(clean_headline)
        features = [1 if word in clean_headline else 0 for word in _refuting_words]
        X.append(features)
    return X


def polarity_features(headlines, bodies):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]

    def calculate_polarity(text):
        tokens = get_tokenized_lemmas(text)
        return sum([t in _refuting_words for t in tokens]) % 2
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        features = []
        features.append(calculate_polarity(clean_headline))
        features.append(calculate_polarity(clean_body))
        X.append(features)
    return np.array(X)


def discuss_features(headlines, bodies):
    _discuss_words = [
        'according',
        'maybe',
        'reporting',
        'reports',
        'say', 'says',
        'claim',
        'claims',
        'purportedly',
        'investigating',
        'told',
        'tells',
        'allegedly',
        'validate',
        'verify'
    ]

    def calculate_discuss(text):
        tokens = get_tokenized_lemmas(text)
        return sum([t in _discuss_words for t in tokens]) % 2
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        features = []
        features.append(calculate_discuss(clean_headline))
        features.append(calculate_discuss(clean_body))
        X.append(features)
    return np.array(X)


def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def chargrams(input, n):
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def append_chargrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in chargrams(" ".join(remove_stopwords(text_headline.split())), size)]
    grams_hits = 0
    grams_early_hits = 0
    grams_first_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
        if gram in text_body[:100]:
            grams_first_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    features.append(grams_first_hits)
    return features


def append_ngrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in ngrams(text_headline, size)]
    grams_hits = 0
    grams_early_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    return features


def hand_features(headlines, bodies):

    def binary_co_occurence(headline, body):
        # Count how many times a token in the title
        # appears in the body text.
        bin_count = 0
        bin_count_early = 0
        for headline_token in clean(headline).split(" "):
            if headline_token in clean(body):
                bin_count += 1
            if headline_token in clean(body)[:255]:
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def binary_co_occurence_stops(headline, body):
        # Count how many times a token in the title
        # appears in the body text. Stopwords in the title
        # are ignored.
        bin_count = 0
        bin_count_early = 0
        for headline_token in remove_stopwords(clean(headline).split(" ")):
            if headline_token in clean(body):
                bin_count += 1
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def count_grams(headline, body):
        # Count how many times an n-gram of the title
        # appears in the entire body, and intro paragraph

        clean_body = clean(body)
        clean_headline = clean(headline)
        features = []
        features = append_chargrams(features, clean_headline, clean_body, 2)
        features = append_chargrams(features, clean_headline, clean_body, 8)
        features = append_chargrams(features, clean_headline, clean_body, 4)
        features = append_chargrams(features, clean_headline, clean_body, 16)
        features = append_ngrams(features, clean_headline, clean_body, 2)
        features = append_ngrams(features, clean_headline, clean_body, 3)
        features = append_ngrams(features, clean_headline, clean_body, 4)
        features = append_ngrams(features, clean_headline, clean_body, 5)
        features = append_ngrams(features, clean_headline, clean_body, 6)
        return features

    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        X.append(binary_co_occurence(headline, body)
                 + binary_co_occurence_stops(headline, body)
                 + count_grams(headline, body))
    return X


def get_all_features(name, headline, headline_cl, body, body_cl, vec):

    X_overlap = gen_or_load_feats(word_overlap_features, headline, body, "../features/overlap." + name + ".npy")
    X_refuting = gen_or_load_feats(refuting_features, headline, body, "../features/refuting." + name + ".npy")
    X_polarity = gen_or_load_feats(polarity_features, headline, body, "../features/polarity." + name + ".npy")
    X_discuss = gen_or_load_feats(discuss_features, headline, body, "../features/discuss." + name + ".npy")
    X_hand = gen_or_load_feats(hand_features, headline, body, "../features/hand." + name + ".npy")
    X_cosine = gen_or_load_feats(get_cosine_similarity, headline_cl, body_cl, "../features/cosine." + name + ".npy", vec)

    X = np.c_[X_hand, X_polarity, X_refuting, X_discuss, X_overlap, X_cosine]

    return X


def get_tfidf_vec(alltext, lim_unigram=None):
    if not lim_unigram is None:
        return TfidfVectorizer(max_features=lim_unigram).fit(alltext)
    return TfidfVectorizer().fit(alltext)  # Train and test sets


def get_tffreq_vec(alltext, lim_unigram):
    bow_vectorizer = CountVectorizer(max_features=lim_unigram)
    bow = bow_vectorizer.fit_transform(alltext)  # Train set only
    return TfidfTransformer(use_idf=False).fit(bow)

def get_headline_body_vec(name,headlines, bodies, tfidf_vec):
    feature_vec = []
    file_name = "../features/head_body_vec." + name + ".npy"
    if not os.path.isfile(file_name):
        for headline, body in tqdm(zip(headlines, bodies)):
            body_setences = body.split(".")
            headline_vector = tfidf_vec.transform([headline]).toarray()
            body_cosine = []
            for body in body_setences:
                body = clean(body)
                body_vector = tfidf_vec.transform([body]).toarray()
                cosine_value = cosine_similarity(headline_vector, body_vector)[0]
                body_cosine.append((body, cosine_value))
            body_cosine = sorted(body_cosine, key=lambda word: word[1], reverse=True)
            len_numb = 3 if 3 < len(body_cosine) else len(body_cosine)
            body_str = ""
            for index in range(len_numb):
                body_str += body_cosine[index][0]
            headline_vector = tfidf_vec.transform([headline]).toarray()
            body_tf_vec = tfidf_vec.transform([body_str]).toarray()
            feat_vec = np.squeeze(np.c_[headline_vector, body_tf_vec])
            feature_vec.append(feat_vec)
        np.save(file_name, feature_vec)

        return np.load(file_name)





