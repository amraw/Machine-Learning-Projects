import nltk as nlp
import numpy as np
import re
from sklearn import feature_extraction
from nltk.stem.snowball import SnowballStemmer
from enum import Enum

class Stance(Enum):
    agree = 0
    unrelated = 1
    discuss = 2
    disagree = 3

class FeatureExtraction:

    def __init__(self):
        self.word_lemmatize = nlp.WordNetLemmatizer()

    def preprocess_string(self, text):
        processed = " ".join(re.findall(r'\w+', text, flags=re.UNICODE)).lower()
        return self.remove_stop_words_str(processed)

    def stem_normalize(self, word):
        return self.word_lemmatize.lemmatize(word).lower()

    def get_tokens(self, text):
        return [self.stem_normalize(word) for word in nlp.word_tokenize(text)]

    def remove_stop_words(self, token):
        return [word for word in token if word not in feature_extraction.text.ENGLISH_STOP_WORDS]

    def remove_stop_words_str(self, string):
        token = string.split()
        token = [word for word in token if word not in feature_extraction.text.ENGLISH_STOP_WORDS]
        return " ".join(token)

    def remove_stop_words_list(self, list):
        new_list = []
        for string in list:
            new_string = self.remove_stop_words_str(string)
            new_list.append(new_string)
        return new_list

    def perform_stemming(self, string):
        stemmer = SnowballStemmer("english")
        token = string.split()
        token = [stemmer.stem(word) for word in token]
        return " ".join(token)

    def perform_stemming_list(self, list):
        new_list = []
        for string in list:
            new_string = self.perform_stemming(string)
            new_list.append(new_string)
        print(new_list[0])
        return new_list

    def get_clean_data(self, data_list):
        processed_data = list()
        for data in data_list:
            processed_data.append(self.preprocess_string(data))
        return processed_data

    def convert_lable_int(self, label_list):
        new_label = []
        for label in label_list:
            if label == 'agree':
                new_label.append(Stance.agree.value)
            elif label == 'unrelated':
                new_label.append(Stance.unrelated.value)
            elif label == 'discuss':
                new_label.append(Stance.discuss.value)
            elif label == 'disagree':
                new_label.append(Stance.disagree.value)
            else:
                raise ValueError("Invalid Label type")
        return np.array(new_label).reshape(-1, 1)

    def convert_lable_string(self, class_probs):
        new_label = []
        for prob in class_probs:
            label = self.get_max(prob)
            if label == Stance.agree.value:
                new_label.append('agree')
            elif label == Stance.unrelated.value:
                new_label.append('unrelated')
            elif label == Stance.disagree.value:
                new_label.append('disagree')
            elif label == Stance.discuss.value:
                new_label.append('discuss')
            else:
                raise ValueError("Invalid Label type")
        return new_label

    def get_max(self, probs):
        index = 0
        if probs[index] < probs[1]:
            index = 1
        if probs[index] < probs[2]:
            index = 2
        if probs[index] < probs[3]:
            index = 3
        return index

    def get_bag_words(self, data_list):
        for data in data_list:
            headline = self.preprocess_string(data[0])
            body = self.preprocess_string(data[1])
