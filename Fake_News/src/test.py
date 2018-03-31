from DatasetRead import DatasetLoad
from gensim import corpora
from feature_extraction import FeatureExtraction
import nltk as nlp
import re
import numpy as np

## Feature Extraction code ##
fexc = FeatureExtraction()

## Train Data Load ##
data = DatasetLoad()
data.set_path(path='/home/amraw/my_repository/Machine-Learning-Projects/Fake_News/fnc-1-master')
train_stance_data = data.get_stance()
train_bodies_data = data.get_bodies()
train_headlines, train_bodies, train_stances = data.get_combined_data(train_stance_data, train_bodies_data)
train_headlines_cl = fexc.get_clean_data(train_headlines)
train_bodies_cl = fexc.get_clean_data(train_bodies)
train_stances_cl = fexc.get_clean_data(train_stances)

## Test Data Load ##
data.set_name("test")
test_stance_data = data.get_stance()
test_bodies_data = data.get_bodies()
test_headlines, test_bodies = data.get_combined_data(test_stance_data, test_bodies_data, data_type="test")
test_headlines_cl = fexc.get_clean_data(test_headlines)
test_bodies_cl = fexc.get_clean_data(test_bodies)

## Bag of Words ##

#new_data_train = fexc.get_clean_data(combined_data_train)
#new_data_test = fexc.get_clean_data(combined_data_test, data_type="test")
##  get_bag_of_words  ##
