import os
from DatasetRead import DatasetLoad
from feature_extraction import FeatureExtraction
import numpy as np
from data_analysis import DataAnalysis
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import models
import pickle
import csv
import save_images as sv

GLOVE_DIR = "/home/amraw/my_repository/Machine-Learning-Projects/Fake_News/gloVe"
PREDICTIONS_FILE = '../prediction/predicted_test_bog.csv'
TEST_FILE = '../fnc-1-master/test_stances.csv'
OBJECT_DUMP = '../objects'
IMAGES_PATH = '../figures/bog'

## Feature Extraction code ##
fexc = FeatureExtraction()

## Train Data Load ##
data = DatasetLoad()
data.set_path(path='/home/amraw/my_repository/Machine-Learning-Projects/Fake_News/fnc-1-master')
train_stance_data = data.get_stance()
train_bodies_data = data.get_bodies()
train_headlines, train_bodies, train_stances = data.get_combined_data(train_stance_data, train_bodies_data)

# Train headlines
train_headlines_cl = fexc.get_clean_data(train_headlines)
train_bodies_cl = fexc.get_clean_data(train_bodies)
train_stances_cl = fexc.get_clean_data(train_stances)

# remove stop words
train_headlines_cl = fexc.remove_stop_words_list(train_headlines_cl)
train_bodies_cl = fexc.remove_stop_words_list(train_bodies_cl)

# Word to integer
train_stances_in = fexc.convert_lable_int(train_stances_cl)

# perform stemming
# train_headlines_cl = fexc.perform_stemming_list(train_headlines_cl)
# train_bodies_cl = fexc.perform_stemming_list(train_bodies_cl)

## Test Data Load ##
data.set_name("test")
test_stance_data = data.get_stance()
test_bodies_data = data.get_bodies()
test_headlines, test_bodies = data.get_combined_data(test_stance_data, test_bodies_data, data_type="test")

test_headlines_cl = fexc.get_clean_data(test_headlines)
test_bodies_cl = fexc.get_clean_data(test_bodies)

# Remove Stop words #
test_headlines_cl = fexc.remove_stop_words_list(test_headlines_cl)
test_bodies_cl = fexc.remove_stop_words_list(test_bodies_cl)

# Perform stemming #
# test_headlines_cl = fexc.perform_stemming_list(test_headlines_cl)
# test_bodies_cl = fexc.perform_stemming_list(test_bodies_cl)

MAX_HEADLINE_LENGTH = 75
MAX_BODY_LENGTH = 75
EMBEDDING_DIM = 100

alltext = train_headlines_cl + train_bodies_cl + test_headlines_cl + test_bodies_cl
token = Tokenizer(num_words=30000)
token.fit_on_texts(alltext)
print(len(token.word_index.keys()))

train_headlines_seq = token.texts_to_sequences(train_headlines_cl)
train_bodies_seq = token.texts_to_sequences(train_bodies_cl)
word_index = token.word_index

train_headlines_seq = pad_sequences(train_headlines_seq, maxlen=MAX_HEADLINE_LENGTH)
train_bodies_seq = pad_sequences(train_bodies_seq, maxlen=MAX_BODY_LENGTH)

onehotencoder = OneHotEncoder()
train_stances_in = onehotencoder.fit_transform(train_stances_in).toarray()

#print(train_stances_in)
#print(train_stances_in.shape)
train_headlines_final, headlines_val, train_bodies_final, bodies_val, train_stances_final, stances_val = \
    train_test_split(train_headlines_seq, train_bodies_seq, train_stances_in, test_size=0.2, random_state=42)

test_headlines_seq = token.texts_to_sequences(test_headlines_cl)
test_bodies_seq = token.texts_to_sequences(test_bodies_cl)


test_headlines_seq = pad_sequences(test_headlines_seq, maxlen=MAX_HEADLINE_LENGTH)
test_bodies_seq = pad_sequences(test_bodies_seq, maxlen=MAX_BODY_LENGTH)

# Getting embedding index
embeddings_index = models.get_embeddings_index(GLOVE_DIR)

print('Found %s word vectors.' % len(embeddings_index))

# Getting embedding matrix
embedding_matrix = models.get_embedding_matrix(embedding_dim=EMBEDDING_DIM, embeddings_index=embeddings_index,
                                               word_index=word_index)

fake_nn = models.bog_model(headline_length=MAX_HEADLINE_LENGTH, body_length=MAX_BODY_LENGTH,
                           embedding_dim=EMBEDDING_DIM, word_index=word_index, embedding_matrix=embedding_matrix,
                           activation='relu',
                           drop_out=0.2, numb_layers=100)

early_stopping =EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = 'Fake_news_nlp.h5'

model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

fake_hist = fake_nn.fit([train_headlines_final, train_bodies_final], train_stances_final, batch_size=100,
                        epochs=40, shuffle=True, validation_data=([headlines_val, bodies_val], stances_val),
                        callbacks=[early_stopping, model_checkpoint])

print(fake_hist.history.keys())
bog_list_data = []

with open(os.path.join(OBJECT_DUMP, 'bog_history'), 'wb') as bog_hist:
    bog_list_data.append(fake_hist.history['acc'])
    bog_list_data.append(fake_hist.history['val_acc'])
    bog_list_data.append(fake_hist.history['loss'])
    bog_list_data.append(fake_hist.history['val_loss'])
    pickle.dump(bog_list_data, bog_hist)

sv.save_plt_images(fake_hist, IMAGES_PATH)

result = fake_nn.predict([test_headlines_seq, test_bodies_seq], batch_size=100)
#result = np.random.randint(low=0, high=1, size=(len(test_bodies_seq), 4))
#print(result)
#print(result.shape)
result_str = fexc.convert_lable_string(result)
with open(TEST_FILE, 'r') as read_file:
    test_stance = csv.DictReader(read_file)
    with open(PREDICTIONS_FILE, 'w') as write_file:
        writer = csv.DictWriter(write_file, fieldnames=['Headline', 'Body ID', 'Stance'])
        writer.writeheader()
        for sample, prediction in zip(test_stance, result_str):
            writer.writerow({'Body ID': sample['Body ID'], 'Headline': sample['Headline'], 'Stance': prediction})