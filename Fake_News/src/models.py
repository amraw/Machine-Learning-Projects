import os

from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import Input, merge, TimeDistributed, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import GRU, LSTM
from keras.layers.core import Flatten, Dropout, Dense
from keras.models import Model
from keras.layers import Bidirectional
import numpy as np


def get_embeddings_index(glove_dir):
    embeddings_index = {}
    with open(os.path.join(glove_dir, 'glove.6B.100d.txt')) as embedding:
        for line in embedding:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index


def get_embedding_matrix(word_index, embedding_dim, embeddings_index):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def lstm_model(headline_length, body_length, embedding_dim, word_index, embedding_matrix, activation, numb_layers, drop_out, cells):
    headline_embedding_layer = Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix],
                                         input_length=headline_length, trainable=False)

    bodies_embedding_layer = Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix],
                                       input_length=body_length, trainable=False)

    headline_input = Input(shape=(headline_length,), dtype='int32')
    headline_embedding = headline_embedding_layer(headline_input)
    headline_nor = BatchNormalization()(headline_embedding)
    headline_lstm = LSTM(cells, dropout_U=0.25, dropout_W=0.25)(headline_nor)

    body_input = Input(shape=(body_length,), dtype='int32')
    body_embedding = bodies_embedding_layer(body_input)
    body_nor = BatchNormalization()(body_embedding)
    body_lstm = LSTM(cells, dropout_U=0.25, dropout_W=0.25)(body_nor)

    concat = concatenate([headline_lstm, body_lstm])
    normalize = BatchNormalization()(concat)
    dense = Dense(numb_layers, activation=activation)(normalize)
    dropout = Dropout(drop_out)(dense)
    dense2 = Dense(numb_layers, activation=activation)(dropout)
    dropout1 = Dropout(drop_out)(dense2)
    dense3 = Dense(numb_layers, activation=activation)(dropout1)
    dropout2 = Dropout(drop_out)(dense3)
    normalize2 = BatchNormalization()(dropout2)

    preds = Dense(4, activation='softmax')(normalize2)

    fake_nn = Model([headline_input, body_input], outputs=preds)
    print(fake_nn.summary())
    fake_nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return fake_nn


def bow_model(headline_length, body_length, embedding_dim, word_index, embedding_matrix, activation, numb_layers, drop_out):
    headline_embedding_layer = Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix],
                                         input_length=headline_length, trainable=False)

    bodies_embedding_layer = Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix],
                                       input_length=body_length, trainable=False)

    headline_input = Input(shape=(headline_length,), dtype='int32')
    headline_embedding = headline_embedding_layer(headline_input)
    headline_nor = BatchNormalization()(headline_embedding)

    body_input = Input(shape=(body_length,), dtype='int32')
    body_embedding = bodies_embedding_layer(body_input)
    body_nor = BatchNormalization()(body_embedding)
    flatten1 = Flatten()(headline_nor)
    flatten2 = Flatten()(body_nor)
    concat = concatenate([flatten1, flatten2])
    dense = Dense(numb_layers, activation=activation)(concat)
    dropout = Dropout(drop_out)(dense)
    dense2 = Dense(numb_layers, activation=activation)(dropout)
    dropout1 = Dropout(drop_out)(dense2)
    dense3 = Dense(numb_layers, activation=activation)(dropout1)
    dropout2 = Dropout(drop_out)(dense3)
    normalize2 = BatchNormalization()(dropout2)
    #flatten = Flatten()(nomralize2)
    preds = Dense(4, activation='softmax')(normalize2)

    fake_nn = Model([headline_input, body_input], preds)
    print(fake_nn.summary())
    fake_nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return fake_nn


def bow_model_2(headline_length, body_length, embedding_dim, word_index, embedding_matrix, activation, numb_layers, drop_out):
    embedding_layer = Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix],
                                         input_length=headline_length+body_length, trainable=False)

    input = Input(shape=(headline_length+body_length,), dtype='int32')
    embedding = embedding_layer(input)
    nomrmalization_1 = BatchNormalization()(embedding)

    dense = Dense(numb_layers, activation=activation)(nomrmalization_1)
    dropout = Dropout(drop_out)(dense)
    dense2 = Dense(numb_layers, activation=activation)(dropout)
    dropout1 = Dropout(drop_out)(dense2)
    dense3 = Dense(numb_layers, activation=activation)(dropout1)
    dropout2 = Dropout(drop_out)(dense3)
    normalize2 = BatchNormalization()(dropout2)
    flatten = Flatten()(normalize2)
    preds = Dense(4, activation='softmax')(flatten)
    fake_nn = Model(input, preds)
    print(fake_nn.summary())
    fake_nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return fake_nn

def bi_dir_lstm_model(headline_length, body_length, embedding_dim, word_index, embedding_matrix, activation, numb_layers, drop_out, cells):
    headline_embedding_layer = Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix],
                                         input_length=headline_length, trainable=False)

    bodies_embedding_layer = Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix],
                                       input_length=body_length, trainable=False)

    headline_input = Input(shape=(headline_length,), dtype='int32')
    headline_embedding = headline_embedding_layer(headline_input)
    headline_nor = BatchNormalization()(headline_embedding)
    head_bi_dir = Bidirectional(LSTM(cells, dropout_U=0.25, dropout_W=0.25))(headline_nor)

    body_input = Input(shape=(body_length,), dtype='int32')
    body_embedding = bodies_embedding_layer(body_input)
    body_nor = BatchNormalization()(body_embedding)
    body_bi_dir = Bidirectional(LSTM(cells, dropout_U=0.25, dropout_W=0.25))(body_nor)

    concat = concatenate([head_bi_dir, body_bi_dir])
    normalize = BatchNormalization()(concat)
    dense = Dense(numb_layers, activation=activation)(normalize)
    dropout = Dropout(drop_out)(dense)
    dense2 = Dense(numb_layers, activation=activation)(dropout)
    dropout1 = Dropout(drop_out)(dense2)
    dense3 = Dense(numb_layers, activation=activation)(dropout1)
    dropout2 = Dropout(drop_out)(dense3)
    normalize2 = BatchNormalization()(dropout2)

    preds = Dense(4, activation='softmax')(normalize2)

    fake_nn = Model([headline_input, body_input], outputs=preds)
    print(fake_nn.summary())
    fake_nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return fake_nn