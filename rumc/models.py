import os
import csv
import sklearn_crfsuite
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from features import doc2features, doc2labels, doc2tokens
import pickle
import json
import spacy
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.metrics import Precision, Recall
import tensorflow_addons as tfa
from random import random


# https://sklearn-crfsuite.readthedocs.io/en/latest/index.html
class CRF():

    def __init__(self, model_name, output="crf_output.txt", weights_file="crf_weights.pkl"):
        self.model_name = model_name
        self.weights = weights_file
        self.output = output
        self.model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=50,
            all_possible_transitions=True
        )

    def train(self, documents):
        X_train, y_train = self._get_features(documents)
        self.model.fit(X_train, y_train)
        return

    def validate(self, documents, k=3, n_iter=1):
        X_train, y_train = self._get_features(documents)
        params_space = {
            'c1': scipy.stats.expon(scale=0.5),
            'c2': scipy.stats.expon(scale=0.05),
        }

        f1_scorer = make_scorer(metrics.flat_f1_score,
                                average='weighted')
                                
        rs = RandomizedSearchCV(self.model, params_space,
                                cv=k,
                                verbose=1,
                                n_jobs=-1,
                                n_iter=n_iter,
                                scoring=f1_scorer)
        rs.fit(X_train, y_train)
        self.model = rs.best_estimator_
        with open(self.output, "a") as f:
            print("Best parameters: ", rs.best_params_, file=f)
            print("Best f1-score: ", rs.best_score_, file=f)
        f.close()

        with open(self.weights, "wb") as f:
            pickle.dump(self.model, f)
        f.close()
        return

    def label(self, documents):
        X_test, y_test = self._get_features(documents)
        y_pred = self.model.predict(X_test)
        return y_pred

    def _get_features(self, data):
        x = [doc2features(s) for s in data]
        y = [doc2labels(s) for s in data]
        return (x, y)

# https://www.kaggle.com/bhagone/bi-lstm-for-ner
class BILSTM():

    def __init__(self, model_name, labels, output="bilstm_output.txt", weights_file="bilstm_weights.pkl"):
        self.model_name = model_name
        self.weights = weights_file
        self.output = output
        self.input_size = 0
        self.max_length = 0
        self.nr_labels = labels
        checkpoint_filepath = './model_bilstm.{epoch:02d}-{val_loss:.2f}.hdf5'
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_best_only=True)

    def train(self, train, val):
        docs = train + val
        words, tags = self._get_word_tag_vocab(docs)
        word2idx, idx2word = self._get_mapping(words)
        tag2idx, idx2tag = self._get_mapping(tags)
        X_train, y_train, train_og_lengths  = self._split_data(train, word2idx, tag2idx)
        X_train_pad = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=self.max_length, padding="post", dtype="float64")
        y_train_pad = tf.keras.preprocessing.sequence.pad_sequences(y_train, maxlen=self.max_length, padding="post", dtype="float64")
        y_train_pad = [tf.keras.utils.to_categorical(y, num_classes=len(self.nr_labels)+1) for y in y_train_pad] 
        y_train_pad = np.reshape(y_train_pad, (self.input_size, self.max_length, len(self.nr_labels)+1))   

        X_val, y_val, val_og_lengths  = self._split_data(val, word2idx, tag2idx)
        X_val_pad = tf.keras.preprocessing.sequence.pad_sequences(X_val, maxlen=self.max_length, padding="post", dtype="float64")
        y_val_pad = tf.keras.preprocessing.sequence.pad_sequences(y_val, maxlen=self.max_length, padding="post", dtype="float64")
        y_val_pad = [tf.keras.utils.to_categorical(y, num_classes=len(self.nr_labels)+1) for y in y_val_pad] 
        y_val_pad = np.reshape(y_val_pad, (self.input_size, self.max_length, len(self.nr_labels)+1))          

        self.model = self._get_model()
        self.model.summary()
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", 
                            metrics=[Precision(), Recall()])
        self.model.fit(X_train_pad, y_train_pad, epochs=5, verbose=2, validation_split=0.2,
                        callbacks=[self.model_checkpoint_callback])
        result = self.model.predict(X_val_pad)
        result = self._post_processing(result, val_og_lengths)
        y_val_pred = self._get_bio_sequence(result, idx2tag)
        return y_val_pred, y_val

    def _split_data(self, docs, word2idx, tag2idx):
        X, y, og_length = [], [], []
        self.input_size = 0
        for d in docs:
            self.input_size += 1
            tmpX, tmpY = [], []
            tmp_length = len(d)
            og_length.append(tmp_length)
            if tmp_length > self.max_length:
                self.max_length = len(d)
            for tok,_,tag in d:
                tmpX.append(word2idx[tok])
                tmpY.append(tag2idx[tag])
            X.append(tmpX)
            y.append(tmpY)
        return X, y, og_length

    def _post_processing(self, sequences, lengths):
        new_sequences = []
        for i in range(len(lengths)):
            tmp = sequences[i][:lengths[i]] 
            new_sequences.append(tmp)
        return new_sequences

    def _get_bio_sequence(self, sequences, idx2tag):
        all_sequences = []
        for seq in sequences:
            tmp = []
            for i in seq:
                o = np.argmax(i, axis=-1)
                tmp.append(idx2tag[o.item()])
            all_sequences.append(tmp)
        return all_sequences

    def _get_mapping(self, vocab):
        mapping_1 = {i:idx+1 for idx,i in enumerate(vocab)}
        mapping_2 = {idx+1:i for idx,i in enumerate(vocab)}
        return mapping_1, mapping_2

    def _get_word_tag_vocab(self, docs):
        words_vocab = set()
        tags_vocab = set()
        for d in docs:
            for word,_,tag in d:
                words_vocab.add(word)
                tags_vocab.add(tag)
        return words_vocab, tags_vocab

    def _get_model(self):
        inputs = Input(shape=(self.max_length,))
        x = layers.Embedding(input_dim=50000, output_dim=100, mask_zero=True)(inputs)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        outputs = layers.TimeDistributed(layers.Dense(len(self.nr_labels)+1, activation="sigmoid"))(x)
        #crf = tfa.layers.CRF(len(self.nr_labels))
        #outputs = crf(x)
        return Model(inputs, outputs)