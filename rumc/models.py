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
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from random import random

class SpacyBaseline():

    def __init__(self, model_name, spm="nl_core_news_lg"):
        self.model_name = model_name
        self.model = spacy.load(spm)

    def label(self, documents):
        for doc, labels in documents:
            output = self.model(doc)
            print("Entities", [(e.text, e.label_) for e in output.ents])
            print(labels)

class CRF():

    def __init__(self, model_name, output="crf_output.txt", weights_file="crf_weights.pkl"):
        self.model_name = model_name
        self.weights = weights_file
        self.output = output
        self.model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )

    def train(self, documents):
        X_train, y_train = self._get_features(documents)
        self.model.fit(X_train, y_train)
        return

    def validate(self, documents, k=3, n_iter=5):
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
        crf = rs.best_estimator_
        with open(self.output, "a") as f:
            print("Best parameters: ", rs.best_params_, file=f)
            print("Best f1-score: ", rs.best_score_, file=f)
        f.close()

        with open(self.weights, "wb") as f:
            pickle.dump(crf, f)
        f.close()

        return crf

    def label(self, documents):
        X_test, y_test = self._get_features(documents)
        y_pred = self.model.predict(X_test, y_test)
        return y_pred

    def _get_features(self, data):
        x = [doc2features(s) for s in data]
        y = [doc2labels(s) for s in data]
        return (x, y)

class BILSTM():

    def __init__(self, model_name, output="bilstm_output.txt", weights_file="bilstm_weights.pkl"):
        self.model_name = model_name
        self.weights = weights_file
        self.output = output
        self.input_size = 0
        self.max_length = 0

    def train(self, docs):
        words, tags = self._get_word_tag_vocab(docs)
        word2idx, idx2word = self._get_mapping(words)
        tag2idx, idx2tag = self._get_mapping(tags)
        
        X, y = [], []
        for d in docs:
            self.input_size += 1
            tmpX, tmpY = [], []
            if len(d) > self.max_length:
                self.max_length = len(d)
            for tok,_,tag in d:
                tmpX.append(word2idx[tok])
                tmpY.append(tag2idx[tag])
            X.append(tmpX)
            y.append(tmpY)

        X_pad = tf.keras.preprocessing.sequence.pad_sequences(X, padding="post")
        y_pad = tf.keras.preprocessing.sequence.pad_sequences(y, padding="post")
        y_pad = y_pad.reshape(self.input_size, self.max_length, 1)

        self.model = self._get_model()
        self.model.summary()
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", 
                            metrics=[keras.metrics.Precision(), keras.metrics.Recall()])
        self.model.fit(X_pad, y_pad, epochs=50, verbose=2)
        result = self.model.predict(X_pad)
        print(result)

    def _get_mapping(self, vocab):
        mapping_1 = {i:idx+1 for idx,i in enumerate(vocab)}
        mapping_2 = {idx+1:i for idx,i in enumerate(vocab)}
        print(mapping_1, mapping_2)
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
        inputs = keras.Input(shape=(self.max_length,))
        x = layers.Embedding(input_dim=5000, output_dim=5, mask_zero=True)(inputs)
        x = layers.Bidirectional(layers.LSTM(64, input_shape=(self.max_length,1), return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(64, input_shape=(self.max_length,1), return_sequences=True))(x)
        outputs = layers.TimeDistributed(layers.Dense(1, activation="softmax"))(x)
        return keras.Model(inputs, outputs)

# DATA = []
# STANDARD_PIPE = ["tagger", "parser"]
# path = os.getcwd() + "/data-rumc/"

# with open(path + "anon_ground_truth_sample.jsonl", "r") as f:
#     test_file = list(f)

# # Extract the actual text and the PHI
# for doc in test_file:
#     result = json.loads(doc)
#     DATA.append((result['text'], result['labels']))

# spacy = SpacyBaseline("spacy")
# spacy.label(DATA)

# test = [[("I", 1, "O"), ("am", 1, "O"), ("Ismail", 1, "B-PER"), ("Guclu", 1, "I-PER")],
#         [("I", 1, "O"), ("am", 1, "O"), ("Ismail", 1, "B-PER")],
#         [("I", 1, "O"), ("am", 1, "O"), ("Ismail", 1, "B-PER"), ("Guclu", 1, "I-PER"), ("SOME", 1, "O"), ("SAY", 1, "O")]]

# bilstm = BILSTM("bilstm")
# bilstm.train(test)