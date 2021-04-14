import re
import spacy
import json
import pandas as pd
import numpy as np
import sources_rumc
import sources_i2b2
from snorkel.labeling import labeling_function
from snorkel.labeling.model import LabelModel
from scipy.sparse import dok_matrix, vstack, csr_matrix
from snorkel.labeling import LFAnalysis

# https://github.com/snorkel-team/snorkel/issues/1254
# dict_match() and create_token_L_mat()
def dict_match(sentence, dictionary, max_ngrams=4):
   m = {}
   for i in range(len(sentence)):
       for j in range(i+1, min(len(sentence), i + max_ngrams) + 1):
           term = ' '.join(sentence[i:j])
           if term in dictionary:
               m.update({idx:1 for idx in range(i,j+1)})
   return m
           
def create_token_L_mat(Xs, Ls, num_lfs):
   """
   Create token-level LF matrix from LFs indexed by sentence
   """
   Yws = []
   for sent_i in range(len(Xs)):
       ys = dok_matrix((len(Xs[sent_i]), num_lfs))
       for lf_i in range(num_lfs):
           for word_i,y in Ls[sent_i][lf_i].items():
               ys[word_i, lf_i] = y
       Yws.append(ys)
   return csr_matrix(vstack(Yws))

def helper_dict_values(dict_list):
    return [list(d.values()) for d in dict_list]

def helper_snorkel_representation(df):
    lfs = sources_rumc.get_rumc_sources()
    df["Y"] = df.apply(sources_rumc.apply_true_split, axis=1)
    df["X"] = df["text"].apply(sources_rumc.apply_token_split)
    df["L"] = df["text"].apply(sources_rumc.apply_lfs, args=(lfs,))
    Xs, Ls, Ys = df["X"].to_list(), df["L"].to_list(), df["Y"].to_list()
    L = create_token_L_mat(Xs, Ls, len(lfs))
    L = np.asarray(L.astype(np.int8).todense())
    Y = helper_dict_values(Ys)
    Y = np.concatenate(Y).ravel()
    return L, Y, lfs, df

def helper_snorkel_i2b2_representation(df):
    lfs = sources_i2b2.get_i2b2_sources()
    df["Y"] = df.apply(sources_i2b2.apply_true_split, axis=1)
    df["X"] = df["bio"].apply(sources_i2b2.apply_token_split)
    df["L"] = df["text"].apply(sources_i2b2.apply_lfs, args=(lfs,))
    Xs, Ls, Ys = df["X"].to_list(), df["L"].to_list(), df["Y"].to_list()
    L = create_token_L_mat(Xs, Ls, len(lfs))
    L = np.asarray(L.astype(np.int8).todense())
    Y = helper_dict_values(Ys)
    Y = np.concatenate(Y).ravel()
    return L, Y, lfs, df

def validate(df, label_model, type_data="rumc"):
    if type_data == "rumc":
        L, Y, lfs, df = helper_snorkel_representation(df)
        label_model_acc = label_model.score(L=L, Y=Y, tie_break_policy="random")["accuracy"]
        print(f"{'Label Model Accuracy (Validation):':<25} {label_model_acc * 100:.1f}%")
    else:
        print("here comes i2b2 validation")
    return

def train(df, cardinality=8, type_data="rumc", gold_labels=False):
    if type_data == "rumc":
        L, Y, lfs, df = helper_snorkel_representation(df)
    else:
        L, Y, lfs, df = helper_snorkel_i2b2_representation(df)

    label_model = LabelModel(cardinality=cardinality)
    label_model.fit(L)
    if gold_labels:
        print(LFAnalysis(L=L, lfs=lfs).lf_summary(Y=Y))
        print(label_model.score(L, Y=Y, metrics=["f1_micro"]))
        label_model_acc = label_model.score(L=L, Y=Y, tie_break_policy="random")["accuracy"]
        print(f"{'Label Model Accuracy (Development):':<25} {label_model_acc * 100:.1f}%")
    return label_model, df