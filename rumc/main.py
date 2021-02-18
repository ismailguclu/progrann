import os
import json
import spacy
from spacy.language import Language
import argparse
import csv
from EntityPlacer import EntityPlacer
import models
from seqeval.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import sklearn_crfsuite
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from features import doc2features, doc2labels, doc2tokens
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from nltk.stem.wordnet import WordNetLemmatizer 
from random import random

CONTEXT = 2
lemmatizer = WordNetLemmatizer() 
STANDARD_PIPE = ["tagger", "parser"]
nlp = spacy.load("nl_core_news_lg")
nlp.remove_pipe("ner")
RUMC_LABELS = ["B-PERSOON", "I-PERSOON", "B-DATUM", "I-DATUM", "O", "B-TIJD", "I-TIJD",
               "B-TELEFOONNUMMER", "I-TELEFOONNUMMER", "B-PATIENTNUMMER", "I-PATIENTNUMMER",
               "B-ZNUMMER", "I-ZNUMMER", "B-LEEFTIJD", "I-LEEFTIJD", "B-PLAATS", "I-PLAATS"]
# TEST_PATH = os.getcwd() + "/data/test-gold-df/"
# MODEL_PATH = os.getcwd() + "/model/"
# TEST = os.listdir(TEST_PATH)
# TEST.sort()

def get_i2b2_data(data, data_path):
   tmp_data = []
   tmp_labels = []
   for filename in data:
      tmp_file = []
      with open(os.path.join(data_path, filename), newline="") as f:
         reader = csv.reader(f, delimiter="\t")
         for r in reader:
            tmp_file.append((r[0], r[1], r[4]))
            tmp_labels.append(r[4])
      tmp_data.append(tmp_file)
   return tmp_data, set(tmp_labels)

def get_i2b2_sequences(data):
    all_sequences = []
    for doc in data:
        tmp = []
        for ent in doc:
            tmp.append(ent[2])
        all_sequences.append(tmp)
    return all_sequences

@Language.factory("entity_component")
def my_entity_component(nlp, name, entities):
    return EntityPlacer(entities)

def bio_tagging(data):
    all_reports = []
    all_sequences = []
    for text, labels in data:
        entity_placer = EntityPlacer(labels)
        if entity_placer == -1:
            continue
        else:
            nlp.add_pipe("entity_component", name="placer", last=True, config={"entities":labels})
            doc = nlp(text)
            report = []
            seq = []
            for ent in doc:
                if ent.ent_iob_ == "O":
                    report.append((ent.text, ent.pos_, "O"))
                    seq.append("O")
                else:
                    report.append((ent.text, ent.pos_, ent.ent_iob_ + "-" + ent.ent_type_))
                    seq.append(ent.ent_iob_ + "-" + ent.ent_type_)
            all_reports.append(report)
            all_sequences.append(seq)
            nlp.remove_pipe("placer")
    return all_reports, all_sequences

def get_rumc_data(path):
    data = []

    with open(path + "anon_ground_truth_sample.jsonl", "r") as f:
        test_file = list(f)

    for doc in test_file:
        result = json.loads(doc)
        text, labels = result['text'], result['labels']
        new_labels = []
        for s, e, ent in labels:
            ent_text = text[s:e]
            new_labels.append((s, e, ent[1:-1], ent_text))
        data.append((text, new_labels))
    return data

def evaluate(sequences, gold, output, labels, per_entity=True):
    if per_entity:
        with open(output, "a") as f:
            print(output, file=f)
            print(classification_report(gold, sequences), file=f)
        f.close()

def args_parser():
    parser = argparse.ArgumentParser(description='Run models on selected data.')
    parser.add_argument("--rumc", action="store_true", help="Load RUMC data.")
    parser.add_argument("--path", help="Absolute path to data file/folder.")
    parser.add_argument("--output", help="Name of file for report output.")
    parser.add_argument("--i2b2", action="store_true", help="Load i2b2 data.")
    parser.add_argument("--crf", action="store_true", help="Load CRF model.")
    parser.add_argument("--bilstm", action="store_true", help="Load bi-LSTM model.")
    return parser.parse_args()

def main(args):
    if args.path:
        data_path = args.path

    if args.i2b2:
        TRAIN = os.listdir(data_path)
        TRAIN.sort()
        bio, LABELS = get_i2b2_data(TRAIN, data_path)
        sequences = get_i2b2_sequences(bio)
        print(LABELS)

    if args.rumc:
        rumc = get_rumc_data(data_path)
        bio, sequences= bio_tagging(rumc)
        LABELS = RUMC_LABELS

    if args.crf:
        crf = models.CRF("crf")
        crf.train(bio)
        crf_y_pred = crf.label(bio)      
        evaluate(crf_y_pred, sequences, args.output, LABELS)

    if args.bilstm:
        bilstm = models.BILSTM("bilstm", LABELS)
        bilstm_y_pred = bilstm.train(bio)    
        evaluate(bilstm_y_pred, sequences, args.output, LABELS)

if __name__ == "__main__":
    main(args_parser())