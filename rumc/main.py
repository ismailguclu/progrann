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
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from nltk.stem.wordnet import WordNetLemmatizer 
from random import random
import helper
from snorkel.labeling.model import LabelModel
import sequence_labeling

CONTEXT = 2
lemmatizer = WordNetLemmatizer() 
STANDARD_PIPE = ["tagger", "parser"]
nlp = spacy.load("nl_core_news_lg")
nlp.remove_pipe("ner")
RUMC_LABELS = ["B-PERSOON", "I-PERSOON", "B-DATUM", "I-DATUM", "O", "B-TIJD", "I-TIJD",
               "B-TELEFOONNUMMER", "I-TELEFOONNUMMER", "B-PATIENTNUMMER", "I-PATIENTNUMMER",
               "B-ZNUMMER", "I-ZNUMMER", "B-LEEFTIJD", "I-LEEFTIJD", "B-PLAATS", "I-PLAATS"]
RUMC_LABELS_EN = ["B-PERSON", "I-PERSON", "B-DATE", "I-DATE", "O", "B-TIME", "I-TIME",
                  "B-PHONE", "I-PHONE", "B-ZNUMMER", "I-ZNUMMER", "B-AGE", "I-AGE", 
                  "B-LOCATION", "I-LOCATION"]
ENTITIES_DICT = {-1:"ABSTAIN", 0:"DATE", 1:"PERSON", 2:"TIME", 3:"LOCATION", 4:"PHONE", 5:"AGE", 6:"ZNUMMER"}
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

def replace_tags_data(df, pred):
    idx = 0
    text = df["text"].to_list()
    x_tokens = df["X"].to_list()
    all_sequences = []
    all_reports = []
    nlp = spacy.load("nl_core_news_lg")
    nlp.remove_pipe("ner")


    for i in range(len(df)):
        elem = x_tokens[i]
        text_elem = text[i]
        selection = pred[idx:idx+len(elem)]
        temp_sequence = []
        temp_report = []
        prev_elem = -1

        # BIO tagging WS annotated labels 
        # https://pythonprogramming.net/using-bio-tags-create-named-entity-lists/    
        for s in selection:
            if s == -1:
                temp_sequence.append("O")
                prev_elem = -1
            if s != -1 and prev_elem == -1:
                temp_sequence.append("B-" + ENTITIES_DICT[s])
                prev_elem = s
            elif s == prev_elem and prev_elem != -1:
                temp_sequence.append("I-" + ENTITIES_DICT[s])
                prev_elem = s
            elif prev_elem != -1 and prev_elem != s: 
                temp_sequence.append("B-"+ ENTITIES_DICT[s])
                prev_elem = s
        all_sequences.append(temp_sequence)
        idx += len(elem)

        # Create tuple: (text, POS, BIO-tag)
        doc = nlp(text_elem)
        for ent in doc:
            temp_report.append((ent.text, ent.pos_))
        
        for i in range(len(temp_sequence)):
            temp_report[i] = temp_report[i] + (temp_sequence[i],)
        all_reports.append(temp_report)
    return all_reports, all_sequences

def get_snorkel_rumc_data():
    df = pd.read_pickle("./data-rumc/rumc-snorkel-9-5.pkl")
    df['data'] = list(zip(df.text, df.labels))
    train, validation, dev = np.split(
                                df.sample(frac=1, random_state=42), 
                                [int(.7*len(df)), int(.9*len(df))]
                                )
    L_train, Y = helper.helper_snorkel_representation(train, 9)
    label_model = LabelModel(verbose=False)
    label_model.load("./models/rumc-label-model-9-5.pkl")
    annotations_pred = label_model.predict(L=L_train)
    b_train, s_train = replace_tags_data(train, annotations_pred)

    L_val, Y_val = helper.helper_snorkel_representation(validation, 9)
    annotations_pred_val = label_model.predict(L=L_val)
    b_val, s_val = replace_tags_data(validation, annotations_pred_val)
    return b_train, s_train, train['data'].to_list(), b_val, s_val, validation["data"].to_list()

def get_rumc_data(path):
    data = []

    with open(path + "anon_ground_truth_v3_surrogates.jsonl", "r") as f:
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

def evaluate(sequences, gold, labels, per_entity=True):
    print(classification_report(gold, sequences))
    return
    # if per_entity:
    #     with open(output, "a") as f:
    #         print(output, file=f)
    #         print(classification_report(gold, sequences), file=f)
    #     f.close()

def evaluate_partial(sequences, gold):
    print("F1", sequence_labeling.f1_score(gold, sequences))
    print("PARTIAL", sequence_labeling.partial_f1_score(gold, sequences))
    print(sequence_labeling.classification_report(gold, sequences))
    return

def split_train_val_dev(bio, sequences):
    df = pd.DataFrame({"bio": bio, "seq":sequences})
    train, val, dev = np.split(
                        df.sample(frac=1, random_state=42), 
                        [int(.7*len(df)), int(.9*len(df))]
                        )
    b_train, b_val, b_dev = train["bio"].to_list(), val["bio"].to_list(), dev["bio"].to_list()
    s_train, s_val, s_dev = train["seq"].to_list(), val["seq"].to_list(), dev["seq"].to_list()
    return b_train, b_val, b_dev, s_train, s_val, s_dev

def statistics_data(sequences):
    nr_of_tokens = 0
    nr_of_phi_tokens = 0
    for s in sequences:
        nr_of_tokens += len(s)
        for token in s:
            if token != "O":
                nr_of_phi_tokens += 1
    
    print("Overview dataset:")
    print("-----------------")
    print("Nr of tokens: ", nr_of_tokens)
    print("Nr of PHI tokens: ", nr_of_phi_tokens)
    print("Percentage: ", nr_of_phi_tokens/nr_of_tokens)
    return

def args_parser():
    parser = argparse.ArgumentParser(description='Run models on selected data.')
    parser.add_argument("--rumc", action="store_true", help="Load RUMC data.")
    parser.add_argument("--path", help="Absolute path to data file/folder.")
    parser.add_argument("--output", help="Name of file for report output.")
    parser.add_argument("--i2b2", action="store_true", help="Load i2b2 data.")
    parser.add_argument("--crf", action="store_true", help="Load CRF model.")
    parser.add_argument("--bilstm", action="store_true", help="Load bi-LSTM model.")
    parser.add_argument("--srumc", action="store_true", help="Load Snorkel model and train model on using WS.")
    return parser.parse_args()

def main(args):
    """
    Variable notation:
        - b_xxxx = bio (train/dev/val)
        - s_xxxx = sequences (train/dev/val)
        - bg_xxx = bio gold
        - sg_xxx = sequences gold
    """
    print("Start...")
    if args.path:
        data_path = args.path

    if args.i2b2:
        TRAIN = os.listdir(data_path)
        TRAIN.sort()
        bio, LABELS = get_i2b2_data(TRAIN, data_path)
        sequences = get_i2b2_sequences(bio)
        b_train, b_val, b_dev, s_train, s_val, s_dev = split_train_val_dev(bio, sequences)
        print(LABELS)
        statistics_data(sequences)

    if args.rumc:
        rumc = get_rumc_data(data_path)
        bio, sequences = bio_tagging(rumc)
        b_train, b_val, b_dev, s_train, s_val, s_dev = split_train_val_dev(bio, sequences)
        LABELS = RUMC_LABELS
        statistics_data(sequences)

    if args.srumc:
        print("Loading data...")
        b_train, s_train, gold_train, b_val, s_val, gold_val = get_snorkel_rumc_data()
        bg_val, sg_val = bio_tagging(gold_val)
        LABELS = RUMC_LABELS_EN
        print("Snorkel generative model performance on validation set.")
        #evaluate(s_val, sg_val, LABELS)
        evaluate_partial(s_val, sg_val)

    if args.crf:
        print("Starting CRF...")
        crf = models.CRF("crf")
        crf.validate(b_train)             
        crf_y_pred_val = crf.label(b_val)      
        evaluate(crf_y_pred_val, sg_val, args.output, LABELS)

    if args.bilstm:
        bilstm = models.BILSTM("bilstm", LABELS)
        bilstm_y_val_pred, bilstm_y_val = bilstm.train(train, val)    
        #evaluate(bilstm_y_val_pred, bilstm_y_val, args.output, LABELS)
        print(classification_report(bilstm_y_val, bilstm_y_val_pred))

if __name__ == "__main__":
    main(args_parser())