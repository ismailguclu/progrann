import os
import json
import spacy
from spacy.language import Language
import argparse
import models
from seqeval.metrics import classification_report
from sklearn.metrics import make_scorer
import pickle
import numpy as np
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer 
import random
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
CONVERSION = {"B-DATUM":"B-DATE", "I-DATUM":"I-DATE", "B-PERSOON":"B-PERSON", "I-PERSOON":"I-PERSON",
              "B-TELEFOONNUMMER":"B-PHONE", "I-TELEFOONNUMMER":"I-PHONE", "B-TIJD":"B-TIME", 
              "I-TIJD":"I-TIME", "B-PLAATS":"B-LOCATION", "I-PLAATS":"I-LOCATION"}
ENTITIES_DICT = {-1:"ABSTAIN", 0:"DATE", 1:"PERSON", 2:"TIME", 3:"LOCATION", 4:"PHONE", 5:"AGE", 6:"ZNUMMER"}

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
    b_train, s_train = helper.replace_tags_data(train, annotations_pred)

    L_val, Y_val = helper.helper_snorkel_representation(validation, 9)
    annotations_pred_val = label_model.predict(L=L_val)
    b_val, s_val = helper.replace_tags_data(validation, annotations_pred_val)
    return b_train, s_train, train['data'].to_list(), b_val, s_val, validation["data"].to_list()

def get_snorkel_i2b2_data():
    df = pd.read_pickle("./snorkel-data/i2b2-snorkel-18-2.pkl")
    df['data'] = list(zip(df.text, df.labels))
    train, validation, dev = np.split(
                                df.sample(frac=1, random_state=42), 
                                [int(.7*len(df)), int(.9*len(df))]
                                )
    L_train, Y = helper.helper_snorkel_representation(train, 18)
    label_model = LabelModel(verbose=False)
    label_model.load("./models/i2b2-label-model-18-2.pkl")
    annotations_pred = label_model.predict(L=L_train)
    b_train, s_train = helper.replace_tags_data_i2b2(train, annotations_pred)

    L_val, Y_val = helper.helper_snorkel_representation(validation, 18)
    annotations_pred_val = label_model.predict(L=L_val)
    b_val, s_val = helper.replace_tags_data_i2b2(validation, annotations_pred_val)
    return b_train, s_train, train['bio'].to_list(), b_val, s_val, validation["bio"].to_list()

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


def run_subset_data(b_train, s_train, b_val, s_val, b_test, s_test, emb_file, model, LABELS):
    f1_scores = []
    fractions = [20, 40, 60, 80, 100]
    print("Starting training splits...")
    
    for f in fractions:
        random.seed(42)
        b_sub, s_sub = zip(*random.sample(list(zip(b_train, s_train)), int(len(b_train)/100 * f)))
        if model == "crf":
            crf = models.CRF("crf")
            crf.validate(b_sub)             
            y_pred = crf.label(b_test)
        if model == "bilstm":
            bilstm = models.BILSTM("bilstm", LABELS, emb_file=emb_file)
            y_pred, _ = bilstm.train(list(b_sub), b_val, b_test)  
        f1_score = sequence_labeling.f1_score(s_val, y_pred)
        f1_scores.append(f1_score)
        print("Current split: {} and corresponding F1 score (val): {}".format(f, f1_score))
    print("F1 SCORES: ", f1_scores)
    return

def to_english_bio(b_test):
    english_y_test = []
    for bio in b_test:
        temp = []
        for word,pos,tag in bio:
            if tag in CONVERSION:
                temp.append((word,pos,CONVERSION[tag]))
            else:
                temp.append((word,pos,tag))
        english_y_test.append(temp)
    return english_y_test

def to_english(y_test):
    english_y_test = []
    for y in y_test:
        temp = []
        for i in y:
            if i in CONVERSION:
                temp.append(CONVERSION[i])
            else:
                temp.append(i)
        english_y_test.append(temp)
    return english_y_test

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
    parser.add_argument("--extra", help="Run experiment with subsets of data using [crf, bilstm].")
    parser.add_argument("--srumc", action="store_true", help="Load Snorkel model and train model on using WS (rumc).")
    parser.add_argument("--si2b2", action="store_true", help="Load Snorkel model and train model on using WS (i2b2).")
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
        bio, LABELS = helper.get_i2b2_data(TRAIN, data_path)
        sequences = helper.get_i2b2_sequences(bio)
        b_train, b_val, b_dev, s_train, s_val, s_dev = split_train_val_dev(bio, sequences)
        b_train = b_train[:10]
        b_val = b_val[:10]
        statistics_data(sequences)

    if args.rumc:
        rumc = get_rumc_data(data_path)
        bio, sequences = helper.bio_tagging(rumc)
        b_train, b_val, b_dev, s_train, sg_val, s_dev = split_train_val_dev(bio, sequences)
        test_path = "./data-rumc/500_extra_shuffled_clean.jsonl"
        test_rumc = get_rumc_data(test_path)
        b_test, sg_test = helper.bio_tagging(test_rumc)
        LABELS = RUMC_LABELS
        emb_file = "./fasttext/cc.nl.300.bin"

    if args.srumc:
        print("Loading data...")
        test_set = get_rumc_data("./data-rumc/500_extra_shuffled_clean.jsonl")
        train_set = get_rumc_data("./data-rumc/anon_ground_truth_v3_surrogates.jsonl")
        train_set, _,_,_,_,_ = split_train_val_dev(train_set, train_set)
        b_train, s_train, gold_train, b_val, s_val, gold_val = get_snorkel_rumc_data(test_set)
        b_test, s_test = helper.bio_tagging(test_set)
        b_test = to_english_bio(b_test)
        s_test = to_english(s_test)
        LABELS = RUMC_LABELS_EN
        emb_file = "./fasttext/cc.nl.300.bin"

    if args.si2b2:
        print("Loading data...")
        b_train, s_train, bio_train, b_val, s_val, bio_val = get_snorkel_i2b2_data()
        gold_bio_val = helper.extract_bio_type(bio_val)
        sg_val = helper.get_i2b2_sequences(gold_bio_val)

        TRAIN = os.listdir("./data/training-df/")
        TRAIN.sort()
        _, LABELS = helper.get_i2b2_data(TRAIN, "./data/training-df/")
        print(LABELS)

        TEST = os.listdir("./data/test-gold-df/")
        TEST.sort()
        emb_file = "./fasttext/cc.en.300.bin"
        b_test, _ = helper.get_i2b2_data(TEST, "./data/test-gold-df/")
        sg_test = helper.get_i2b2_sequences(b_test)
        print("Snorkel generative model performance on validation set.")

    if args.extra:
        print("Running experiment...")
        run_subset_data(b_train, s_train, b_val, s_val, emb_file, args.extra)

    if args.crf:
        print("Starting CRF...")
        crf = models.CRF("crf")
        crf.validate(b_train)             
        crf_y_pred_val = crf.label(b_val)      
        evaluate(crf_y_pred_val, sg_val, args.output, LABELS)

    if args.bilstm:
        bilstm = models.BILSTM("bilstm", LABELS, emb_file=emb_file)
        print("Start training..")
        bilstm_y_pred, _ = bilstm.train(b_train, b_val, b_test)    
        evaluate_partial(bilstm_y_pred, sg_test)
        with open("./output/output.txt", "wb") as fn:
            pickle.dump(bilstm_y_pred, fn)

if __name__ == "__main__":
    main(args_parser())