import re
import os
import csv
import spacy
import json
import pandas as pd
import numpy as np
from EntityPlacer import EntityPlacer
from spacy.language import Language
from snorkel.labeling import labeling_function
from snorkel.labeling.model import LabelModel
from scipy.sparse import dok_matrix, vstack, csr_matrix
from snorkel.labeling import LFAnalysis

nlp = spacy.load("nl_core_news_lg")
nlp.remove_pipe("ner")
ENTITIES_DICT = {-1:"ABSTAIN", 0:"DATE", 1:"PERSON", 2:"TIME", 
                  3:"LOCATION", 4:"PHONE", 5:"AGE", 6:"ZNUMMER"}
ENTITIES_DICT_2 = {"<DATUM>":"DATE", "<PERSOON>":"PERSON", "<TIJD>":"TIME", 
                   "<PLAATS>":"LOCATION", "<TELEFOONNUMMER>":"PHONE", 
                   "<LEEFTIJD>":"AGE", "<ZNUMMER>":"ZNUMMER"}
ENTITIES_DICT_i2b2 = {-1:'ABSTAIN', 0:'HEALTHPLAN', 1:'LOCATION-OTHER', 2:'ORGANIZATION', 
                      3:'DEVICE', 4:'STREET', 5:'CITY', 6:'ZIP', 7:'HOSPITAL', 
                      8:'MEDICALRECORD', 9:'IDNUM', 10:'FAX', 11:'DATE', 12:'PHONE', 
                      13:'COUNTRY', 14:'URL', 15:'PROFESSION', 16:'STATE', 17:'PATIENT', 
                      18:'EMAIL', 19:'DOCTOR', 20:'BIOID', 21:'AGE', 24:'USERNAME'} # why 24 suddenly?

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

def extract_bio_type(bio):
    all_bio_types = []
    for elem in bio:
        temp = []
        for el in elem:
            if el[2] == "O":
                temp.append((el[0], el[1], "O"))
            else:
                temp.append((el[0], el[1], el[2] + "-" + el[3]))
        all_bio_types.append(temp)
    return all_bio_types

def get_rumc_data(fn):
    data = []
    labels = []

    with open("./data-rumc/"+fn, "r") as f:
        files = list(f)

    for doc in files:
        result = json.loads(doc)
        text = result['text']
        data.append(text)
        temp_labels = result['labels']
        if temp_labels:
            temp = []
            for tl in temp_labels:
                s,e = tl[0], tl[1]
                l,t = ENTITIES_DICT_2[tl[2]], text[s:e]
                temp.append([s,e,l,t])
            labels.append(temp)
        else:
            labels.append(temp_labels)
    return pd.DataFrame({"text" : data, "labels" : labels})

def helper_dict_values(dict_list):
    return [list(d.values()) for d in dict_list]

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

def helper_snorkel_representation(df, nr_of_lfs):
    # lfs = get_rumc_sources()
    # df["Y"] = df.apply(apply_true_split, axis=1)
    # df["X"] = df["text"].apply(apply_token_split)
    # df["L"] = df["text"].apply(apply_lfs, args=(lfs,))
    Xs, Ls, Ys = df["X"].to_list(), df["L"].to_list(), df["Y"].to_list()
    L = create_token_L_mat(Xs, Ls, nr_of_lfs)
    L = np.asarray(L.astype(np.int8).todense())
    Y = helper_dict_values(Ys)
    Y = np.concatenate(Y).ravel()
    return L, Y

def replace_tags_data(df, pred):
    idx = 0
    text = df["text"].to_list()
    x_tokens = df["X"].to_list()
    all_sequences = []
    all_reports = []

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

def replace_tags_data_i2b2(df, pred):
    idx = 0
    text = df["text"].to_list()
    bio = df["bio"].to_list()
    x_tokens = df["X"].to_list()
    all_sequences = []
    all_reports = []

    for i in range(len(df)):
        elem = x_tokens[i]
        text_elem = text[i]
        selection = pred[idx:idx+len(elem)]
        temp_sequence = []
        prev_elem = -1

        # BIO tagging WS annotated labels 
        # https://pythonprogramming.net/using-bio-tags-create-named-entity-lists/    
        for s in selection:
            if s == -1:
                temp_sequence.append("O")
                prev_elem = -1
            if s != -1 and prev_elem == -1:
                temp_sequence.append("B-" + ENTITIES_DICT_i2b2[s])
                prev_elem = s
            elif s == prev_elem and prev_elem != -1:
                temp_sequence.append("I-" + ENTITIES_DICT_i2b2[s])
                prev_elem = s
            elif prev_elem != -1 and prev_elem != s: 
                temp_sequence.append("B-"+ ENTITIES_DICT_i2b2[s])
                prev_elem = s
        all_sequences.append(temp_sequence)
        idx += len(elem)

    temp_report = []
    # Create tuple: (text, POS, BIO-tag)
    for b in bio:
        for elem in b:
            temp_report.append((elem[0], elem[1]))

    for seq in all_sequences:
        for k in range(len(temp_sequence)):
            temp_report[k] = temp_report[k] + (temp_sequence[k],)
        all_reports.append(temp_report)
    return all_reports, all_sequences

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
            nlp.add_pipe("entity_component", 
                         name="placer", last=True, 
                         config={"entities":labels})
            doc = nlp(text)
            report = []
            seq = []
            for ent in doc:
                if ent.ent_iob_ == "O" or ent.ent_iob_ == "":
                    report.append((ent.text, ent.pos_, "O"))
                    seq.append("O")
                else:
                    report.append((ent.text, ent.pos_, 
                                   ent.ent_iob_ + "-" + ent.ent_type_))
                    seq.append(ent.ent_iob_ + "-" + ent.ent_type_)
            all_reports.append(report)
            all_sequences.append(seq)
            nlp.remove_pipe("placer")
    return all_reports, all_sequences