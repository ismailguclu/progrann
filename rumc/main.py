import os
import json
import spacy
import argparse
import csv
from EntityPlacer import EntityPlacer
import models
from seqeval.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer

STANDARD_PIPE = ["tagger", "parser"]
nlp = spacy.load("nl_core_news_lg")
nlp.remove_pipe("ner")
LABELS = ["B-PERSOON", "I-PERSOON", "B-DATUM", "I-DATUM", "O", "B-TIJD", "I-TIJD",
          "B-TELEFOONNUMMER", "I-TELEFOONNUMMER", "B-PATIENTNUMMER", "I-PATIENTNUMMER",
          "B-ZNUMMER", "I-ZNUMMER", "B-LEEFTIJD", "I-LEEFTIJD", "B-PLAATS", "I-PLAATS"]
TRAIN_PATH = os.getcwd() + "/data/training-df/"
TEST_PATH = os.getcwd() + "/data/test-gold-df/"
MODEL_PATH = os.getcwd() + "/model/"
TRAIN = os.listdir(TRAIN_PATH)
TRAIN.sort()
TEST = os.listdir(TEST_PATH)
TEST.sort()

def get_i2b2_data(data, data_path):
   tmp_data = []
   for filename in data:
      tmp_file = []
      with open(os.path.join(data_path, filename), newline="") as f:
         reader = csv.reader(f, delimiter="\t")
         for r in reader:
            tmp_file.append((r[0], r[1], r[4]))
      tmp_data.append(tmp_file)
   return tmp_data

def get_i2b2_sequences(data):
    all_sequences = []
    for doc in data:
        tmp = []
        for ent in doc:
            tmp.append(ent[2])
        all_sequences.append(tmp)
    return all_sequences

def bio_tagging(data):
    all_reports = []
    all_sequences = []
    for text, labels in data:
        entity_placer = EntityPlacer(labels)
        nlp.add_pipe(entity_placer, name="placer", last=True)
        doc = nlp(text)
        report = []
        seq = []
        for ent in doc:
            if ent.ent_iob_ == "":
                report.append((ent.text, ent.pos_, "O"))
                seq.append("O")
            else:
                report.append((ent.text, ent.pos_, ent.ent_iob_ + "-" + ent.ent_type_))
                seq.append(ent.ent_iob_ + "-" + ent.ent_type_)
        all_reports.append(report)
        all_sequences.append(seq)
        nlp.remove_pipe("placer")
    return all_reports, all_sequences

def get_rumc_data():
    data = []
    path = os.getcwd() + "/data-rumc/"

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

def evaluate(sequences, gold, labels=LABELS, per_entity=True):
    if per_entity:
        print(classification_report(gold, sequences))
    # nr_of_tokens = len(gold)
    # for i in range(nr_of_tokens):


    # y_pred = MultiLabelBinarizer(classes=LABELS).fit_transform(sequences)
    # y_true = MultiLabelBinarizer(classes=LABELS).fit_transform(gold)
    # print(y_pred)
    # print(y_true)
    # if per_label:

    #     output = precision_recall_fscore_support(y_true, y_pred, average='weighted', labels=labels)
    # else:
    #     output = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    # return output

def args_parser():
    parser = argparse.ArgumentParser(description='Run models on selected data.')
    parser.add_argument("--rumc", action="store_true", help="Load RUMC data.")
    parser.add_argument("--i2b2", action="store_true", help="Load i2b2 data.")
    parser.add_argument("--crf", action="store_true", help="Load CRF model.")
    parser.add_argument("--bilstm", action="store_true", help="Load bi-LSTM model.")
    return parser.parse_args()

def main(args):
    if args.i2b2:
        bio = get_i2b2_data(TRAIN, TRAIN_PATH)
        sequences = get_i2b2_sequences(bio)

    if args.rumc:
        rumc = get_rumc_data()
        bio, sequences= bio_tagging(rumc)

    if args.crf:
        crf = models.CRF("crf")
        crf.train(bio)
        crf_y_pred = crf.label(bio)      
        print(evaluate(crf_y_pred, sequences))  

    if args.bilstm:
        bilstm = models.BILSTM("bilstm")
        bilstm_y_pred = bilstm.train(bio)    
        print(evaluate(bilstm_y_pred, sequences)) 

if __name__ == "__main__":
    main(args_parser())