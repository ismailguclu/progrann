from seqeval.metrics import classification_report
from features import doc2features, doc2labels
import pickle
import helper
import os
import json

DATA = "rumc"
ENGLISH = False
CONVERSION = {"B-DATUM":"B-DATE", "I-DATUM":"I-DATE", "B-PERSOON":"B-PERSON", "I-PERSOON":"I-PERSON",
              "B-TELEFOONNUMMER":"B-PHONE", "I-TELEFOONNUMMER":"I-PHONE", "B-TIJD":"B-TIME", 
              "I-TIJD":"I-TIME", "B-PLAATS":"B-LOCATION", "I-PLAATS":"I-LOCATION"}

def get_rumc_data(path):
    data = []
    with open(path, "r") as f:
        test_file = list(f)
    f.close()

    for doc in test_file:
        result = json.loads(doc)
        text, labels = result['text'], result['labels']
        new_labels = []
        for s, e, ent in labels:
            ent_text = text[s:e]
            new_labels.append((s, e, ent[1:-1], ent_text))
        data.append((text, new_labels))
    return data

# Define CRF model to load
crf_model_filename = "./final-1/crf_weights.pkl"
# model = sklearn_crfsuite.CRF(model_filename=crf_model_filename)
with open(crf_model_filename, "rb") as f:
    model = pickle.load(f)
f.close()

# Define path and load test data
if DATA == "rumc":
    test_path = "./data-rumc/500_extra_shuffled_clean.jsonl"
    test_rumc = get_rumc_data(test_path)
    bio, sequences = helper.bio_tagging(test_rumc)
else:
    test_path = "./data/test-gold-df/"
    TEST = os.listdir(test_path)
    TEST.sort()
    bio, LABELS = helper.get_i2b2_data(TEST, test_path)

# Extract features from test set
X_test = [doc2features(s) for s in bio]
y_test = [doc2labels(s) for s in bio]

if ENGLISH:
    english_y_test = []
    for y in y_test:
        temp = []
        for i in y:
            if i in CONVERSION:
                temp.append(CONVERSION[i])
            else:
                temp.append(i)
        english_y_test.append(temp)
    y_test = english_y_test

# Predict using model and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))