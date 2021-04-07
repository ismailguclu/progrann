from seqeval.metrics import classification_report
from features import doc2features, doc2labels
import sklearn_crfsuite
import pickle
import helper
import os

# Define CRF model to load
crf_model_filename = "./final-2/crf_weights.pkl"
# model = sklearn_crfsuite.CRF(model_filename=crf_model_filename)
with open(crf_model_filename, "rb") as f:
    model = pickle.load(f)
f.close()

# Define path and load test data
test_path = "./data/test-gold-df/"
TEST = os.listdir(test_path)
TEST.sort()
bio, LABELS = helper.get_i2b2_data(TEST, test_path)

# Extract features from test set
X_test = [doc2features(s) for s in bio]
y_test = [doc2labels(s) for s in bio]

# Predict using model and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))
print("BIG SUCCESS")