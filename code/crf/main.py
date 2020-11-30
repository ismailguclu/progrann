"""
A baseline CRF model to evaluate the performance of a NER on the 
i2b2 dataset for de-identification. This file is mainly build 
upon the tutorial in the documentations: 
https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html

Result: 0.8513459946541471 for feature set with standard tutorial.
Best parameters:  {'c1': 0.04655479644991304, 'c2': 0.007784780059581591}
"""
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
import argparse

# Load data files/path
TRAIN_PATH = os.getcwd() + "/data/training-df/"
TEST_PATH = os.getcwd() + "/data/test-gold-df/"
MODEL_PATH = os.getcwd() + "/model/"
TRAIN = os.listdir(TRAIN_PATH)
TRAIN.sort()
TEST = os.listdir(TEST_PATH)
TEST.sort()

def get_arguments():
   parser = argparse.ArgumentParser(description="Train/test CRF model on i2b2 data.")
   parser.add_argument("--train", help="train/cross-validate model", action="store_true")
   parser.add_argument("--test", help="test the model", action="store_true")
   args = parser.parse_args()
   return args

def get_data(data, data_path):
   tmp_data = []
   for filename in data:
      tmp_file = []
      with open(os.path.join(data_path, filename), newline="") as f:
         reader = csv.reader(f, delimiter="\t")
         for r in reader:
            tmp_file.append((r[0], r[1], r[4]))
      tmp_data.append(tmp_file)
   return tmp_data

def get_features(data):
   x = [doc2features(s) for s in data]
   y = [doc2labels(s) for s in data]
   return (x, y)

def train_model(X_train, y_train, k=3, n_iter=5):
   crf = sklearn_crfsuite.CRF(
      algorithm='lbfgs',
      c1=0.1,
      c2=0.1,
      max_iterations=100,
      all_possible_transitions=True
   )

   params_space = {
    'c1': scipy.stats.expon(scale=0.5),
    'c2': scipy.stats.expon(scale=0.05),
   }

   f1_scorer = make_scorer(metrics.flat_f1_score,
                           average='weighted')

   rs = RandomizedSearchCV(crf, params_space,
                           cv=k,
                           verbose=1,
                           n_jobs=-1,
                           n_iter=n_iter,
                           scoring=f1_scorer)
   rs.fit(X_train, y_train)
   crf = rs.best_estimator_
   print("Best parameters: ", rs.best_params_)
   print("Best f1-score: ", rs.best_score_)

   with open(MODEL_PATH + "/final_crf_model.pkl", "wb") as f:
      pickle.dump(crf, f)

   return crf

def test_model(crf, X_test, y_test):
   labels = list(crf.get_classes_)
   labels.remove("O")
   y_pred = crf.predict(X_test)
   print(metrics.flat_f1_score(y_test, y_pred,
                        average='weighted', labels=labels))
   return

if __name__ == "__main__":
   args = get_arguments()

   if args.train:
      train_data = get_data(TRAIN, TRAIN_PATH)
      X_train, y_train = get_features(train_data)
      crf = train_model(X_train, y_train)

   if args.test:
      test_data = get_data(TEST, TEST_PATH)
      X_test, y_test = get_features(test_data)

      with open(MODEL_PATH + "final_crf_model.pkl", "rb") as f:
         crf = pickle.load(f)
         
      test_model(crf, X_test, y_test, labels)