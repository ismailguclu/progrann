import sklearn
import pycrfsuite
import os

# Load data
TRAIN_PATH = os.getcwd() + "/data/training-df/"

for filename in os.listdir(TRAIN_PATH):
   with open(os.path.join(TRAIN_PATH, filename), 'r') as f: 
       


# if/else for train/dev/test

# add features

# train crf
# pred crf

# evaluate performance