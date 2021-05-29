# Programmatically generating annotations for de-identification of clinical data
Source code for master thesis.

## Instructions
To make use of this code, create virtual environment using the following instructions:
* Create a Python environment with the right packages `pip install -r requirements.txt`.
* Folder structure should look like this (ask for access to the two datasets):
```
.
├── annotations
├── data-i2b2
│   ├── test
│   ├── test-gold
│   └── training
├── data-rumc
├── models
└── snorkel-data
```
* cd to /path/to/progrann/

## Commands Experiment 1: de-identification of clinical records
We have employed two methods to deidentify clinical records: CRF and BI-LSTM.
The following commands are useful to reproduce the results:

* To prepare i2b2 dataset: `python annotations/preprocessor.py` which creates a folder named `training-df` in `data-i2b2`.
* To train discriminative model: `python annotations/main.py --[bilstm, crf] --[i2b2, rumc] --path [./data-i2b2/training-df/, ./data-rumc/]`

## Commands Experiment 2: programmatic annotations using weak supervision
As we have seen so far, the results are good on both datasets.
The datasets at hand have human annotated labels, but can we obtain similar 
performance with programmatic annotation using weak sources of information?
The followings commands are useful to reproduce the results:

### Reproduce results:
First, apply the labeling functions on the clinical reports using: `python annotations/dive.py --[rumc, i2b2] --train` which saves the generative model to `./models/` and the snorkel training data to `./snorkel-data/`.
Now we can use the following commands:
* To train discriminative model using weakly tagged training data: `python annotations/main.py --[si2b2, srumc] --[crf, bilstm]`
* To experiment with increasing weakly tagged training data: `python annotations/main.py --[si2b2, srumc] --extra [crf, bilstm]` 
