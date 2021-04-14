# Programmatically generating annotations for clinical data
Source code for master thesis.

## Instructions
* Create a Python environment with the right packages `pip install -r requirements.txt`.
* Folder structure should look like this (ask for access to the two datasets):
```
.
├── deep
├── i2b2
│   ├── crf
│   ├── preprocessing
│   └── snorkel-method
├── rumc
└── tests
```
* cd to /path/to/master-thesis/
* ....


## Commands Experiment 1: de-identification of clinical records
The first experiment is about comparing state-of-the-art methods to de-identify 
medical records.
We have employed two methods: CRF and BI-LSTM.
The following commands are useful to reproduce the results:

[list of commands]

[table of main results]

## Commands Experiment 2: programmatic annotations using weak supervision
As we have seen so far, the results are good on both datasets.
The datasets at hand have human annotated labels, but can we obtain similar 
performance with programmatic annotation using weak sources of information?
The followings commands are useful to reproduce the results:

[list of commands]