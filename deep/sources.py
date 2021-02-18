import re
import snorkel
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

MONTHS = ["januari", "jan", "februari", "feb", "maart", "mrt", "april", "apr", 
          "mei", "juni", "jun", "juli", "jul", "augustus", "aug", "september", 
          "sep", "oktober", "okt", "november", "nov", "december", "dec"]
ENTITIES = ["DATE", "PERSON", "TIME", "LOCATION", "PHONE", "AGE", "ZNUMMER"]
ENTITIES_DICT = {"ABSTAIN":-1, "DATE":0, "PERSON":1, "TIME":2, "LOCATION":3, "PHONE":4, "AGE":5, "ZNUMMER":6}
ENTITIES_DICT_2 = {"<DATUM>":"DATE", "<PERSOON>":"PERSON", "<TIJD>":"TIME", "<PLAATS>":"LOCATION", 
                   "<TELEFOONNUMMER>":"PHONE", "<LEEFTIJD>":"AGE", "<ZNUMMER>":"ZNUMMER"}
PREFIXES = ["\s", " van de ", " van der ", " vd ", " van den ", " de "]
ner_1 = spacy.load("nl_core_news_lg")
nlp = spacy.load("nl_core_news_lg")
nlp.remove_pipe("ner")

@Language.factory("entity_component")
def my_entity_component(nlp, name, entities):
    return EntityPlacer(entities)

@labeling_function()
def lf_date_1(doc):
    dates = re.finditer(r"\d{2}[-/]\d{2}[-/]\d{4}", doc)
    labels = []
    for d in dates:
        s, e = d.start(), d.end()
        labels.append((s, e, "DATE", doc[s:e]))
    return labels

@labeling_function()
def lf_date_2(doc):
    labels = []
    for m in MONTHS:
        reg = "\d{2}\s"+m+"\s\d{4}"
        dates = re.finditer(reg, doc)
        for d in dates:
            s, e = d.start(), d.end()
            labels.append((s, e, "DATE", doc[s:e]))
    return labels

@labeling_function()
def lf_date_3(doc):
    labels = []
    for m in MONTHS:
        dates = re.finditer(m, doc)
        for d in dates:
            s, e = d.start(), d.end()
            labels.append((s, e, "DATE", doc[s:e]))
    return labels

@labeling_function()
def lf_model_1(doc):
    d = ner_1(doc)
    labels = []
    for entity in d.ents:
        if entity.label_ in ENTITIES:
            labels.append((entity.start_char, entity.end_char, entity.label_, entity.text))
    return labels

@labeling_function()
def lf_number_1(doc):
    labels = []
    numbers = re.finditer(r"[\d{3}]+", doc)
    for n in numbers:
        s, e = n.start(), n.end()
        if s-15 < 0:
            s = 0
        if e+15 > len(doc):
            e = len(doc)
        word_number = doc.find("nummer", s-15, e-15)
        if word_number != -1:
            labels.append([s, e, "PHONE", doc[s:e]])
    return labels

@labeling_function()
def lf_time_1(doc):
    labels = []
    times = re.finditer(r"\d{2}:\d{2}", doc)
    for t in times:
        s, e = t.start(), t.end()
        labels.append((s, e, "TIME", doc[s:e]))
    return labels

@labeling_function()
def lf_name_1(doc):
    labels = []
    for p in PREFIXES:
        reg = "[A-Z][a-z]+"+p+"[A-Z][a-z]+"
        names = re.finditer(reg, doc)
        for n in names:
            s, e = n.start(), n.end()
            labels.append((s, e, "PERSON", doc[s:e]))
    return labels

def get_rumc_data():
    data = []
    labels = []

    with open("./data-rumc/anon_ground_truth_v3_surrogates.jsonl", "r") as f:
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

def get_rumc_sources():
    lf = [lf_date_1,
          lf_date_2,
          lf_date_3,
          lf_model_1,
          lf_name_1,
          lf_number_1,
          lf_time_1
    ]
    return lf

def helper_doc2token(text, labels):
    temp = []
    nlp.add_pipe("entity_component", name="placer", last=True, config={"entities":labels})
    doc = nlp(text)
    for tok in doc:
        if tok.ent_type_:
            temp.append((tok, tok.ent_type_))
        else:
            temp.append((tok, "ABSTAIN")) # or simply "O"?
    nlp.remove_pipe("placer")
    return temp

def helper_token2dict(tokens):
    i = 0
    temp = {}
    for _,lab in tokens:
        temp[i] = ENTITIES_DICT[lab]
        i += 1
    return temp

def helper_dict_values(dict_list):
    return [list(d.values()) for d in dict_list]

def apply_lfs(text, lfs):
    temp = []
    for lf in lfs:
        labels_list = lf(text)
        token_ent_list = helper_doc2token(text, labels_list)
        token_ent_dict = helper_token2dict(token_ent_list)
        temp.append(token_ent_dict)
    return temp

def apply_token_split(text):
    return [token.text for token in ner_1(text)]
        
def apply_true_split(df):
    token_ent_list = helper_doc2token(df["text"], df["labels"])
    token_ent_dict = helper_token2dict(token_ent_list)
    return token_ent_dict

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

def snorkel_label_model(Xs, Ls, Ys, lfs):
    L = create_token_L_mat(Xs, Ls, len(lfs))
    L = np.asarray(L.astype(np.int8).todense())
    Y = helper_dict_values(Ys)
    Y = np.concatenate(Y).ravel()
    label_model = LabelModel(cardinality=len(lfs))
    label_model.fit(L)
    print(LFAnalysis(L=L, lfs=lfs).lf_summary())
    return label_model.score(L, Y=Y, metrics=["f1_micro"])

def run(df):
    lfs = get_rumc_sources()
    df["Y"] = df.apply(apply_true_split, axis=1)
    df["X"] = df["text"].apply(apply_token_split)
    df["L"] = df["text"].apply(apply_lfs, args=(lfs,))
    print(df[["Y", "L"]].head())
    score = snorkel_label_model(df["X"].to_list(), df["L"].to_list(), df["Y"].to_list(), lfs)
    print(score)
    return
    # applier = PandasLFApplier(lfs)
    # print(applier.apply(dev))
    
if __name__ == "__main__":
    df = get_rumc_data()
    dev = df.sample(frac=0.1, random_state=42)
    print(dev.head())
    run(dev)