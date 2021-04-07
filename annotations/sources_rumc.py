import re
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
from dateutil.parser import parse
import stanza

MONTHS = ["januari", "jan", "februari", "feb", "maart", "mrt", "april", "apr", 
          "mei", "juni", "jun", "juli", "jul", "augustus", "aug", "september", 
          "sep", "oktober", "okt", "november", "nov", "december", "dec"]
ENTITIES = ["DATE", "PERSON", "TIME", "LOCATION", "PHONE", "AGE", "ZNUMMER"]
TEST_ENTS = ["DATE", "TIME", "LOCATION", "PHONE", "AGE", "ZNUMMER"]
#TITLES = ["prof", "dr", "mevr", "mw", "mr", "mevrouw", "meneer", "heer"]
ENTITIES_DICT = {"ABSTAIN":-1, "DATE":0, "PERSON":1, "TIME":2, "LOCATION":3, "PHONE":4, "AGE":5, "ZNUMMER":6}
ENTITIES_DICT_2 = {"<DATUM>":"DATE", "<PERSOON>":"PERSON", "<TIJD>":"TIME", "<PLAATS>":"LOCATION", 
                   "<TELEFOONNUMMER>":"PHONE", "<LEEFTIJD>":"AGE", "<ZNUMMER>":"ZNUMMER"}
PREFIXES = ["", "\s", " van de ", " van der ", " vd ", " van den ", " de "]
with open("./data-rumc/dutch_cities.txt") as cities:
    CITIES = cities.read().split("\n")
cities.close()
with open("./data-rumc/family_names.txt") as fn:
    FAMILY_NAMES = fn.read().split("\n")
fn.close()
with open("./data-rumc/first_names.txt") as fn:
    FIRST_NAMES = fn.read().split("\n")
fn.close()
with open("./data-rumc/titles_before.txt") as btitles:
    BTITLES = btitles.read().split("\n")
btitles.close()
with open("./data-rumc/titles_after.txt") as atitles:
    ATITLES = atitles.read().split("\n")
atitles.close()
stanza.download("nl")
ner_1 = spacy.load("nl_core_news_lg")
ner_2 = spacy.load("en_core_web_sm")
ner_3 = stanza.Pipeline(lang="nl", processors="tokenize,ner")
nlp = spacy.load("nl_core_news_lg")
nlp.remove_pipe("ner")

# https://stackoverflow.com/questions/25341945/check-if-string-has-date-any-format
def is_date(date):
    try: 
        parse(date, fuzzy=False)
        return True
    except ValueError:
        return False

# #@Language.factory("entity_component", default_config={"entities": []})
# def my_entity_component(nlp, name, entities):
#     return EntityPlacer(entities)

# Language.factory(
#     "entity_component",
#     default_config={"entities": []},
#     func=my_entity_component
# )

@labeling_function()
def lf_date_1(doc):
    dates = re.finditer(r"\d{2}[-/]\d{2}[-/]\d{4}", doc)
    labels = []
    for d in dates:
        s, e = d.start(), d.end()
        if is_date(doc[s:e]):
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
            if is_date(doc[s:e]):
                labels.append((s, e, "DATE", doc[s:e]))
    return labels

@labeling_function()
def lf_date_3(doc):
    labels = []
    for m in MONTHS:
        dates = re.finditer(m, doc)
        for d in dates:
            s, e = d.start(), d.end()
            if is_date(doc[s:e]):
                labels.append((s, e, "DATE", doc[s:e]))
    return labels

@labeling_function()
def lf_model_1(doc):
    d = ner_1(doc)
    labels = []
    for entity in d.ents:
        if entity.label_ in TEST_ENTS:
            labels.append((entity.start_char, entity.end_char, entity.label_, entity.text))
        # if entity.label_ == "GPE":
        #     labels.append((entity.start_char, entity.end_char, "LOCATION", entity.text))
    return labels

@labeling_function()
def lf_model_2(doc):
    d = ner_2(doc)
    labels = []
    for entity in d.ents:
        if entity.label_ in TEST_ENTS:
            labels.append((entity.start_char, entity.end_char, entity.label_, entity.text))
        # if entity.label_ == "GPE" and entity.text in CITIES:
        #     labels.append((entity.start_char, entity.end_char, "LOCATION", entity.text))
    return labels

@labeling_function()
def lf_model_3(docs):
    doc = ner_3(docs)
    labels = []
    for entity in doc.ents:
        # if entity.type == "PER":
        #     labels.append((entity.start_char, entity.end_char, "PERSON", entity.text))
        if entity.type == "LOC" and entity.text in CITIES:  
            labels.append((entity.start_char, entity.end_char, "LOCATION", entity.text))
    #print(*[f'entity: {ent.text}\ttype: {ent.type}' for ent in doc.ents], sep='\n')
    return labels

@labeling_function()
def lf_number_1(doc):
    labels = []
    numbers = re.finditer(r"\d{5,}", doc)
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
def lf_number_2(doc):
    labels = []
    numbers = re.finditer(r"\d{4}", doc)
    for n in numbers:
        s, e = n.start(), n.end()
        selection = doc[s-15:s]
        if "sein" in selection.lower():
            new_s = s-15+selection.lower().index("sein")
            labels.append([new_s, e, "PHONE", doc[new_s:e]])
    return labels

@labeling_function()
def lf_time_1(doc):
    labels = []
    times = re.finditer(r"\d{2}:\d{2}", doc)
    for t in times:
        s, e = t.start(), t.end()
        selection = doc[e:e+5]
        if "uur" in selection:
            new_e = e + selection.lower().index("uur") + 3
            labels.append((s, new_e, "TIME", doc[s:new_e]))
        else:
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

# @labeling_function()
# def lf_name_2(doc):
#     labels = []
#     for p in PREFIXES:
#         reg = "dr."+p+"[A-Z][a-z]+"
#         names = re.finditer(reg, doc)
#         for n in names:
#             s, e = n.start(), n.end()
#             labels.append((s, e, "PERSON", doc[s:e]))
#     return labels

# @labeling_function()
# def lf_name_2(docs):
#     doc = ner_3(docs)
#     labels = []
#     for entity in doc.ents:
#         if entity.type == "PER":
#             s, e = entity.start_char, entity.end_char
#             selection = docs[s-30:e]
#             for t in TITLES:
#                 if t in selection:
#                     s = s-30+selection.lower().index(t)
#                     break
#             labels.append((s, e, "PERSON", docs[s:e]))
#     return labels

def helper_name(docs, s, e):
    starts, ends = [s], [e]
    ts, te = helper_selection(docs, s-50, e+50)
    t_token = docs[s:e]
    selection = docs[ts:te]
    for t in BTITLES:
        if t in selection.split():
            ind_token = selection.index(t_token)
            ind_title = selection.index(t)
            if ind_title < ind_token:
                diff = ind_token - ind_title
                starts.append(s - diff)
    for t in ATITLES:
        if t in selection.split():
            ind_token = selection.index(t_token)+len(t_token)
            ind_title = selection.index(t)+len(t)
            if ind_title > ind_token:
                diff = ind_title - ind_token
                ends.append(diff + e)
    return min(starts), max(ends)

def helper_selection(text, s, e):
    new_s = 0 if s < 0 else s
    lnt = len(text)
    new_e = lnt if e > lnt else e
    return new_s, new_e

@labeling_function()
def lf_name_2(docs):
    doc = ner_3(docs)
    labels = []
    for entity in doc.ents:
        if entity.type == "PER":
            for ent in entity.text.split():
                if (ent in FAMILY_NAMES) or (ent in FIRST_NAMES):
                    s, e = entity.start_char, entity.end_char
                    new_s, new_e = helper_name(docs, s, e) 
                    labels.append((new_s,new_e,'PERSON',docs[new_s:new_e]))
    return labels

@labeling_function()
def lf_name_3(docs):
    doc = ner_3(docs)
    labels = []
    for entity in doc.ents:
        if entity.type == "PER":
            for ent in entity.text.split():
                s, e = entity.start_char, entity.end_char
                new_s, new_e = helper_name(docs, s, e) 
                labels.append((new_s,new_e,'PERSON',docs[new_s:new_e]))
    return labels

def get_rumc_sources():
    lf = [lf_date_1,
          lf_date_2,
          lf_date_3,
          lf_model_1,
          #lf_model_2,
          lf_model_3,
          #lf_name_1,
          lf_name_2,
          #lf_name_3,
          lf_number_1,
          lf_number_2,
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

def cast_majority_vote(df):
    L, _, _, df = helper_snorkel_representation(df)
    majority_predictions = []
    for l in L:
        if np.all(l == -1):
            majority_predictions.append(-1)
        else:
            non_abstain_votes = np.where(l != -1)[0]
            vote = np.bincount(non_abstain_votes).argmax()
            majority_predictions.append(vote)
    assert len(L) == len(majority_predictions)
    return np.array(majority_predictions)

def helper_snorkel_representation(df):
    lfs = get_rumc_sources()
    df["Y"] = df.apply(apply_true_split, axis=1)
    df["X"] = df["text"].apply(apply_token_split)
    df["L"] = df["text"].apply(apply_lfs, args=(lfs,))
    Xs, Ls, Ys = df["X"].to_list(), df["L"].to_list(), df["Y"].to_list()
    L = create_token_L_mat(Xs, Ls, len(lfs))
    L = np.asarray(L.astype(np.int8).todense())
    Y = helper_dict_values(Ys)
    Y = np.concatenate(Y).ravel()
    return L, Y, lfs, df

def validate(df, label_model):
    L, Y, lfs, df = helper_snorkel_representation(df)
    label_model_acc = label_model.score(L=L, Y=Y, tie_break_policy="random")["accuracy"]
    print(f"{'Label Model Accuracy (Validation):':<25} {label_model_acc * 100:.1f}%")
    return

def train(df, gold_labels=False, mv=False):
    L, Y, lfs, df = helper_snorkel_representation(df)
    label_model = LabelModel(cardinality=len(ENTITIES)+1)
    label_model.fit(L)
    if gold_labels:
        print(LFAnalysis(L=L, lfs=lfs).lf_summary(Y=Y))
        print(label_model.score(L, Y=Y, metrics=["f1_micro"]))
        label_model_acc = label_model.score(L=L, Y=Y, tie_break_policy="random")["accuracy"]
        print(f"{'Label Model Accuracy (Development):':<25} {label_model_acc * 100:.1f}%")

    if mv:
        Y = cast_majority_vote(df)
        print(LFAnalysis(L=L, lfs=lfs).lf_summary(Y=Y))
        print(label_model.score(L, Y=Y, metrics=["f1_micro"]))
        label_model_acc = label_model.score(L=L, Y=Y, tie_break_policy="random")["accuracy"]
        print(f"{'Majority Vote Accuracy (Development):':<25} {label_model_acc * 100:.1f}%")
    return label_model, df