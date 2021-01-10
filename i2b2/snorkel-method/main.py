import snorkel
import pandas
import os
import numpy as np
import spacy
from bs4 import BeautifulSoup

nlp = spacy.load("en_core_web_sm")
problem_files = ["180-03.xml", "256-02.xml", "200-04.xml", "218-02.xml"]
BASES = ['HOSPITAL']

def make_knowledge_base(entities, target):
    tmp = set()
    for ent in entities:
        for s,e,l,t in ent:
            if l == target:
                tmp.add(t)     
    return tmp

def split_train_data(data):
    np.random.seed(42)
    msk = np.random.rand(len(data)) < 0.8
    return data[msk], data[~msk]

def load_data(dir_name):
    path = os.getcwd() + dir_name
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    files.sort()

    filenames, all_text, all_entities, all_sents = [], [], [], []
    for fn in files:
        if fn not in problem_files:
            with open(path + fn) as file:
                content = file.read()

            soup = BeautifulSoup(content, "xml")
            c_text = soup.TEXT.text
            new_text = c_text.lstrip("\n")
            diff_char = len(c_text) - len(new_text)
            entities = [(int(tag["start"]), int(tag["end"]), tag["TYPE"], tag["text"]) 
                        for tag in soup.TAGS.find_all()]

            new_entities = []
            for start, end, label, e_text in entities:
                n_start, n_end = start-diff_char, end-diff_char
                new_entities.append((n_start, n_end, label, e_text))

            new_text = new_text.rstrip("\n")
            doc = nlp(new_text)
            sents = [s.string.strip() for s in doc.sents]
            filenames.append(fn), all_text.append(new_text), all_entities.append(new_entities), all_sents.append(sents)

    df = pandas.DataFrame({"filename": filenames, "records": all_text, 
                           "entities": all_entities, "sentences": all_sents})
    return df.set_index("filename")

if __name__ == "__main__":
    bases = True
    
    # load data
    df_train, df_test = load_data("/data/training/"), load_data("/data/test-gold/")
    print(df_train.head())
    print(len(df_train))

    if bases:
        for t in BASES:
            base = make_knowledge_base(df_train['entities'].tolist(), t)
            with open(t.lower() + ".txt", "w") as f:
                for b in base:
                    f.write(b + "\n")
            f.close()



    df_train, df_val = split_train_data(df_train)
    print(len(df_train))
    print(len(df_val))
    # represent data in Candidate objects in Snorkel
    # apply LFs on spans in a candidate doc
    # train end model.