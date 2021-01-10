import os
import json
import spacy
from EntityPlacer import EntityPlacer
import models

STANDARD_PIPE = ["tagger", "parser"]
nlp = spacy.load("nl_core_news_lg")
nlp.remove_pipe("ner")

def bio_tagging(data):
    all_reports = []
    for text, labels in data:
        entity_placer = EntityPlacer(labels)
        nlp.add_pipe(entity_placer, name="placer", last=True)
        doc = nlp(text)
        report = []
        for ent in doc:
            if ent.ent_iob_ == "":
                report.append((ent.text, ent.pos_, "O"))
            else:
                report.append((ent.text, ent.pos_, ent.ent_iob_ + "-" + ent.ent_type_))
        all_reports.append(report)
        nlp.remove_pipe("placer")
    return all_reports

def get_data():
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

def evaluate(data, bio_data):
    # spacy_method = models.SpacyBaseline("spacy")
    # spacy_method.label(data)
    # crf = models.CRF("crf")
    # crf.validate(bio_data)
    bilstm = models.BILSTM("bilstm")
    bilstm.train(bio_data)
    return

if __name__ == "__main__":
    data = get_data()
    bio_data = bio_tagging(data)
    print(bio_data)
    evaluate(data, bio_data)
