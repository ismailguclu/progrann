import os
import json
from EntityPlacer import EntityPlacer
import spacy
from spacy.tokens import Doc

DATA = []
STANDARD_PIPE = ["tagger", "parser"]
path = os.getcwd() + "/data-rumc/"

with open(path + "anon_ground_truth_sample.jsonl", "r") as f:
    test_file = list(f)

# Extract the actual text and the PHI
for doc in test_file:
    result = json.loads(doc)
    DATA.append((result['text'], result['labels']))

nlp = spacy.load("en_core_web_sm")
nlp.remove_pipe("ner")
for text, labels in DATA:
    entities = []
    for start, end, ent_type in labels:
        ent_text = text[start:end]
        entities.append((start, end, ent_type, ent_text))

    entity_placer = EntityPlacer(entities)
    nlp.add_pipe(entity_placer, name="placer", last=True)

    Doc.set_extension("fn", default="test")
    doc = nlp(text)
    
    print(doc.ents)
    for pipe in nlp.pipe_names:
        if pipe not in STANDARD_PIPE:
            nlp.remove_pipe(pipe)
    Doc.remove_extension("fn")
    