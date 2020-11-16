from bs4 import BeautifulSoup
import spacy
import os
import pandas
from spacy.tokens import Span, Doc
from EntityPlacer import EntityPlacer
import tokenizer_regex
import special_cases
import special_merge

REGEX_LIST_PREFIX = tokenizer_regex.REGEX_LIST_PREFIX
REGEX_LIST_INFIX = tokenizer_regex.REGEX_LIST_INFIX
REGEX_LIST_SUFFIX = tokenizer_regex.REGEX_LIST_SUFFIX
SPECIAL_CASES_LIST = special_cases.SPECIAL_CASES_LIST
MERGE_DICT = special_merge.MERGE_DICT
STANDARD_PIPE = ["tagger", "parser"]

def glue_bio_type(x):
    return x[0] if x[0] == "O" else x[0] + "-" + x[1]

def glue_tokens(doc):
    doc.merge(MERGE_DICT[doc._.fn][0], MERGE_DICT[doc._.fn][1])
    return doc

fn_path = "./data/training/"
target_path = "./data/training-pandas/"
files = [f for f in os.listdir(fn_path) if os.path.isfile(os.path.join(fn_path, f))]
files.sort()

nlp = spacy.load("en_core_web_sm")
nlp.remove_pipe("ner")

# Compile new tokenizer suffix
prefix = nlp.Defaults.prefixes
for r in REGEX_LIST_PREFIX:
    prefix = prefix + (r,)
prefix_regex = spacy.util.compile_prefix_regex(prefix)
nlp.tokenizer.prefix_search = prefix_regex.search

# Compile new tokenizer infix
infix = nlp.Defaults.infixes
for r in REGEX_LIST_INFIX:
    infix = infix + (r,)
infix_regex = spacy.util.compile_infix_regex(infix)
nlp.tokenizer.infix_finditer = infix_regex.finditer

# Compile new tokenizer suffix
suffix = nlp.Defaults.suffixes
for r in REGEX_LIST_SUFFIX:
    suffix = suffix + (r,)
suffix_regex = spacy.util.compile_suffix_regex(suffix)
nlp.tokenizer.suffix_search = suffix_regex.search

for sc in SPECIAL_CASES_LIST:
    nlp.tokenizer.add_special_case(sc[0], sc[1])

counter = 0
print(files.index("256-02.xml")) 
for fn in files[378:]:
    print("---------------------------------" + fn)
    with open(fn_path + fn) as file:
        content = file.read()

    soup = BeautifulSoup(content, "xml")
    text = soup.TEXT.text
    new_text = text.lstrip("\n")
    diff_char = len(text) - len(new_text)
    entities = [(int(tag["start"]), int(tag["end"]), tag["TYPE"]) 
                for tag in soup.TAGS.find_all()]

    new_entities = []
    for start, end, label in entities:
        n_start, n_end = start-diff_char, end-diff_char
        new_entities.append((n_start, n_end, label))

    new_text = new_text.rstrip("\n")
    entity_placer = EntityPlacer(new_entities)

    #print(nlp.tokenizer.explain(new_text))
    if fn in MERGE_DICT:
        nlp.add_pipe(glue_tokens, name="glue")
        nlp.add_pipe(entity_placer, name="placer", last=True)
    else:
        nlp.add_pipe(entity_placer, name="placer", last=True)
    Doc.set_extension("fn", default=fn)
    doc = nlp(new_text)
    

    for pipe in nlp.pipe_names:
        if pipe not in STANDARD_PIPE:
            nlp.remove_pipe(pipe)
    Doc.remove_extension("fn")

    #print(fn + "Number of faulty: " + str(entity_placer.faulty))
    if entity_placer.faulty > 0:
        counter += 1
    # df = pandas.DataFrame([(ent.text, ent.ent_iob_, ent.ent_type_) 
    #                         for ent in doc], 
    #                         columns = ["Token", "BIO", "Type"])
    # df["BIO"] = df["BIO"].replace(r'^\s*$', "O", regex=True)
    # df["BIO-Type"] = df[["BIO", "Type"]].apply(glue_bio_type, axis=1) 
    # df.to_csv(target_path+fn, header=None, index=None, sep='\t')
print(counter)