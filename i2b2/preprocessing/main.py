from bs4 import BeautifulSoup
import spacy
import os
import pandas
from spacy.tokens import Span, Doc
from EntityPlacer import EntityPlacer
import tokenizer_regex
import special_cases
import special_merge
from spacy.language import Language

REGEX_LIST_PREFIX = tokenizer_regex.REGEX_LIST_PREFIX
REGEX_LIST_INFIX = tokenizer_regex.REGEX_LIST_INFIX
REGEX_LIST_SUFFIX = tokenizer_regex.REGEX_LIST_SUFFIX
SPECIAL_CASES_LIST = special_cases.SPECIAL_CASES_LIST
MERGE_DICT = special_merge.MERGE_DICT
STANDARD_PIPE = ["tagger", "parser"]
SPECIALIZE = True

def glue_bio_type(x):
    return x[0] if x[0] == "O" else x[0] + "-" + x[1]

def glue_tokens(doc):
    doc.merge(MERGE_DICT[doc._.fn][0], MERGE_DICT[doc._.fn][1])
    return doc

@Language.factory("entity_component")
def my_entity_component(nlp, name, entities):
    return EntityPlacer(entities)

# Initialize path to training data and assign target folder
fn_path = "./data/training/"
target_path = "./data/training-df-plain/"
files = [f for f in os.listdir(fn_path) if os.path.isfile(os.path.join(fn_path, f))]
files.sort()

nlp = spacy.load("en_core_web_sm")
nlp.remove_pipe("ner")

# If specialize, then add very specific rules to split tokens.
# This is, split tokens with a certain pattern in this particular
# dataset, for example: 12June --> 12 June.
if SPECIALIZE:
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

problem_files = ["180-03.xml", "256-02.xml", "200-04.xml", "218-02.xml"]
for fn in files:
    if fn not in problem_files:
        with open(fn_path + fn) as file:
            content = file.read()

        soup = BeautifulSoup(content, "xml")
        text = soup.TEXT.text
        new_text = text.lstrip("\n")
        diff_char = len(text) - len(new_text)
        entities = [(int(tag["start"]), int(tag["end"]), tag["TYPE"], tag["text"]) 
                    for tag in soup.TAGS.find_all()]

        new_entities = []
        for start, end, label, e_text in entities:
            n_start, n_end = start-diff_char, end-diff_char
            new_entities.append((n_start, n_end, label, e_text))

        new_text = new_text.rstrip("\n")
        #entity_placer = EntityPlacer(new_entities)

        if fn in MERGE_DICT:
            nlp.add_pipe(glue_tokens, name="glue")
            nlp.add_pipe("entity_component", name="placer", last=True, config={"entities":new_entities})
        else:
            nlp.add_pipe("entity_component", name="placer", last=True, config={"entities":new_entities})
        Doc.set_extension("fn", default=fn)
        doc = nlp(new_text)
        
        # why this?
        nlp.remove_pipe("placer")
        Doc.remove_extension("fn")

        df = pandas.DataFrame([(ent.text, ent.pos_, ent.ent_iob_, ent.ent_type_) 
                                for ent in doc], 
                                columns = ["Token", "POS", "BIO", "Type"])
        df["BIO"] = df["BIO"].replace(r'^\s*$', "O", regex=True)
        df["BIO-Type"] = df[["BIO", "Type"]].apply(glue_bio_type, axis=1) 
        target = fn.split(".")[0]
        df.to_csv(target_path+target, header=None, index=None, sep='\t')