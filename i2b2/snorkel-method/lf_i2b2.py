from snorkel.labeling import labeling_function
from snorkel.preprocess.nlp import SpacyPreprocessor
import re

CONVERSION = {'ABSTAIN':-1, 'DATE':0}
spacy = SpacyPreprocessor(text_field="text", doc_field="doc", memoize=True)

@labeling_function()
def lf_date_1(x):
    "Pattern matching: 25/12/2020, 25-12-2020 etc."
    matches = []
    for match in re.finditer(r'(\d+[/-]\d+/[\-]d+)', x):
        matches.append((match.start(), match.end(), match.group(0), CONVERSION['DATE']))
    return matches if matches else CONVERSION['ABSTAIN']

@labeling_function()
def lf_date_2(x):
    "Pattern matching: December 25, 2020 etc."
    matches = []
    for match in re.finditer(r'([A-Z](?:[a-zA-Z]+))(\s*[0-9]+)(,\s[0-9]+)', x):
        matches.append((match.start(), match.end(), match.group(0), CONVERSION['DATE']))
    return matches if matches else CONVERSION['ABSTAIN']

@labeling_function()
def lf_date_3(x):
    "Pattern matching: 25 December 2020 etc."
    matches = []
    for match in re.finditer(r'([0-9]+)(\s[A-Z](?:[a-zA-Z]+))(\s[0-9]+)', x):
        matches.append((match.start(), match.end(), match.group(0), CONVERSION['DATE']))
    return matches if matches else CONVERSION['ABSTAIN']

# @labeling_function(pre=[spacy])
# def lf_hospital(x):
#     return


if __name__ == "__main__":
    test_string = "this is a date notation: 11/11/2020 and this 12/11/2020"
    print(lf_date_1(test_string))
    print(lf_date_2("cutting-edge research on December 25, 2020"))