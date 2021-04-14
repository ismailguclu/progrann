from snorkel.labeling import labeling_function
from dateutil.parser import parse
import re
import spacy

ENTITIES = ['HEALTHPLAN', 'LOCATION-OTHER', 'ORGANIZATION', 'DEVICE', 'STREET', 
            'CITY', 'ZIP', 'HOSPITAL', 'MEDICALRECORD', 'IDNUM', 'FAX', 'DATE', 
            'PHONE', 'COUNTRY', 'URL', 'PROFESSION', 'STATE', 'PATIENT', 'EMAIL', 
            'DOCTOR', 'BIOID', 'AGE', 'USERNAME']
CONVERSION = {'ORG':'ORGANIZATION'}

with open("./data/hospital.txt") as fn:
    HOSPITALS = fn.read().split("\n")
fn.close()

with open("./data/country.txt") as fn:
    COUNTRIES = fn.read().split("\n")
fn.close()

with open("./data/states.txt") as fn:
    STATES = fn.read().split("\n")
fn.close()

with open("./data/professions.txt") as fn:
    PROFESSIONS = fn.read().split("\n")
fn.close()

ner_1 = spacy.load("en_core_web_sm")

def is_date(date):
    "Checks whether a valid date."
    try: 
        parse(date, fuzzy=False)
        return True
    except ValueError:
        return False

def is_doctor(span):
    print(span)
    return "M.D." in span

def is_patient(name, start, length):
    third = length/3
    return start < third

@labeling_function()
def lf_date_1(doc):
    "Pattern matching: 25/12/2020, 25-12-2020 etc."
    labels = []
    dates = re.finditer(r'(\d+[/-]\d+/[\-]d+)', doc)
    for d in dates:
        s, e = d.start(), d.end()
        if is_date(doc[s:e]):
            labels.append((s, e, "DATE", doc[s:e]))
    return labels

@labeling_function()
def lf_date_2(doc):
    "Pattern matching: December 25, 2020 etc."
    labels = []
    dates = re.finditer(r'([A-Z](?:[a-zA-Z]+))(\s*[0-9]+)(,\s[0-9]+)', doc)
    for d in dates:
        s, e = d.start(), d.end()
        if is_date(doc[s:e]):
            labels.append((s, e, "DATE", doc[s:e]))
    return labels

@labeling_function()
def lf_date_3(doc):
    "Pattern matching: 25 December 2020 etc."
    labels = []
    dates = re.finditer(r'([0-9]+)(\s[A-Z](?:[a-zA-Z]+))(\s[0-9]+)', doc)
    for d in dates:
        s, e = d.start(), d.end()
        if is_date(doc[s:e]):
            labels.append((s, e, "DATE", doc[s:e]))
    return labels

@labeling_function()
def lf_hospital_1(doc):
    "Knowledge base: find matching tokens that are hospitals."
    labels = []
    for h in HOSPITALS:
        if h in doc:
            hosp = re.finditer(h, doc)
            for i in hosp:
                s, e = i.start(), i.end()
                labels.append((s, e, "HOSPITAL", doc[s:e]))
    return labels

@labeling_function()
def lf_country_1(doc):
    "Knowledge base: find matching tokens that are countries."
    labels = []
    for c in COUNTRIES:
        if c in doc:
            cnt = re.finditer(c, doc)
            for i in cnt:
                s, e = i.start(), i.end()
                labels.append((s, e, "COUNTRY", doc[s:e]))
    return labels

@labeling_function()
def lf_state_1(doc):
    "Knowledge base: find matching tokens that are US states."
    labels = []
    for s in STATES:
        if s in doc:
            cnt = re.finditer(s, doc)
            for i in cnt:
                s, e = i.start(), i.end()
                labels.append((s, e, "STATE", doc[s:e]))
    return labels    

@labeling_function()
def lf_profession_1(doc):
    "Knowledge base: find matching tokens that are professions"
    "(https://gist.github.com/wsc/1083459)"
    labels = []
    for p in PROFESSIONS:
        if p.lower() in doc:
            cnt = re.finditer(p, doc)
            for i in cnt:
                s, e = i.start(), i.end()
                labels.append((s, e, "PROFESSION", doc[s:e]))
    return labels    

@labeling_function()
def lf_model_1(doc):
    "Pre-trained model: find entities and filter on our target entities."
    d = ner_1(doc)
    labels = []
    for ent in d.ents:
        if ent.label_ in ENTITIES:
            labels.append((ent.start_char, ent.end_char, ent.label_, ent.text))
        if ent.label_ in CONVERSION:
            labels.append((ent.start_char, ent.end_char, CONVERSION[ent.label_], ent.text))
    return labels

@labeling_function()
def lf_name_1(doc):
    d = ner_1(doc)
    labels = []
    for ent in d.ents:
        print(ent)
        if ent.label_ == "PERSON":
            s, e = ent.start_char, ent.end_char
            selection = doc[s-10:e+10]
            if is_doctor(selection):
                labels.append((s, e, "DOCTOR", ent.text))
                continue
            
            if is_patient(ent.text, s, len(doc)):
                labels.append((s, e, "PATIENT", ent.text))
                continue
    return labels

@labeling_function()
def lf_age_1(doc):
    "Regular expression: find tokens that are likely an age."
    labels = []
    ages = re.finditer(r"(?<=\D)\d{2}(?=\D)", doc)
    for a in ages:
        s, e = a.start(), a.end()
        selection = doc[a:e+15]
        if "year" in selection or "yo" in selection:
            labels.append((s, e, "AGE", doc[s:e]))
    return labels

@labeling_function()
def lf_idnum_1(doc):
    labels = []
    idnums = re.finditer(r"(\d{1,2}-\d{7,})|(\w{2}\d{2,}/\d{3,})", doc)
    for ids in idnums:
        s, e = ids.start(), ids.end()
        labels.append((s, e, "IDNUM", doc[s:e]))
    return labels

@labeling_function()
def lf_mr_1(doc):
    labels = []
    mrs = re.finditer(r"(\d{7,8})|(\d{3}-\d{2}-\d{2}-\d{1})", doc)
    length = len(doc)
    for m in mrs:
        s, e = m.start(), m.end()
        if s < (length/3):
            labels.append((s, e, "MEDICALRECORD", doc[s:e]))
    return labels

@labeling_function()
def lf_mail_1(doc):
    labels = []
    mails = re.finditer(r"[A-z]+@.*$", doc)
    for m in mails:
        s, e = m.start(), m.end()
        labels.append((s, e, "EMAIL", doc[s:e]))
    return labels

@labeling_function()
def lf_phone_1(doc):
    labels = []
    numbers = re.finditer(r"\d{4,11}", doc)
    for n in numbers:
        s, e = n.start(), n.end()
        selection = doc[s-20:e]
        if "pager" in selection.lower():
            labels.append((s, e, "PHONE", doc[s:e]))
    return labels

@labeling_function()
def lf_phone_2(doc):
    labels = []
    numbers = re.finditer(r"[^\s]{0,1}\d{3}.{1,2}\d{3}.\d{4}", doc)
    for n in numbers:
        s, e = n.start(), n.end()
        labels.append((s, e, "PHONE", doc[s:e]))
    return labels

@labeling_function()
def lf_url_1(doc):
    labels = []
    urls = re.finditer(r"[www.].*$", doc)
    for u in urls:
        s, e = u.start(), u.end()
        labels.append((s, e, "URL", doc[s:e]))
    return labels

@labeling_function()
def lf_username_1(doc):
    labels = []
    usernames = re.finditer(r"\w{2}\d{2,5}", doc)
    for u in usernames:
        s, e = u.start(), u.end()
        labels.append((s, e, "USERNAME", doc[s:e]))
    return labels

@labeling_function()
def lf_street_1(doc):
    labels = []
    streets = re.finditer(r"\d{2,3}\s\w+\s\w+(?=\n)", doc)
    for st in streets:
        s, e = st.start(), st.end()
        labels.append((s, e, "STREET", doc[s:e]))
    return labels

@labeling_function()
def lf_address_1(doc):
    # ugly, fix later?
    labels = []
    addresses = re.finditer(r"((\w+\s\w+)|(\w+)),(\s{1,2}\w{2}\s{1,2}\d{5})", doc)
    for adr in addresses:
        s, e = adr.start(), adr.end()
        txt = doc[s:e]
        splits = txt.split(",")
        city = splits[0]
        labels.append((s, s+len(city), "CITY", city))
        state_zip = splits[1].split()
        state, zip = state_zip[0], state_zip[1]
        s_state = s + txt.index(state)
        labels.append((s_state, s_state+2, "STATE", state))
        s_zip = s + txt.index(zip)
        labels.append((s_zip, s_zip+5, "ZIP", zip))
    return labels

if __name__ == "__main__":
    test_string = "this is an address notation: 21 Jump Street\n Ede, NL 12345 and Colorado City,  NY  43414"
    print(lf_address_1(test_string))
    print(test_string[71:87])
    #print(lf_name_1(test_string))
    #print(lf_mail_1("something is 11/22/3333 is ismail@guclu.com"))