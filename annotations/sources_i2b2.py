from snorkel.labeling import labeling_function
from dateutil.parser import parse
import re
import preprocessor
import spacy
import geograpy

ENTITIES = ['HEALTHPLAN', 'LOCATION-OTHER', 'ORGANIZATION', 'DEVICE', 'STREET', 
            'CITY', 'ZIP', 'HOSPITAL', 'MEDICALRECORD', 'IDNUM', 'FAX', 'DATE', 
            'PHONE', 'COUNTRY', 'URL', 'PROFESSION', 'STATE', 'PATIENT', 'EMAIL', 
            'DOCTOR', 'BIOID', 'AGE', 'USERNAME']
ENTITIES_DICT = {'ABSTAIN':-1, 'HEALTHPLAN':0, 'LOCATION-OTHER':1, 'ORGANIZATION':2, 
                 'DEVICE':3, 'STREET':4, 'CITY':5, 'ZIP':6, 'HOSPITAL':7, 
                 'MEDICALRECORD':8, 'IDNUM':9, 'FAX':10, 'DATE':11, 'PHONE':12, 
                 'COUNTRY':13, 'URL':14, 'PROFESSION':15, 'STATE':16, 'PATIENT':17, 
                 'EMAIL':18, 'DOCTOR':19, 'BIOID':20, 'AGE':21, 'USERNAME':24}
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

# https://raw.githubusercontent.com/grammakov/USA-cities-and-states/master/us_cities_states_counties.csv
# with open("./data/us_states_cities.txt") as fn:
#     lines = fn.readlines()
#     STATES = []
#     for item in lines:
#         STATES.append(item.split("|")[0])
# fn.close()

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
    return "M.D." in span or "R.N." in span

def is_patient(name, start, length):
    third = length/3
    return start < third

@labeling_function()
def lf_date_1(doc):
    "Pattern matching: 25/12/2020, 25-12-2020 etc."
    labels = []
    dates = re.finditer(r'\d+[/-]\d+[/-]\d+', doc)
    for d in dates:
        s, e = d.start(), d.end()
        if is_date(doc[s:e]):
            labels.append((s, e, "DATE", doc[s:e]))
    return labels

@labeling_function()
def lf_date_2(doc):
    "Pattern matching: December 25, 2020 etc."
    labels = []
    dates = re.finditer(r'([A-Z](?:[a-zA-Z]+)\s*\d{1,2},\s[0-9]+)', doc)
    for d in dates:
        s, e = d.start(), d.end()
        if is_date(doc[s:e]):
            labels.append((s, e, "DATE", doc[s:e]))
    return labels

@labeling_function()
def lf_date_3(doc):
    "Pattern matching: 25 December 2020 etc."
    labels = []
    dates = re.finditer(r'(\d{1,2}\s[A-Z](?:[a-zA-Z]+)\s\d{4})', doc)
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
        if h in doc and len(h) > 3:
            reg = "(?<=\s)" + h + "(?=\S|\s)"
            hosp = re.finditer(reg, doc)
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
    for st in STATES:
        if st in doc:
            reg = "(?<=\s)" + st + "(?=\s)"
            cnt = re.finditer(reg, doc)
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
        # if ent.label_ in CONVERSION:
        #     labels.append((ent.start_char, ent.end_char, CONVERSION[ent.label_], ent.text))
    return labels

@labeling_function()
def lf_name_1(doc):
    d = ner_1(doc)
    labels = []
    for ent in d.ents:
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
        selection = doc[s:e+15]
        if "year" in selection or "yo" in selection:
            labels.append((s, e, "AGE", doc[s:e]))
    return labels

@labeling_function()
def lf_idnum_1(doc):
    labels = []
    idnums = re.finditer(r"(\d{1,2}-\d{7,})|([A-z]{2}\d{2,}/\d{3,})", doc)
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
    # https://www.geeksforgeeks.org/python-check-url-string/
    labels = []
    reg = "(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    urls = re.finditer(reg, doc)
    for u in urls:
        s, e = u.start(), u.end()
        labels.append((s, e, "URL", doc[s:e]))
    return labels

@labeling_function()
def lf_username_1(doc):
    labels = []
    usernames = re.finditer(r"(?<=[\s\[.,])[A-z]{2}\d{2,3}(?=[\s\].,])", doc)
    length = len(doc)
    for u in usernames:
        s, e = u.start(), u.end()
        if s > length*0.8:
            labels.append((s, e, "USERNAME", doc[s:e]))
    return labels

@labeling_function()
def lf_street_1(doc):
    labels = []
    streets = re.finditer(r"\d{2,4}\s\w+\s\w+(?=\n)", doc)
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
        state, zip_code = state_zip[0], state_zip[1]
        s_state = s + txt.index(state)
        labels.append((s_state, s_state+2, "STATE", state))
        s_zip = s + txt.index(zip_code)
        labels.append((s_zip, s_zip+5, "ZIP", zip_code))
    return labels

@labeling_function()
def lf_address_2(doc):
    labels = []
    places = geograpy.get_geoPlace_context(text=doc)
    for city in places.cities:
        s = doc.find(city)
        if s != -1:
            labels.append((s, s+len(city), "CITY", city))
    
    for country in places.countries:
        s = doc.find(country)
        if s != -1:
            labels.append((s, s+len(country), "COUNTRY", country))
    return labels

@labeling_function()
def lf_city_1(doc):
    labels = []
    for c in CITIES:
        if c in doc:
            cnt = re.finditer(c, doc)
            for i in cnt:
                s, e = i.start(), i.end()
                labels.append((s, e, "CITY", doc[s:e]))
    return labels

def get_i2b2_sources():
    lf = [lf_date_1,
          lf_date_2,
          #lf_date_3,
          lf_hospital_1,
          lf_country_1,
          lf_state_1,
          lf_profession_1,
          lf_model_1,
          lf_name_1,
          lf_age_1,
          lf_idnum_1,
          lf_mr_1,
          lf_mail_1,
          lf_phone_1,
          lf_phone_2,
          lf_url_1,
          lf_username_1,
          lf_street_1,
          lf_address_1,
          lf_address_2
          #lf_city_1
    ]
    return lf

def helper_doc2token(bio):
    temp = []
    for tok in bio:
        if tok[3] == '':
            temp.append((tok[0], 'ABSTAIN'))
        else:
            temp.append((tok[0], tok[3]))
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
        token_ent_list = preprocessor.run_token_ent(text, labels_list) # REDO THIS LINE ONLY
        token_ent_dict = helper_token2dict(token_ent_list)
        temp.append(token_ent_dict)
    return temp

def apply_token_split(bio):
    return [tok[0] for tok in bio]
        
def apply_true_split(df):
    token_ent_list = helper_doc2token(df["bio"])
    token_ent_dict = helper_token2dict(token_ent_list)
    return token_ent_dict

if __name__ == "__main__":
    test_string = "this is an address notation: 21 Jump Street\n Ede, NL 12345 and Colorado City,  NY  43414"
    print(lf_address_1(test_string))
    print(test_string[71:87])
    #print(lf_name_1(test_string))
    #print(lf_mail_1("something is 11/22/3333 is ismail@guclu.com"))