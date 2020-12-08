from nltk.stem import WordNetLemmatizer 
  
CONTEXT = 2
lemmatizer = WordNetLemmatizer() 

def helper_word(word, doc, i, j, features, before):
    if before:
        pre_word = doc[i-j][0]
        pre_postag = doc[i-j][1]
        prefix = "-" + str(j)
    else:
        pre_word = doc[i+j][0]
        pre_postag = doc[i+j][1]   
        prefix = "+" + str(j)     

    features.update({
        prefix + ':word.lower()': pre_word.lower(),
        prefix + ':word.istitle()': pre_word.istitle(),
        prefix + ':word.isupper()': pre_word.isupper(),
        prefix + ':postag': pre_postag,
        prefix + ':postag[:2]': pre_postag[:2],
    })
    return features

def token2features(doc, i):
    word = doc[i][0]
    postag = doc[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[:3]': word[:3],
        'word[:2]': word[:2],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        'lemma': lemmatizer.lemmatize(word)
    }
    # Take context into account: -2 tokens
    if i > 0 and i < CONTEXT:
        features = helper_word(word, doc, i, 1, features, True)
    elif i >= CONTEXT:
        for j in range(1, CONTEXT+1):
            features = helper_word(word, doc, i, j, features, True)

    # Take context into account: +2 tokens
    if i < len(doc)-CONTEXT:
        for j in range(1, CONTEXT+1):
            features = helper_word(word, doc, i, j, features, False)

    return features

def doc2features(doc):
    return [token2features(doc, i) for i in range(len(doc))]

def doc2labels(doc):
    return [label for token, postag, label in doc]

def doc2tokens(doc):
    return [token for token, postag, label in doc]
