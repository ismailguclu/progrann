import json
import pandas as pd
import random

LABELS = {}
to_file = True

def get_data(files):
    data = []
    labels = []

    for doc in files:
        result = json.loads(doc)
        data.append(result['text'])
        labels.append(result['labels'])
    return pd.DataFrame({"records" : data, "labels" : labels})

def get_distr_labels(doc):
    for d in doc:
        if d is not None:
            l = d[2]
            if l in LABELS:
                LABELS[l] += 1
            else:
                LABELS[l] = 1
    return

def stats(docs):
    df = get_data(docs)
    df['labels'].apply(get_distr_labels)
    total = sum(LABELS.values())
    print(total)
    for l, p in LABELS.items():
        print("label: {} N: {} percentage: {}".format(l, p, p/total))

    LABELS.clear()
    return

def write_to_file(fn, files):
    with open(fn, mode="w") as output:
        for s in files:
            output.write(s)
    output.close()
    return

if __name__ == "__main__":
    with open("./data-rumc/500_extra_shuffled_clean.jsonl", "r") as f:
        files = list(f)

    random.seed(12)
    random.shuffle(files)
    extra, test = files[0:150], files[150:500]
    print("Stats extra...")
    stats(extra)
    if to_file:
        write_to_file("./data-rumc/extra_clean_150.txt", extra)
    print()
    print("Stats test...")
    stats(test)
    if to_file:
        write_to_file("./data-rumc/test_clean_350.txt", test)
