import pandas as pd
import json
import argparse
import numpy as np

LABELS = {}

def args_parser():
    parser = argparse.ArgumentParser(description='Print instance in dev set given index number.')
    parser.add_argument("--index", type=int, help="Index number.")
    parser.add_argument("--gold", action="store_true", help="Print label(s) for given index number.")
    parser.add_argument("--distr", action="store_true", help="Print distribution of labels in dev set.")
    return parser.parse_args()

def get_data():
    data = []
    labels = []

    with open("./data-rumc/anon_ground_truth_v3_surrogates.jsonl", "r") as f:
        files = list(f)

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

def main(args):
    df = get_data()
    train, val, dev = np.split(
                        df.sample(frac=1, random_state=42), 
                        [int(.7*len(df)), int(.9*len(df))]
                        )
    if args.distr:
        val['labels'].apply(get_distr_labels)
        print(LABELS)
    nr_items = len(val)

    if args.index:
        x = args.index
        if x < nr_items:
            print(dev.iloc[x]['records'])
            if args.gold:
                print(dev.iloc[x]['labels'])
    return

if __name__ == "__main__":
    main(args_parser())