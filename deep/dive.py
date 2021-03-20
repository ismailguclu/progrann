from random import random
import pandas as pd
import argparse
import json
import sources
import numpy as np

ENTITIES_DICT_2 = {"<DATUM>":"DATE", "<PERSOON>":"PERSON", "<TIJD>":"TIME", "<PLAATS>":"LOCATION", 
                   "<TELEFOONNUMMER>":"PHONE", "<LEEFTIJD>":"AGE", "<ZNUMMER>":"ZNUMMER"}

def args_parser():
    parser = argparse.ArgumentParser(description='Run Snorkel on unlabeled data.')
    parser.add_argument("--rumc", action="store_true", help="Load RUMC data.")
    parser.add_argument("--train", action="store_true", help="Train Snorkel token model.")
    parser.add_argument("--dev", action="store_true", help="Evaluate Snorkel token model on dev set.")
    parser.add_argument("--path", help="Absolute path to data file/folder.")
    parser.add_argument("--extra", action="store_true", help="Experiment 2: effect of additional data.")
    return parser.parse_args()

def get_rumc_data(fn):
    data = []
    labels = []

    with open("./data-rumc/"+fn, "r") as f:
        files = list(f)

    for doc in files:
        result = json.loads(doc)
        text = result['text']
        data.append(text)
        temp_labels = result['labels']
        if temp_labels:
            temp = []
            for tl in temp_labels:
                s,e = tl[0], tl[1]
                l,t = ENTITIES_DICT_2[tl[2]], text[s:e]
                temp.append([s,e,l,t])
            labels.append(temp)
        else:
            labels.append(temp_labels)
    return pd.DataFrame({"text" : data, "labels" : labels})

def main(args):
    if args.rumc:
        df = get_rumc_data("anon_ground_truth_v3_surrogates.jsonl")
        # 70/20/10 split
        train, validation, dev = np.split(
                                    df.sample(frac=1, random_state=42), 
                                    [int(.7*len(df)), int(.9*len(df))]
                                    )
        print("Load RUMC data: done.")

    if args.dev:
        label_model, df_snorkel_dev = sources.train(dev, gold_labels=True, mv=args.majority)
        print("Snorkel development model: done.")

    if args.train:
        label_model, df_snorkel_train = sources.train(train, gold_labels=False)
        _, _, _, df = sources.helper_snorkel_representation(df)
        label_model.save("./models/rumc-label-model-9-7-extra.pkl")
        df.to_pickle("./data-rumc/rumc-snorkel-9-7-extra.pkl")
        print("Snorkel train model: done.")

    if args.extra:
        df_extra = get_rumc_data("500_extra_shuffled.jsonl")
        fractions = [0, 25, 50, 75, 100]
        for f in fractions:
            extra = df_extra.sample(frac=f/100, random_state=42)
            add_train = pd.concat([train, extra], sort=False)
            label_model, df_snorkel_train = sources.train(add_train, gold_labels=False)

    return

if __name__ == "__main__":
    main(args_parser())