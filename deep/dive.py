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
    parser.add_argument("--path", help="Absolute path to data file/folder.")
    return parser.parse_args()

def get_rumc_data():
    data = []
    labels = []

    with open("./data-rumc/anon_ground_truth_v3_surrogates.jsonl", "r") as f:
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
        df = get_rumc_data()
        train, validation, development = np.split(
                                            df.sample(frac=1, random_state=42), 
                                            [int(.7*len(df)), int(.9*len(df))]
                                        )
        sources.run(development)
    return

if __name__ == "__main__":
    main(args_parser())