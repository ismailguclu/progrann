import pandas as pd
import argparse
import models
import sequence_labeling
import helper
import preprocessor
import sources
import numpy as np

def args_parser():
    parser = argparse.ArgumentParser(description='Run Snorkel on unlabeled data.')
    parser.add_argument("--rumc", action="store_true", help="Load RUMC data.")
    parser.add_argument("--i2b2", action="store_true", help="Load i2b2 data.")
    parser.add_argument("--train", action="store_true", help="Train Snorkel token model.")
    parser.add_argument("--dev", action="store_true", help="Evaluate Snorkel token model on dev set.")
    parser.add_argument("--path", help="Absolute path to data file/folder.")
    parser.add_argument("--extra", action="store_true", help="Experiment 2: effect of additional data.")
    return parser.parse_args()

def main(args):
    if args.i2b2:
        CARDINALITY = 25
        VERSION = "18-2"
        DATA= "i2b2"
        all_text, all_labels, all_bio = preprocessor.run()
        df = pd.DataFrame({"text":all_text, "labels":all_labels, "bio":all_bio})
        train, val, dev = np.split(
                            df.sample(frac=1, random_state=42), 
                            [int(.7*len(df)), int(.9*len(df))]
                            )
        print("Load i2b2 data: done.")

    if args.rumc:
        CARDINALITY = 8 # number of entities + 1
        VERSION = "9-7"
        DATA = "rumc"
        df = helper.get_rumc_data("anon_ground_truth_v3_surrogates.jsonl")
        # 70/20/10 split
        train, val, dev = np.split(
                                    df.sample(frac=1, random_state=42), 
                                    [int(.7*len(df)), int(.9*len(df))]
                                    )
        print("Load RUMC data: done.")

    if args.dev:
        label_model, df_snorkel_dev = sources.train(dev, type_data=DATA, cardinality=CARDINALITY, gold_labels=True)
        print("Snorkel development model: done.")

    if args.train:
        label_model, df_snorkel_train = sources.train(train, type_data=DATA, cardinality=CARDINALITY, gold_labels=False)
        if DATA == "rumc":
            _, _, _, df = sources.helper_snorkel_representation(df)
        else:
            _, _, _, df = sources.helper_snorkel_i2b2_representation(df)

        label_model.save("./models/"+DATA+"-label-model-"+VERSION+".pkl")
        df.to_pickle("./snorkel-data/"+DATA+"-snorkel-"+VERSION+".pkl")
        print("Snorkel train model: done.")

    if args.extra:
        df_test = helper.get_rumc_data("500_extra_shuffled.jsonl")
        fractions = [25, 50, 75, 100]
        f1_scores = []
        print("Starting training splits...")
        for f in fractions:
            subtrain = train.sample(frac=f/100, random_state=42)
            #extra_train = pd.concat([train, extra], sort=False)
            label_model, df_snorkel_subtrain = sources.train(subtrain, gold_labels=False)
            L_train, _, _, _ = sources.helper_snorkel_representation(df_snorkel_subtrain)
            annotations_pred = label_model.predict(L=L_train)
            b_train, s_train = helper.replace_tags_data(subtrain, annotations_pred)

            val_rumc = list(zip(val.text, val.labels))
            b_val, s_val = helper.bio_tagging(val_rumc)

            crf = models.CRF("crf")
            crf.validate(b_train)             
            crf_y_pred_val = crf.label(b_val)
            f1_score = sequence_labeling.f1_score(s_val, crf_y_pred_val)
            f1_scores.append(f1_score)
            print("Current split: {} and corresponding F1 score (val): {}".format(f, f1_score))
        print("F1 SCORES: ", f1_scores)
    return

if __name__ == "__main__":
    main(args_parser())