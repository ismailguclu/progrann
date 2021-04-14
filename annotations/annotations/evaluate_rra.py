from seqeval.metrics import classification_report
import sequence_labeling
import helper
import json

faulty = []

def get_rumc_data(path):
    data = []
    with open(path, "r") as f:
        test_file = list(f)
    f.close()

    for doc in test_file:
        result = json.loads(doc)
        text, labels = result['text'], result['labels']
        new_labels = []
        for s, e, ent in labels:
            ent_text = text[s:e]
            new_labels.append((s, e, ent[1:-1], ent_text))
        data.append((text, new_labels))
    return data

def get_rra_data(path, true_data):
    data = []
    count = 0
    i = 0
    with open(path, "r") as f:
        test_file = list(f)
    f.close()

    for doc in test_file:
        try:
            result = json.loads(doc)
            labels = result['labels']
            text = true_data[i][0]
            new_labels = []
            for s, e, ent in labels:
                ent_text = text[s:e]
                new_labels.append((s, e, ent[1:-1], ent_text))
            data.append((text, new_labels))
            i += 1
        except:
            faulty.append(test_file.index(doc))
            count += 1
            i += 1
    print("Number of faulty reports: ", count)
    return data

def evaluate_partial(sequences, gold):
    print("F1", sequence_labeling.f1_score(gold, sequences))
    print("PARTIAL", sequence_labeling.partial_f1_score(gold, sequences))
    print(sequence_labeling.classification_report(gold, sequences))
    return

ground_truth = get_rumc_data("./data-rumc/500_extra_shuffled_clean.jsonl")
annotations = get_rra_data("./data-rumc/500_extra_shuffled_clean_preds.jsonl", ground_truth)

new_ground_truth = []
for i in range(len(ground_truth)):
    if i not in faulty:
        new_ground_truth.append(ground_truth[i])
print("Length new ground truth list: ", len(new_ground_truth))

# print(ground_truth[1])
# print(annotations[1])

# Ground truth bio/sequences
bio, sequences = helper.bio_tagging(new_ground_truth)

# Prediction bio/sequences
p_bio, p_sequences = helper.bio_tagging(annotations)

print(classification_report(sequences, p_sequences))