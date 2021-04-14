"""Metrics to assess performance on sequence labeling task given prediction
Functions named as ``*_score`` return a scalar value to maximize: the higher
the better
"""

from __future__ import absolute_import, division, print_function

from collections import defaultdict

import numpy as np


def overlaps(span_cand, span_gold):
    cand_start, cand_stop = span_cand
    gold_start, gold_stop = span_gold
    return bool(
        set(range(cand_start, cand_stop + 1)) & set(range(gold_start, gold_stop + 1))
    )


def get_entities(seq, suffix=False):
    """Gets entities from sequence.
    Args:
        seq (list): sequence of labels.
        suffix (bool): BIO tag at end instead of start (e.g. LOC-B)
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        >>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ["O"]]

    prev_tag = "O"
    prev_type = ""
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ["O"]):
        if suffix:
            tag = chunk[-1]
            type_ = chunk.split("-")[0]
        else:
            tag = chunk[0]
            type_ = chunk.split("-")[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == "E":
        chunk_end = True
    if prev_tag == "L":
        chunk_end = True
    if prev_tag == "S":
        chunk_end = True
    if prev_tag == "U":
        chunk_end = True

    if prev_tag == "B" and tag == "B":
        chunk_end = True
    if prev_tag == "B" and tag == "S":
        chunk_end = True
    if prev_tag == "B" and tag == "U":
        chunk_end = True
    if prev_tag == "B" and tag == "O":
        chunk_end = True
    if prev_tag == "I" and tag == "B":
        chunk_end = True
    if prev_tag == "I" and tag == "S":
        chunk_end = True
    if prev_tag == "I" and tag == "U":
        chunk_end = True
    if prev_tag == "I" and tag == "O":
        chunk_end = True

    if prev_tag != "O" and prev_tag != "." and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == "B":
        chunk_start = True
    if tag == "S":
        chunk_start = True
    if tag == "U":
        chunk_start = True

    if prev_tag == "E" and tag == "E":
        chunk_start = True
    if prev_tag == "L" and tag == "L":
        chunk_start = True
    if prev_tag == "E" and tag == "I":
        chunk_start = True
    if prev_tag == "L" and tag == "I":
        chunk_start = True
    if prev_tag == "S" and tag == "E":
        chunk_start = True
    if prev_tag == "U" and tag == "L":
        chunk_start = True
    if prev_tag == "S" and tag == "I":
        chunk_start = True
    if prev_tag == "U" and tag == "I":
        chunk_start = True
    if prev_tag == "O" and tag == "E":
        chunk_start = True
    if prev_tag == "O" and tag == "L":
        chunk_start = True
    if prev_tag == "O" and tag == "I":
        chunk_start = True

    if tag != "O" and tag != "." and prev_type != type_:
        chunk_start = True

    return chunk_start


def f1_score(y_true, y_pred, average="micro", suffix=False, call_get_entities=True):
    """Compute the F1 score.
    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::
        F1 = 2 * (precision * recall) / (precision + recall)
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    Example:
        >>> from seqeval.metrics import f1_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> f1_score(y_true, y_pred)
        0.50
    """
    if call_get_entities:
        true_entities = set(get_entities(y_true, suffix))
        pred_entities = set(get_entities(y_pred, suffix))
    else:
        true_entities = y_true
        pred_entities = y_pred

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return score


def partial_f1_score(
    y_true, y_pred, average="micro", suffix=False, call_get_entities=True
):
    """Compute the partial F1 score.
    The partial F1 score is similar to the F1 score. The difference is that entities are
    also counted as correct when there is partial overlap. I.e. some boundary
    mismatching is allowed.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    """
    if call_get_entities:
        true_entities = set(get_entities(y_true, suffix))
        pred_entities = set(get_entities(y_pred, suffix))
    else:
        true_entities = y_true
        pred_entities = y_pred

    nb_pred = len(pred_entities)
    nb_true = len(true_entities)
    nb_correct = 0

    for ent_type, start, end in pred_entities:
        # TODO: and gold_ent[0] == ent_type
        if any(
            overlaps((start, end), gold_ent[1:]) and ent_type == gold_ent[0]
            for gold_ent in true_entities
        ):
            nb_correct += 1

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return p, r, score


def type_invariant_f1_score(
    y_true, y_pred, average="micro", suffix=False, call_get_entities=True
):
    """Compute the type invariant F1 score.
    This can be used when you want to evaluate binary mention detection; e.g. does the
    tagger find an entity but assigns the type incorrectly.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    """
    # just use the start/end indices, by means of x[1:]
    if call_get_entities:
        true_entities = set(x[1:] for x in get_entities(y_true, suffix))
        pred_entities = set(x[1:] for x in get_entities(y_pred, suffix))
    else:
        true_entities = set(x[1:] for x in y_true)
        pred_entities = set(x[1:] for x in y_pred)

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return score


def accuracy_score(y_true, y_pred):
    """Accuracy classification score.
    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    Example:
        >>> from seqeval.metrics import accuracy_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> accuracy_score(y_true, y_pred)
        0.80
    """
    if any(isinstance(s, list) for s in y_true):
        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]

    nb_correct = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    nb_true = len(y_true)

    score = nb_correct / nb_true

    return score


def precision_score(
    y_true, y_pred, average="micro", suffix=False, call_get_entities=True
):
    """Compute the precision.
    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample.
    The best value is 1 and the worst value is 0.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    Example:
        >>> from seqeval.metrics import precision_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> precision_score(y_true, y_pred)
        0.50
    """
    if call_get_entities:
        true_entities = set(get_entities(y_true, suffix))
        pred_entities = set(get_entities(y_pred, suffix))
    else:
        true_entities = y_true
        pred_entities = y_pred

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)

    score = nb_correct / nb_pred if nb_pred > 0 else 0

    return score


def recall_score(y_true, y_pred, average="micro", suffix=False, call_get_entities=True):
    """Compute the recall.
    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.
    The best value is 1 and the worst value is 0.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    Example:
        >>> from seqeval.metrics import recall_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> recall_score(y_true, y_pred)
        0.50
    """
    if call_get_entities:
        true_entities = set(get_entities(y_true, suffix))
        pred_entities = set(get_entities(y_pred, suffix))
    else:
        true_entities = y_true
        pred_entities = y_pred

    nb_correct = len(true_entities & pred_entities)
    nb_true = len(true_entities)

    score = nb_correct / nb_true if nb_true > 0 else 0

    return score


def performance_measure(y_true, y_pred):
    """
    Compute the performance metrics: TP, FP, FN, TN
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        performance_dict : dict
    Example:
        >>> from seqeval.metrics import performance_measure
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'O', 'B-ORG'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> performance_measure(y_true, y_pred)
        (3, 3, 1, 4)
    """
    performance_dict = dict()
    if any(isinstance(s, list) for s in y_true):
        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]
    performance_dict["TP"] = sum(
        y_t == y_p for y_t, y_p in zip(y_true, y_pred) if ((y_t != "O") or (y_p != "O"))
    )
    performance_dict["FP"] = sum(y_t != y_p for y_t, y_p in zip(y_true, y_pred))
    performance_dict["FN"] = sum(
        ((y_t != "O") and (y_p == "O")) for y_t, y_p in zip(y_true, y_pred)
    )
    performance_dict["TN"] = sum(
        (y_t == y_p == "O") for y_t, y_p in zip(y_true, y_pred)
    )

    return performance_dict


def classification_report(
    y_true, y_pred, digits=2, suffix=False, output_json=False, call_get_entities=True
):
    """Build a text report showing the main classification metrics.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a classifier.
        digits : int. Number of digits for formatting output floating point values.
    Returns:
        report : string. Text summary of the precision, recall, F1 score for each class.
    Examples:
        >>> from seqeval.metrics import classification_report
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> print(classification_report(y_true, y_pred))
                     precision    recall  f1-score   support
        <BLANKLINE>
               MISC       0.00      0.00      0.00         1
                PER       1.00      1.00      1.00         1
        <BLANKLINE>
          micro avg       0.50      0.50      0.50         2
          macro avg       0.50      0.50      0.50         2
        <BLANKLINE>
    """
    if call_get_entities:
        true_entities = set(get_entities(y_true, suffix))
        pred_entities = set(get_entities(y_pred, suffix))
    else:
        true_entities = y_true
        pred_entities = y_pred

    name_width = 0
    d1 = defaultdict(set)
    d2 = defaultdict(set)
    for e in true_entities:
        d1[e[0]].add(e)
        name_width = max(name_width, len(e[0]))
    for e in pred_entities:
        d2[e[0]].add(e)

    last_line_heading = "macro avg"
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1", "partial-f1", "t-inv-f1", "support"]
    head_fmt = u"{:>{width}s} " + u" {:>11}" * len(headers)
    report = head_fmt.format(u"", *headers, width=width)
    report += u"\n\n"

    row_fmt = u"{:>{width}s} " + u" {:>11.{digits}f}" * 5 + u" {:>11}\n"

    # build dict with scores as well
    result_dict = {"ents_per_type": {}}  # type: ignore

    ps, rs, f1s, partial_f1s, t_inv_f1s, s = [], [], [], [], [], []
    for type_name, true_entities in d1.items():
        result_dict["ents_per_type"][type_name] = {}
        pred_entities = d2[type_name]
        p = precision_score(true_entities, pred_entities, call_get_entities=False)
        r = recall_score(true_entities, pred_entities, call_get_entities=False)
        f1 = f1_score(true_entities, pred_entities, call_get_entities=False)
        pp, pr, partial_f1 = partial_f1_score(
            true_entities, pred_entities, call_get_entities=False
        )
        if type_name == "PERSON":
            print("PRECISION/RECALL FOR PERSON:")
            print(pp, pr)
        t_inv_f1 = type_invariant_f1_score(
            true_entities, pred_entities, call_get_entities=False
        )
        nb_true = len(true_entities)

        report += row_fmt.format(
            *[type_name, p, r, f1, partial_f1, t_inv_f1, nb_true],
            width=width,
            digits=digits
        )

        for key, score in zip(
            ["p", "r", "f1", "partial-f1", "t-inv-f1", "support"],
            [p, r, f1, partial_f1, t_inv_f1, nb_true],
        ):
            result_dict["ents_per_type"][type_name][key] = score

        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        partial_f1s.append(partial_f1)
        t_inv_f1s.append(t_inv_f1)
        s.append(nb_true)

    report += u"\n"

    # compute averages
    result_dict["micro_avg"] = {
        "p": precision_score(
            y_true, y_pred, suffix=suffix, call_get_entities=call_get_entities
        ),
        "r": recall_score(
            y_true, y_pred, suffix=suffix, call_get_entities=call_get_entities
        ),
        "f1": f1_score(
            y_true, y_pred, suffix=suffix, call_get_entities=call_get_entities
        ),
        "partial-f1": partial_f1_score(
            y_true, y_pred, call_get_entities=call_get_entities
        )[2],
        "t-inv-f1": type_invariant_f1_score(
            y_true, y_pred, call_get_entities=call_get_entities
        ),
        "support": int(np.sum(s)),
    }
    report += row_fmt.format(
        "micro avg",
        result_dict["micro_avg"]["p"],
        result_dict["micro_avg"]["r"],
        result_dict["micro_avg"]["f1"],
        result_dict["micro_avg"]["partial-f1"],
        result_dict["micro_avg"]["t-inv-f1"],
        result_dict["micro_avg"]["support"],
        width=width,
        digits=digits,
    )

    result_dict["macro_avg"] = {
        "p": float(np.average(ps, weights=s)),
        "r": float(np.average(rs, weights=s)),
        "f1": float(np.average(f1s, weights=s)),
        "partial-f1": float(np.average(partial_f1s, weights=s)),
        "t-inv-f1": float(np.average(t_inv_f1s, weights=s)),
        "support": int(np.sum(s)),
    }
    report += row_fmt.format(
        last_line_heading,
        result_dict["macro_avg"]["p"],
        result_dict["macro_avg"]["r"],
        result_dict["macro_avg"]["f1"],
        result_dict["macro_avg"]["partial-f1"],
        result_dict["macro_avg"]["t-inv-f1"],
        result_dict["macro_avg"]["support"],
        width=width,
        digits=digits,
    )

    return result_dict if output_json else report