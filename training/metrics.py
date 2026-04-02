import torch
import numpy as np
import pandas as pd
import os
import networkx as nx
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, multilabel_confusion_matrix

from data.preprocess import get_parents, get_all_shortest_paths


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


def standard_evaluation(target, predict, averages=None):
    assert len(target) == len(predict), "Different label sizes in true vs pred"
    # predict = (predict >= 0.5).astype(int)
    metrics = {}
    for average in averages:
        prec, rec, f1, _ = precision_recall_fscore_support(
            target, predict, average=average, zero_division=0
        )
        metrics[f"{average}_prec"] = prec
        metrics[f"{average}_rec"] = rec
        metrics[f"{average}_f1"] = f1
    return metrics


def hierarchical_evaluation(target, predict, hierarchy, averages=None):  # TODO: FIX
    assert len(target) == len(predict), "Different label sizes in true vs pred"

    extended_predict = []
    for prediction in predict:
        prediction_copy = list(prediction)
        for idx in range(len(prediction)):
            if prediction[idx]:
                parents = get_parents(hierarchy, idx)
                prediction_copy[parents] = 1
                prediction_copy.extend([p for p in parents if p not in prediction_copy])
        extended_predict.append(prediction_copy)
    extended_predict = np.vstack(extended_predict)
    metrics = {}
    for average in averages:
        prec, rec, f1, _ = precision_recall_fscore_support(
            target, extended_predict, average=average, zero_division=0
        )
        metrics[f"{average}_prec"] = prec
        metrics[f"{average}_rec"] = rec
        metrics[f"{average}_f1"] = f1
    return metrics


def ahc_evaluation(target, predict, hierarchy):
    ahc = []
    root_id = len(hierarchy)
    roots = [n for n, d in hierarchy.in_degree() if d == 0]
    hierarchy.add_node(root_id)

    for r in roots:
        hierarchy.add_edge(root_id, r)
    graph = hierarchy.to_undirected()
    paths = get_all_shortest_paths(graph)

    for prediction, true in zip(predict, target):
        distances = []
        pred_indices = np.where(prediction == 1)[0]
        true_indices = np.where(true == 1)[0]
        for pred in pred_indices:
            pred_distances = paths.get(pred, {})
            min_d = min(
                (pred_distances.get(label, np.inf) for label in true_indices),
                default=np.inf,
            )
            # min_d = np.inf
            # for label in true_indices:
            #     try:
            #         d = nx.shortest_path_length(graph, pred, label)
            #         min_d = min(min_d, d)
            #     except nx.NetworkXNoPath:
            #         continue
            if min_d < np.inf:
                distances.append(min_d)
        if distances:
            ahc.append(np.mean(distances))
        else:
            ahc.append(np.nan)
    hierarchy.remove_node(root_id)
    mean_ahc = np.nanmean(ahc)
    return {"ahc": mean_ahc.item() if not np.isnan(mean_ahc) else 0.0}


def analysis(target, predict, classes, log_analysis, levels=None):

    assert predict.shape == target.shape
    assert len(classes) == predict.shape[1]

    results = []

    label_frequency = target.mean(axis=0)

    cms = multilabel_confusion_matrix(target, predict)
    prec, rec, f1, _ = precision_recall_fscore_support(
        target, predict, average=None, zero_division=0
    )

    for i in range(len(classes)):
        # TN, FP, FN, TP = cms[i].ravel()
        results.append(
            {
                "label": classes[i],
                "frequency": label_frequency[i],
                "cm": cms[i],
                "prec": prec[i],
                "rec": rec[i],
                "f1": f1[i],
            }
        )
        # results[f"freq_{classes[i]}"] = label_frequency[i]
    res = pd.DataFrame(results)

    aggregates = []

    for id, level in levels.items():
        prec, rec, f1, _ = precision_recall_fscore_support(
            target[:, level], predict[:, level], average="micro", zero_division=0
        )

        aggregates.append(
            {
                "level": id,
                "num_labels": len(level),
                "avg_label_frequency": label_frequency[level].mean(),
                "cm": cms[level].sum(axis=0),
                "prec": prec,
                "rec": rec,
                "f1": f1,
            }
        )

    prec, rec, f1, _ = precision_recall_fscore_support(
        target, predict, average="micro", zero_division=0
    )
    aggregates.append(
        {
            "level": "total",
            "num_labels": len(classes),
            "avg_label_frequency": label_frequency.mean(),
            "cm": cms.sum(axis=0),
            "prec": prec,
            "rec": rec,
            "f1": f1,
        }
    )

    lvl = pd.DataFrame(aggregates)

    mode = "a" if os.path.exists(log_analysis) else "w"

    with pd.ExcelWriter(log_analysis, engine="openpyxl", mode=mode) as writer:
        res.to_excel(
            writer, sheet_name=f"Label_{datetime.now().timestamp()}", index=False
        )
        lvl.to_excel(
            writer, sheet_name=f"Level_{datetime.now().timestamp()}", index=False
        )


def conditional_probability(target, predict, hierarchy, alpha=1e-5):
    assert predict.shape == target.shape
    states = [(0, 0), (0, 1), (1, 0), (1, 1)]
    # sorted_levels = sorted(levels.keys(), reverse=True)

    state_s = {}
    error_a = target != predict
    for node in hierarchy.nodes:
        state_s[node] = {}
        for s in states:
            state_s[node][s] = (target[:, node] == s[0]) & (predict[:, node] == s[1])
    cond_prob = {}
    for node in hierarchy.nodes:
        ancestors = nx.ancestors(hierarchy, node)
        if not ancestors:
            continue
        for s in states:
            denominator = state_s[node][s].sum().item() + 2*alpha
            # if denominator == 0:
            #     for a in ancestors:
            #         cond_prob[(a, node, s)] = 0.0
            #     continue
            for a in ancestors:
                intersection = error_a[:, a] & state_s[node][s]
                numerator = intersection.sum().item() + alpha

                cond_prob[(a, node, s)] = numerator / denominator
    return cond_prob


def error_propagation(target, predict, hierarchy, cond_prob, alpha=1e-5):
    assert predict.shape == target.shape

    states = [(0, 0), (0, 1), (1, 0), (1, 1)]
    P_S = {}
    for node in hierarchy.nodes:
        P_S[node] = {}
        for s in states:
            match = (target[:, node] == s[0]) & (predict[:, node] == s[1])
            P_S[node][s] = (match.sum().item() + alpha) / (predict.shape[0] + 4*alpha)

    P_E = {}
    for node in hierarchy.nodes:
        children = list(nx.descendants(hierarchy, node))

        if not children:
            P_E[node] = P_S[node][(1, 0)] + P_S[node][(0, 1)]
            continue

        sum_node = 0.0

        for child in children:

            sum_node += sum(
                cond_prob.get((node, child, s), 0.0) * P_S[child][s] for s in states
            )

        P_E[node] = sum_node / len(children)
    return torch.tensor([P_E[k] for k in sorted(P_E)])


def multilabel_cms(target, predict, classes):
    metrics = {}

    cms_array = multilabel_confusion_matrix(target, predict)

    for i, cls in enumerate(classes):
        metrics[cls] = cms_array[i]

    return metrics


def total_average_cm(cms):
    matrices = np.stack(list(cms.values()), axis=0)
    return {"cm": np.mean(matrices, axis=0)}


def level_average_cm(cms, levels):
    averages = {}
    for key, label_matrices in cms.items():
        for i, labels in enumerate(levels):
            matrices = [label_matrices[label] for label in labels]
            averages[key][f"level{i}"] = np.mean(matrices, axis=0)
    return averages
