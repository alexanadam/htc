import torch
import numpy as np
import networkx as nx
from sklearn.metrics import precision_recall_fscore_support
from data.preprocess import get_parents


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
    graph = hierarchy.to_undirected()

    for prediction, true in zip(predict, target):
        distances = []
        pred_indices = np.where(prediction == 1)[0]
        true_indices = np.where(true == 1)[0]
        for pred in pred_indices:
            min_d = np.inf
            for label in true_indices:
                try:
                    d = nx.shortest_path_length(graph, pred, label)
                    min_d = min(min_d, d)
                except nx.NetworkXNoPath:
                    continue
            if min_d < np.inf:
                distances.append(min_d)
        if distances:
            ahc.append(np.mean(distances))
        else:
            ahc.append(np.nan)
    return {"ahc": np.nanmean(ahc)}
