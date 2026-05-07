import torch
import numpy as np
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, multilabel_confusion_matrix

from data.preprocess import get_parents, get_all_shortest_paths, get_parents_list


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
            denominator = state_s[node][s].sum().item() + 2 * alpha
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
            P_S[node][s] = (match.sum().item() + alpha) / (predict.shape[0] + 4 * alpha)

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


def error_probability(target, predict, hierarchy, alpha=1e-5):
    assert predict.shape == target.shape
    states = [(0, 0), (0, 1), (1, 0), (1, 1)]
    N = predict.shape[0]

    error_a = target != predict
    P_E = {}
    for node in hierarchy.nodes:
        children = list(nx.descendants(hierarchy, node))

        if not children:
            P_E[node] = (error_a[:, node].sum().item() + 2 * alpha) / (N + 4 * alpha)
            continue

        sum_node = 0.0
        for child in children:
            prob_child = 0.0
            for s in states:
                state_s = (target[:, child] == s[0]) & (predict[:, child] == s[1])

                intersection = error_a[:, node] & state_s

                joint_prob = (intersection.sum().item() + alpha) / (N + 4 * alpha)
                prob_child += joint_prob

            sum_node += prob_child

        P_E[node] = sum_node / len(children)

    return torch.tensor([P_E[k] for k in sorted(P_E)])


def error_probability_optimized(target, predict, hierarchy, alpha=0):
    assert predict.shape == target.shape
    N = predict.shape[0]

    error_a = target != predict

    P_E = {}
    for node in hierarchy.nodes:
        children = list(nx.descendants(hierarchy, node))

        if not children:
            P_E[node] = (error_a[:, node].sum().item() + 2 * alpha) / (N + 4 * alpha)
        else:
            P_E[node] = (error_a[:, node].sum().item() + 4 * alpha) / (N + 4 * alpha)

    error = torch.tensor([P_E[k] for k in sorted(P_E)])

    return error


def serial_bitflip_error(target, predict):
    assert predict.shape == target.shape
    errors = np.sum(predict != target, axis=1).reshape(-1, predict.shape[1])
    return torch.tensor(np.mean(errors, axis=0))


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


def node_importance(labels, hierarchy):
    size = labels.shape[1]
    predict = []
    parents_map = get_parents_list(hierarchy)
    parents_array = np.zeros((size, size), dtype=int)
    for child, parents in parents_map.items():
        parents_array[child, list(parents)] = 1

    for target in labels.astype(int):
        altered = target ^ np.eye(size, dtype=int)

        indices = np.where((target == 0))[0]
        if indices.size > 0:
            altered[indices] |= parents_array[indices]
        # altered = target ^ np.random.binomial(1, 0.05, size=(size, size))

        # flipped_mask = (target == 0) & (altered == 1)

        # combined_parents = (flipped_mask @ parents_array) > 0

        # altered |= combined_parents

        predict.extend(altered)

    predict = np.stack(predict, axis=0)
    target = np.repeat(labels, repeats=size, axis=0)

    # cond_prob = conditional_probability(target, predict, hierarchy)
    # error = error_propagation(target, predict, hierarchy, cond_prob)

    # error = error_probability(target, predict, hierarchy)

    # error = error_probability_optimized(target, predict, hierarchy, 0)

    error = serial_bitflip_error(target, predict)

    return error


def plot_loss(losses, path, logger):
    plt.figure(figsize=(14, 8))

    for metric_name, loss_values in losses.items():
        epochs = range(1, len(loss_values) + 1)

        clean_label = metric_name.replace("_", " ").title()

        is_val = "val" in metric_name
        line_style = "--" if is_val else "-"
        marker_style = "s" if is_val else "o"

        plt.plot(
            epochs,
            loss_values,
            label=clean_label,
            linestyle=line_style,
            marker=marker_style,
            markersize=4,
        )

    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved successfully to {path}")


def plot_metrics(metrics, path, logger):
    means = [metric["mean"] for metric in metrics.values()]
    stds = [metric["std"] for metric in metrics.values()]

    plt.figure(figsize=(10, 6))
    x_positions = np.arange(len(metrics))

    plt.bar(
        x_positions,
        means,
        yerr=stds,
        capsize=8,
        color="steelblue",
        edgecolor="black",
        alpha=0.8,
    )

    plt.title("Cross-Validation Summary: Final Metrics", fontsize=14)
    plt.ylabel("Score")

    plt.xticks(x_positions, metrics.keys(), rotation=30, ha="right")
    plt.yticks(np.arange(0, 1.1, 0.1))

    plt.grid(axis="y", linestyle=":", alpha=0.7)
    plt.tight_layout()

    plt.savefig(path, dpi=300, bbox_inches="tight")
    logger.info(f"Combined scalar metrics plot saved successfully to {path}")
    plt.close()


def plot_importance(weight, hierarchy, classes, path, threshold=0.0056):
    filtered = {
        node: weight[node] for node in hierarchy.nodes() if weight[node] > threshold
    }
    filtered_nodes = filtered.keys()

    subgraph = hierarchy.subgraph(filtered_nodes)
    filtered_weights = [filtered[node] for node in subgraph.nodes()]

    pos = nx.drawing.layout.bfs_layout(subgraph, start=181, align="vertical")

    pos = {node: (coords[0] * 3, coords[1]) for node, coords in pos.items()}

    # pos = nx.drawing.layout.bfs_layout(hierarchy, start=181, align="vertical")
    # for node in pos:
    #     pos[node][0] *= 10

    # levels = nx.single_source_shortest_path_length(hierarchy, 181)

    # for node, level in levels.items():
    #     hierarchy.nodes[node]['layer'] = level

    # pos = nx.multipartite_layout(hierarchy, subset_key="layer", align="vertical")

    # pos = {node: (coords[0], coords[1]) for node, coords in pos.items()}

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_edges(subgraph, pos, alpha=0.2, edge_color="gray")
    nodes = nx.draw_networkx_nodes(
        subgraph,
        pos,
        node_color=filtered_weights,
        cmap=plt.cm.YlOrRd,
        node_size=300,
        edgecolors="black",
        linewidths=0.5,
    )
    nx.draw_networkx_labels(
        subgraph,
        pos,
        labels={node: classes[node] for node in filtered_nodes if node != 181},
        font_size=8,
    )
    plt.colorbar(nodes, label="Weight Intensity")
    plt.title("Hierarchical Graph Heatmap")
    plt.axis("off")
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close()
