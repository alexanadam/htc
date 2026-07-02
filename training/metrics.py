import torch
import numpy as np
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, multilabel_confusion_matrix

from data.preprocess import (
    get_parents,
    get_all_shortest_paths,
    get_parents_list,
    get_predecessor_map,
)


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


def violations(target, predict, hierarchy):
    assert predict.shape == target.shape
    # count = 0
    # ancestor_map = get_predecessor_map(hierarchy)
    # for i, sample in enumerate(target):
    #     for j, label in enumerate(sample):
    #         if label == 1 and predict[i][j] == 1:
    #             if ancestor_map[j]:
    #                 if predict[i][ancestor_map[j][0]] == 0:
    #                     count += 1
    ancestor_map = get_predecessor_map(hierarchy)
    ancestor_list = [
        ancestor_map[key][0] if ancestor_map[key] else False
        for key in sorted(ancestor_map)
    ]
    true_positives = (target == 1) & (predict == 1)
    parent_predicted_zero = predict[:, ancestor_list] == 0
    has_ancestor = np.array(
        [False if (x is False) else True for x in ancestor_list], dtype=bool
    )
    violations = true_positives & parent_predicted_zero & has_ancestor
    count = np.sum(violations)

    parent_predicted_one = predict[:, ancestor_list] == 1
    no_violation = true_positives & parent_predicted_one & has_ancestor
    no_count = np.sum(no_violation)

    # indices = np.where(np.any(violations, axis=1))[0]
    return {
        "violations": count / predict.size,
        "rate": count / len(predict),
        "count": count,
        "size": predict.size,
        "no_violation": no_count / predict.size,
        "no_rate": no_count / len(predict),
        "no_count": no_count,
        # "indices": indices,
    }


def hierarchical_error_analysis(target, predict, hierarchy, levels):
    node_level = {}

    for lvl, node_ids in levels.items():
        for node_id in node_ids:
            node_level[node_id] = lvl

    children = {node: set(hierarchy.successors(node)) for node in hierarchy.nodes}

    parent = {}

    for child in hierarchy.nodes:
        preds = list(hierarchy.predecessors(child))
        parent[child] = preds[0] if preds else None

    ancestors = {node: nx.ancestors(hierarchy, node) for node in hierarchy.nodes}

    descendants = {node: nx.descendants(hierarchy, node) for node in hierarchy.nodes}

    per_level = {
        lvl: {
            "predicted_labels": 0,
            "target_labels": 0,
            "wrong_predictions": 0,
            "sibling_errors": 0,
            "correct_child": 0,
            "parent_only_errors": 0,
            "parent_only_true": 0,
            "missing_intermediate_errors": 0,
        }
        for lvl in sorted(levels.keys())
    }

    totals = {
        "predicted_labels": 0,
        "target_labels": 0,
        "wrong_predictions": 0,
        "sibling_errors": 0,
        "correct_child": 0,
        "parent_only_errors": 0,
        "parent_only_true": 0,
        "missing_intermediate_errors": 0,
    }
    for tgt_row, pred_row in zip(target, predict):

        pred_nodes = set(np.flatnonzero(pred_row))
        tgt_nodes = set(np.flatnonzero(tgt_row))

        totals["predicted_labels"] += len(pred_nodes)
        totals["target_labels"] += len(tgt_nodes)

        for node in pred_nodes:
            per_level[node_level[node]]["predicted_labels"] += 1

        for node in tgt_nodes:
            per_level[node_level[node]]["target_labels"] += 1

        wrong_nodes = pred_nodes ^ tgt_nodes

        totals["wrong_predictions"] += len(wrong_nodes)

        for node in wrong_nodes:
            per_level[node_level[node]]["wrong_predictions"] += 1

        sibling_errors = 0
        correct_child = 0

        for p, child_set in children.items():

            pred_children = child_set & pred_nodes
            tgt_children = child_set & tgt_nodes

            correct_overlap = pred_children & tgt_children

            if len(correct_overlap) > 0:
                correct_child += len(correct_overlap)
                per_level[node_level[p]]["correct_child"] += len(correct_overlap)

            if len(child_set) < 2:
                continue

            wrong_pred = pred_children - tgt_children
            missed_true = tgt_children - pred_children

            count = min(len(wrong_pred), len(missed_true))

            sibling_errors += count

            per_level[node_level[p]]["sibling_errors"] += count

        totals["sibling_errors"] += sibling_errors
        totals["correct_child"] += correct_child

        parent_only = 0
        parent_only_true = 0

        for p, child_set in children.items():

            if not child_set:
                parent_only_true += 1
                per_level[node_level[p]]["parent_only_true"] += 1
                continue

            if p not in pred_nodes:
                continue

            if p not in tgt_nodes:
                continue

            tgt_children = child_set & tgt_nodes
            pred_children = child_set & pred_nodes

            if len(pred_children) == 0:
                if not tgt_children:
                    parent_only_true += 1
                    per_level[node_level[p]]["parent_only_true"] += 1
                    continue

                else:
                    # ORIGINAL: Target had children, but prediction stopped at parent (error)
                    parent_only += 1
                    per_level[node_level[p]]["parent_only_true"] += 1

        totals["parent_only_errors"] += parent_only
        totals["parent_only_true"] += parent_only_true

        missing_intermediate = 0

        for node in tgt_nodes:

            if node in pred_nodes:
                continue

            has_target_ancestor_predicted = bool(
                ancestors[node] & pred_nodes & tgt_nodes
            )

            has_target_descendant_predicted = bool(
                descendants[node] & pred_nodes & tgt_nodes
            )

            if has_target_ancestor_predicted and has_target_descendant_predicted:

                missing_intermediate += 1

                per_level[node_level[node]]["missing_intermediate_errors"] += 1

        totals["missing_intermediate_errors"] += missing_intermediate

    rates = {
        "wrong_prediction_rate": totals["wrong_predictions"]
        / max(1, totals["target_labels"]),
        "sibling_error_rate": totals["sibling_errors"]
        / max(1, totals["wrong_predictions"]),
        "parent_only_rate": totals["parent_only_errors"] / len(predict),
        "parent_only_no_target_rate": totals["parent_only_errors"] / len(predict),
        "missing_intermediate_rate": totals["missing_intermediate_errors"]
        / len(predict),
    }
    means = {k: v / len(predict) for k, v in totals.items()}

    return {"totals": totals, "per_level": per_level, "rates": rates, "means": means}


def calculate_hierarchical_errors(target, predict, hierarchy, levels):
    fp_matrix = (predict == 1) & (target == 0)
    fn_matrix = (predict == 0) & (target == 1)

    tp_matrix = (predict == 1) & (target == 1)

    ancestors = {n: list(nx.ancestors(hierarchy, n)) for n in hierarchy.nodes()}
    descendants = {n: list(nx.descendants(hierarchy, n)) for n in hierarchy.nodes()}

    siblings = {n: set() for n in hierarchy.nodes()}
    for n in hierarchy.nodes:
        for p in hierarchy.predecessors(n):
            for c in hierarchy.successors(p):
                if c != n:
                    siblings[n].add(c)
    siblings = {n: list(siblings[n]) for n in hierarchy.nodes}

    node_level = {}
    for lvl, node_ids in levels.items():
        for node_id in node_ids:
            node_level[node_id] = lvl

    errors = {
        "totals": 0,
        "branch_omitted": 0,
        "branch_added": 0,
        "additional_descendants": 0,
        "descendant_missing": 0,
        "sibling_error": 0,
        "ancestor_missing": 0,
    }

    per_level = {
        lvl: {
            "branch_omitted": 0,
            "branch_added": 0,
            "additional_descendants": 0,
            "descendant_missing": 0,
            "sibling_error": 0,
            "ancestor_missing": 0,
        }
        for lvl in sorted(levels.keys())
    }

    success = {
        "ancestor_present": 0,
        "descendants_present": 0,
        "descendants_correctly_omitted": 0,
    }
    success_per_level = {
        lvl: {
            "ancestor_present": 0,
            "descendants_present": 0,
            "descendants_correctly_omitted": 0,
        }
        for lvl in sorted(levels.keys())
    }

    # error_nodes = {k: [] for k in errors}

    for n in hierarchy.nodes:
        fps = fp_matrix[:, n]
        if np.any(fps):
            has_fn_sibling = (
                np.any(fn_matrix[:, siblings[n]], axis=1)
                if siblings[n]
                else np.zeros(len(target), dtype=bool)
            )
            has_tgt_ancestor = (
                np.any(target[:, ancestors[n]] == 1, axis=1)
                if ancestors[n]
                else np.zeros(len(target), dtype=bool)
            )

            mask_sibling = fps & has_fn_sibling
            mask_add_desc = fps & ~mask_sibling & has_tgt_ancestor
            mask_branch_add = fps & ~mask_sibling & ~has_tgt_ancestor

            errors["sibling_error"] += np.sum(mask_sibling)
            errors["additional_descendants"] += np.sum(mask_add_desc)
            errors["branch_added"] += np.sum(mask_branch_add)

            per_level[node_level[n]]["sibling_error"] += np.sum(mask_sibling)
            per_level[node_level[n]]["additional_descendants"] += np.sum(mask_add_desc)
            per_level[node_level[n]]["branch_added"] += np.sum(mask_branch_add)

        fns = fn_matrix[:, n]
        if np.any(fns):
            has_prd_descendant = (
                np.any(predict[:, descendants[n]] == 1, axis=1)
                if descendants[n]
                else np.zeros(len(target), dtype=bool)
            )
            has_prd_ancestor = (
                np.any(predict[:, ancestors[n]] == 1, axis=1)
                if ancestors[n]
                else np.zeros(len(target), dtype=bool)
            )
            has_fp_sibling = (
                np.any(fp_matrix[:, siblings[n]], axis=1)
                if siblings[n]
                else np.zeros(len(target), dtype=bool)
            )

            mask_anc_missing = fns & has_prd_descendant
            mask_desc_missing = (
                fns & ~mask_anc_missing & (has_prd_ancestor | has_fp_sibling)
            )
            mask_branch_omit = (
                fns
                & ~mask_anc_missing
                & ~mask_desc_missing
                & (~has_prd_ancestor & ~has_prd_descendant)
            )

            errors["ancestor_missing"] += np.sum(mask_anc_missing)
            errors["descendant_missing"] += np.sum(mask_desc_missing)
            errors["branch_omitted"] += np.sum(mask_branch_omit)

            per_level[node_level[n]]["ancestor_missing"] += np.sum(mask_anc_missing)
            per_level[node_level[n]]["descendant_missing"] += np.sum(mask_desc_missing)
            per_level[node_level[n]]["branch_omitted"] += np.sum(mask_branch_omit)

            #     if any(s in fn_indices for s in siblings[v]):
            #         errors['sibling_error'] += 1
            #         error_nodes['sibling_error'].append(v)
            #     elif any(target[a] == 1 for a in ancestors[v]):
            #         errors['additional_descendants'] += 1
            #         error_nodes['additional_descendants'].append(v)
            #     elif not any(target[a] == 1 for a in ancestors[v]):
            #         errors['branch_added'] += 1
            #         error_nodes['branch_added'].append(v)

            # for v in fn_indices:
            #     # if any(s in fp_indices for s in siblings[v]):
            #     #     errors['sibling_error'] += 1
            #     #     error_nodes['sibling_error'].append(v)
            #     if any(predict[d] == 1 for d in descendants[v]):
            #         errors['ancestor_missing'] += 1
            #         error_nodes['ancestor_missing'].append(v)
            #     elif any(predict[a] == 1 for a in ancestors[v]):
            #         errors['descendant_missing'] += 1
            #         error_nodes['descendant_missing'].append(v)
            #     elif not any(predict[a] == 1 for a in ancestors[v]) and not any(predict[d] == 1 for d in descendants[v]):
            #         errors['branch_omitted'] += 1
            #         error_nodes['branch_omitted'].append(v)

        tps = tp_matrix[:, n]
        if np.any(tps):
            has_prd_ancestor_tp = (
                np.any(predict[:, ancestors[n]] == 1, axis=1)
                if ancestors[n]
                else np.zeros(len(target), dtype=bool)
            )
            has_tp_descendant = (
                np.any(tp_matrix[:, descendants[n]], axis=1)
                if descendants[n]
                else np.zeros(len(target), dtype=bool)
            )
            has_prd_descendant_tp = (
                np.any(predict[:, descendants[n]] == 1, axis=1)
                if descendants[n]
                else np.zeros(len(target), dtype=bool)
            )
            has_tgt_descendant_tp = (
                np.any(target[:, descendants[n]] == 1, axis=1)
                if descendants[n]
                else np.zeros(len(target), dtype=bool)
            )

            mask_anc_present = tps & has_prd_ancestor_tp
            mask_desc_present = tps & has_tp_descendant
            mask_stopped_level = tps & ~has_prd_descendant_tp & ~has_tgt_descendant_tp

            success["ancestor_present"] += np.sum(mask_anc_present)
            success["descendants_present"] += np.sum(mask_desc_present)
            success["descendants_correctly_omitted"] += np.sum(mask_stopped_level)

            success_per_level[node_level[n]]["ancestor_present"] += np.sum(
                mask_anc_present
            )
            success_per_level[node_level[n]]["descendants_present"] += np.sum(
                mask_desc_present
            )
            success_per_level[node_level[n]]["descendants_correctly_omitted"] += np.sum(
                mask_stopped_level
            )
    total_errors_calculated = sum(errors.values())
    actual_wrong_predictions = np.sum(target != predict)
    errors["totals"] = total_errors_calculated

    assert total_errors_calculated == actual_wrong_predictions, "Mismatch in errors"

    return {
        "errors": errors,
        "errors_per_level": per_level,
        "success": success,
        "success_per_level": success_per_level,
    }


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


def serial_bitflip_error(target, predict, norm=False):
    assert predict.shape == target.shape
    errors = np.sum(predict != target, axis=1).reshape(-1, predict.shape[1])
    average_errors = torch.tensor(np.mean(errors, axis=0))
    if norm:
        prob = np.sum(predict, axis=0) / np.sum(predict)
        return average_errors * prob
    return average_errors


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


def node_importance(labels, hierarchy, norm=False):
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

    error = serial_bitflip_error(target, predict, norm)

    return error


def node_weights(labels, hierarchy, norm=False, alpha=1e-5):
    weights = np.zeros(labels.shape[1])
    parents_map = get_predecessor_map(hierarchy)

    for c in hierarchy.nodes:
        p = parents_map[c]
        if len(p) > 0:
            p = p[0]

        parent_active_mask = labels[:, p] == 1

        if np.sum(parent_active_mask) == 0:
            weights[c] = 0.0
            continue

        P_choose = np.sum(labels[parent_active_mask, c] == 1)

        P_nochoose = np.sum(labels[parent_active_mask, c] == 0)

        weights[c] = np.log((P_choose + alpha) / (P_nochoose + alpha))

    if norm:
        prob = np.sum(labels, axis=0) / np.sum(labels)
        weights *= prob

    return torch.tensor(weights)


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

    pos = nx.drawing.layout.bfs_layout(
        subgraph, start=len(weight) - 1, align="vertical"
    )

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
        labels={
            node: classes[node] for node in filtered_nodes if node != len(weight) - 1
        },
        font_size=8,
    )
    plt.colorbar(nodes, label="Weight Intensity")
    plt.title("Hierarchical Graph Heatmap")
    plt.axis("off")
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close()
