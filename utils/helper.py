import networkx as nx
import json
import numpy as np
import matplotlib.pyplot as plt


def get_hierarchy1(hierarchy_path, nodes, root=None):
    with open(hierarchy_path) as f:
        hierarchy = json.load(f)

    graph = nx.DiGraph()
    mapping = {value: idx for idx, value in enumerate(nodes)}
    if root is not None:
        mapping[root] = len(nodes)

    def add_nodes_edges(data, parent=None):
        graph.add_node(mapping[data["name"].strip()])

        if parent is not None:
            graph.add_edge(parent, mapping[data["name"].strip()])

        if "children" in data:
            for child in data["children"]:
                add_nodes_edges(child, mapping[data["name"].strip()])

    add_nodes_edges(hierarchy)

    return graph


def get_parents1(hierarchy, node, root=None):
    parents = nx.ancestors(hierarchy, node)
    if root is not None:
        parents.discard(root)
    return parents


def get_parents_list1(hierarchy, root=None):
    parent_map = {}
    for node in hierarchy.nodes:
        parents = get_parents1(hierarchy, node, root=root)
        parent_map[node] = parents

    return parent_map


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
            markersize=4
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
    means = [metric['mean'] for metric in metrics.values()]
    stds = [metric['std'] for metric in metrics.values()]

    plt.figure(figsize=(10, 6))
    x_positions = np.arange(len(metrics))

    plt.bar(
        x_positions,
        means,
        yerr=stds,
        capsize=8,
        color='steelblue',
        edgecolor='black',
        alpha=0.8
    )

    plt.title('Cross-Validation Summary: Final Metrics', fontsize=14)
    plt.ylabel('Score')

    plt.xticks(x_positions, metrics.keys(), rotation=30, ha='right')
    plt.yticks(np.arange(0, 1.1, 0.1))

    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()

    plt.savefig(path, dpi=300, bbox_inches='tight')
    logger.info(f"Combined scalar metrics plot saved successfully to {path}")
    plt.close()
