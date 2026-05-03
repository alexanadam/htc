import os
import re
import pandas as pd
import numpy as np
import ast
import json
import networkx as nx
import random
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn.preprocessing import MultiLabelBinarizer


def init_seed(seed=156):
    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.mps.manual_seed(seed)


def create_WOS_hierarchy(path, output):

    df = pd.read_excel(os.path.join(path, "Data.xlsx"))

    hierarchy = defaultdict(set)

    for _, row in df.iterrows():
        domain = str(row["Domain"]).strip()
        area = str(row["area"]).strip()

        if domain and area:
            hierarchy[domain].add(area)

    root = {"name": "root", "children": []}

    for domain, areas in sorted(hierarchy.items()):
        root["children"].append(
            {"name": domain, "children": [{"name": area} for area in sorted(areas)]}
        )

    with open(output, "w", encoding="utf-8") as f:
        json.dump(root, f, indent=2, ensure_ascii=False)

    return root


def load_WOS(path, classes):

    data = pd.read_excel(os.path.join(path, "Data.xlsx"))

    data["Domain"] = data["Domain"].apply(lambda x: [x.strip()])
    data["area"] = data["area"].str.strip()

    maps = data.groupby("Y")["area"].agg(lambda x: list(set(x)))
    data["level2"] = data["Y"].map(maps)

    data["labels"] = data["Domain"] + data["level2"]

    data["Abstract"] = data["Abstract"].apply(clean_data)
    data["keywords"] = data["keywords"].apply(clean_data)
    X = data[["Abstract", "keywords"]]

    mlb = MultiLabelBinarizer(classes=classes)
    Y = mlb.fit_transform(data["labels"])

    return X, Y.astype(np.float32)


def load_FORC(path, classes, with_weak=True):
    data = pd.read_csv(os.path.join(path, "train.csv"))
    if with_weak:
        weak = pd.read_csv(os.path.join(path, "weaklylabeled.csv"))
        weak = weak[
            [
                "acl_id",
                "Level1",
                "Level2",
                "Level3",
                "abstract",
                "url",
                "publisher",
                "year",
                "month",
                "booktitle",
                "author",
                "title",
                "doi",
                "venue",
                "data_index",
                "full_text",
            ]
        ]
        data = pd.concat([data, weak], ignore_index=True)

    data["Level1"] = data["Level1"].apply(ast.literal_eval)

    def python_parser(value):
        return [] if isinstance(value, float) else ast.literal_eval(value)

    data["Level2"] = data["Level2"].apply(python_parser)
    data["Level3"] = data["Level3"].apply(python_parser)
    data["labels"] = data["Level1"] + data["Level2"] + data["Level3"]

    data["title"] = data["title"].apply(clean_data)
    data["abstract"] = data["abstract"].astype(str).fillna("").apply(clean_data)
    X = data[["title", "abstract", "publisher", "booktitle", "author", "venue"]]

    mlb = MultiLabelBinarizer(classes=classes)
    Y = mlb.fit_transform(data["labels"])

    return X, Y.astype(np.float32)


def clean_data(text):
    to_remove = [".", '"', "-", "="]

    to_replace = ["[", "]", "(", ")", ","]

    for char in to_remove:
        text = text.replace(char, "")

    for char in to_replace:
        text = text.replace(char, " ")

    rules = [
        (r">\s+", ">"),
        (r"\s+", " "),
        (r"\s*<br\s*/?>\s*", "\n"),
        (r"</div\s*>\s*", "\n"),
        (r"</(p|h\d)\s*>\s*", "\n\n"),
        (r"<head>.*?<\s*(/head|body)[^>]*>", ""),
        (r'<a\s+href="([^"]+)"[^>]*>.*?</a>', r"\1"),
        (r"[ \t]*<[^<]*?/?>", ""),
        (r"^\s+", ""),
    ]

    for pattern, repl in rules:
        text = re.sub(pattern, repl, text, flags=re.DOTALL)
    return text.strip().lower()


def get_classes(hierarchy_path, root_excl=None):
    with open(hierarchy_path) as f:
        hierarchy = json.load(f)
    classes = []

    def dfs_traverse(node):
        if "name" in node:
            name = node["name"].strip()
            if root_excl is None or name != root_excl:
                classes.append(name)
        for child in node.get("children", []):
            dfs_traverse(child)

    dfs_traverse(hierarchy)

    return list(sorted(set(classes)))


def get_hierarchy(hierarchy_path, nodes, root_excl=None):
    with open(hierarchy_path) as f:
        hierarchy = json.load(f)

    graph = nx.DiGraph()
    mapping = {value: idx for idx, value in enumerate(nodes)}
    # if root is not None:
    #     mapping[root] = len(nodes)

    def add_nodes_edges(data, parent=None):
        name = data["name"].strip()
        if name not in mapping and root_excl is None:
            mapping[name] = max(mapping.values()) + 1
        if name != root_excl:
            graph.add_node(mapping[name])

            if parent is not None:
                graph.add_edge(parent, mapping[name])
            parent = mapping[name]
        else:
            parent = None

        if "children" in data:
            for child in data["children"]:
                add_nodes_edges(child, parent)

    add_nodes_edges(hierarchy)

    return graph


def get_parents(hierarchy, node):
    return nx.ancestors(hierarchy, node)
    # if root is not None:
    #     parents.discard(root)
    # return parents


def get_predecessor_map(hierarchy):
    parent_map = {}
    for node in hierarchy.nodes:
        parents = hierarchy.predecessors(node)
        parent_map[node] = list(parents)

    return parent_map


def get_parents_list(hierarchy):
    parent_map = {}
    for node in hierarchy.nodes:
        parents = get_parents(hierarchy, node)
        parent_map[node] = parents

    return parent_map


def remove_root(hierarchy):
    root = [node for node in hierarchy.nodes() if hierarchy.in_degree(node) == 0]
    if len(root) == 1:
        hierarchy.remove_node(root[0])
    return hierarchy


def get_all_shortest_paths(hierarchy):
    return dict(nx.all_pairs_shortest_path_length(hierarchy))


def get_levels(hierarchy):
    root = [node for node in hierarchy.nodes() if hierarchy.in_degree(node) == 0][0]
    grouped = defaultdict(list)
    for node in nx.descendants(hierarchy, root):
        level = nx.shortest_path_length(hierarchy, root, node)
        grouped[level].append(node)
    return grouped


def plot_importance(weight, hierarchy, classes, path):
    filtered_nodes = [node for node in hierarchy.nodes() if weight[node] > 0.0056]
    filtered_weights = [w for w in weight if w > 0.0056]

    subgraph = hierarchy.subgraph(filtered_nodes)

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
