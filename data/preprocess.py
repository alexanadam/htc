import os
import re
import pandas as pd
import numpy as np
import ast
import json
import networkx as nx
from collections import defaultdict

from sklearn.preprocessing import MultiLabelBinarizer


def create_WOS_hierarchy(path, output):

    df = pd.read_excel(os.path.join(path, "Data.xlsx"))

    hierarchy = defaultdict(set)

    for _, row in df.iterrows():
        domain = str(row["Domain"]).strip()
        area = str(row["area"]).strip()

        if domain and area:
            hierarchy[domain].add(area)

    root = {
        "name": "root",
        "children": []
    }

    for domain, areas in sorted(hierarchy.items()):
        root["children"].append({
            "name": domain,
            "children": [{"name": area} for area in sorted(areas)]
        })

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


def load_FORC(path, classes, with_weak=False):
    df = pd.read_csv(os.path.join(path, "train.csv"))
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
        df = pd.concat([df, weak], ignore_index=True)

    df["Level1"] = df["Level1"].apply(ast.literal_eval)

    def python_parser(value):
        return [] if isinstance(value, float) else ast.literal_eval(value)

    df["Level2"] = df["Level2"].apply(python_parser)
    df["Level3"] = df["Level3"].apply(python_parser)
    df["labels"] = df["Level1"] + df["Level2"] + df["Level3"]

    df["title"] = df["title"].apply(clean_data)
    df["abstract"] = df["abstract"].apply(clean_data)
    X = df[["title", "abstract", "publisher", "booktitle", "author", "venue"]]

    # X = data.drop(columns=['labels'])
    Y = df["labels"]
    mlb = MultiLabelBinarizer(classes=classes)
    Y = mlb.fit_transform(Y)

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

    return list(set(classes))


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
