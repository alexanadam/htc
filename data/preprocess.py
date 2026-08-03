import os
import re
import pandas as pd
import numpy as np
import ast
import json
import networkx as nx
import random
import torch
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


def load_saved(path, classes):
    loaded = torch.load(os.path.join(path, "sliced.pt"), weights_only=False)
    return loaded["data"], loaded["labels"]


def load_WOS(path, classes):

    data = pd.read_excel(os.path.join(path, "Data.xlsx"))

    data["Domain"] = data["Domain"].apply(lambda x: [x.strip()])
    data["area"] = data["area"].str.strip()

    data = data[data.groupby("area")["area"].transform("size") >= 10]
    data["labels"] = data["Domain"] + data["area"].apply(lambda x: [x.strip()])

    data["Abstract"] = data["Abstract"].apply(clean_data)
    data["keywords"] = data["keywords"].apply(clean_data)
    # X = data[["Abstract", "keywords"]]
    X = data[["Abstract"]]

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
    # data["labels"] = data["Level1"]

    data["title"] = data["title"].apply(clean_data)
    data["abstract"] = data["abstract"].astype(str).fillna("").apply(clean_data)
    X = data[["title", "abstract", "publisher", "booktitle", "author", "venue"]]
    # X = data[["title", "abstract"]]

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


def get_depths(hierarchy):
    node_depths = np.ones(len(hierarchy))
    roots = [n for n, d in hierarchy.in_degree() if d == 0]

    for root in roots:
        depths = nx.single_source_shortest_path_length(hierarchy, root)
        for node, dist in depths.items():
            node_depths[node] = dist + 1  #18:46

    return node_depths


def get_children(hierarchy):
    return {n: list(hierarchy.successors(n)) for n in hierarchy.nodes()}


import json

def save_to_json(X_subset, Y_subset, classes, output_filename):
    # abstracts = X_subset["Abstract"].tolist()
    abstracts = X_subset.apply(
        lambda row: f"[SEP]".join(row.astype(str)), 
        axis=1
    ).tolist()
    
    data_to_save = []
    
    for text, y_row in zip(abstracts, Y_subset):
        doc_label = [classes[i] for i, val in enumerate(y_row) if val > 0]
        
        doc_dict = {
            'doc_token': text,
            'doc_label': doc_label,
            'doc_topic': [],
            'doc_keyword': []
        }
        data_to_save.append(doc_dict)
        
    with open(output_filename, 'w', encoding='utf-8') as f:
        for item in data_to_save:
            f.write(json.dumps(item) + '\n')


def export_hierarchy(hierarchy, classes, filename):
    classes = classes + ["Root"]
    roots = [n for n, d in hierarchy.in_degree() if d == 0]

    root = roots[0]

    with open(filename, 'w', encoding='utf-8') as f:
        queue = [root]
        visited = {root}

        while queue:
            current_node = queue.pop(0)

            children = list(hierarchy.successors(current_node))
            if children:
                parent_name = classes[current_node]
                children_names = [classes[c] for c in children]
                line_elements = [parent_name] + children_names

                line = "\t".join(line_elements)
                f.write(line + "\n")

                for child in children:
                    if child not in visited:
                        visited.add(child)
                        queue.append(child)
