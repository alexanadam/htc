import networkx as nx
import json


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
