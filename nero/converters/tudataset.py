from dataclasses import dataclass
import pathlib
import os.path
from typing import List, Tuple

import networkx as nx
from torch_geometric.datasets import TUDataset
from tqdm import tqdm

import nero.constants as constants
import nero.tools.datasets as datasets
import nero.tools.graphs as graphs


def separate_labels_and_attributes(graph: nx.Graph) -> nx.Graph:
    result = nx.Graph()
    result.graph['category'] = graph.graph['classes'][0]
    result.add_nodes_from(graph.nodes)
    for node in graph.nodes:
        try:
            labels = graph.nodes[node]['labels']
            for i, label in enumerate(labels):
                result.nodes[node][f'node_label_{i}'] = label
        except KeyError:
            pass
        try:
            attributes = graph.nodes[node]['attributes']
            for i, attribute in enumerate(attributes):
                result.nodes[node][f'node_attribute_{i}'] = attribute
        except KeyError:
            pass
    for edge_from, edge_to, edge_data in graph.edges(data=True):
        result.add_edge(edge_from, edge_to)
        try:
            labels = edge_data['labels']
            for i, label in enumerate(labels):
                result.edges[edge_from, edge_to][f'edge_label_{i}'] = label
        except KeyError:
            pass
        try:
            attributes = edge_data['attributes']
            for i, attribute in enumerate(attributes):
                result.edges[edge_from, edge_to][f'edge_attribute_{i}'] = attribute
        except KeyError:
            pass
    return result


def tudataset2nx(dataset_name: str) -> List[nx.Graph]:
    downloads = pathlib.Path(constants.DOWNLOADS_DIR)

    with open(downloads / dataset_name / "raw" / (dataset_name + "_graph_indicator.txt"), "r") as f:
        graph_indicator = [int(i) - 1 for i in list(f)]

    num_graphs = max(graph_indicator)
    node_indices = []
    offset = []
    c = 0

    for i in range(num_graphs + 1):
        offset.append(c)
        c_i = graph_indicator.count(i)
        node_indices.append((c, c + c_i - 1))
        c += c_i

    dataset = []
    for i in node_indices:
        g = nx.Graph()
        for j in range(i[1] - i[0] + 1):
            g.add_node(j)

        dataset.append(g)

    with open(downloads / dataset_name / "raw" / (dataset_name + "_A.txt"), "r") as f:
        edges = [i.split(',') for i in list(f)]

    edges = [(int(e[0].strip()) - 1, int(e[1].strip()) - 1) for e in edges]
    edge_list = []
    edgeb_list = []
    for e in edges:
        g_id = graph_indicator[e[0]]
        g = dataset[g_id]
        off = offset[g_id]

        if ((e[0] - off, e[1] - off) not in list(g.edges())) and ((e[1] - off, e[0] - off) not in list(g.edges())):
            g.add_edge(e[0] - off, e[1] - off)
            edge_list.append((e[0] - off, e[1] - off))
            edgeb_list.append(True)
        else:
            edgeb_list.append(False)

    if os.path.exists(downloads / dataset_name / "raw" / (dataset_name + "_node_labels.txt")):
        with open(downloads / dataset_name / "raw" / (dataset_name + "_node_labels.txt"), "r") as f:
            node_labels = [str.strip(i) for i in list(f)]

        node_labels = [i.split(',') for i in node_labels]
        int_labels = []
        for i in range(len(node_labels)):
            int_labels.append([int(j) for j in node_labels[i]])

        i = 0
        for g in dataset:
            for v in range(g.number_of_nodes()):
                g.nodes[v]['labels'] = int_labels[i]
                i += 1

    if os.path.exists(downloads / dataset_name / "raw" / (dataset_name + "_node_attributes.txt")):
        with open(downloads / dataset_name / "raw" / (dataset_name + "_node_attributes.txt"), "r") as f:
            node_attributes = [str.strip(i) for i in list(f)]

        node_attributes = [i.split(',') for i in node_attributes]
        float_attributes = []
        for i in range(len(node_attributes)):
            float_attributes.append([float(j) for j in node_attributes[i]])
        i = 0
        for g in dataset:
            for v in range(g.number_of_nodes()):
                g.nodes[v]['attributes'] = float_attributes[i]
                i += 1

    if os.path.exists(downloads / dataset_name / "raw" / (dataset_name + "_edge_labels.txt")):
        with open(downloads / dataset_name / "raw" / (dataset_name + "_edge_labels.txt"), "r") as f:
            edge_labels = [str.strip(i) for i in list(f)]

        edge_labels = [i.split(',') for i in edge_labels]
        e_labels = []
        for i in range(len(edge_labels)):
            if edgeb_list[i]:
                e_labels.append([int(j) for j in edge_labels[i]])

        i = 0
        for g in dataset:
            for e in range(g.number_of_edges()):
                g.edges[edge_list[i]]['labels'] = e_labels[i]
                i += 1

    if os.path.exists(downloads / dataset_name / "raw" / (dataset_name + "_edge_attributes.txt")):
        with open(downloads / dataset_name / "raw" / (dataset_name + "_edge_attributes.txt"), "r") as f:
            edge_attributes = [str.strip(i) for i in list(f)]

        edge_attributes = [i.split(',') for i in edge_attributes]
        e_attributes = []
        for i in range(len(edge_attributes)):
            if edgeb_list[i]:
                e_attributes.append([float(j) for j in edge_attributes[i]])

        i = 0
        for g in dataset:
            for e in range(g.number_of_edges()):
                g.edges[edge_list[i]]['attributes'] = e_attributes[i]
                i += 1

    if os.path.exists(downloads / dataset_name / "raw" / (dataset_name + "_graph_labels.txt")):
        with open(downloads / dataset_name / "raw" / (dataset_name + "_graph_labels.txt"), "r") as f:
            classes = [str.strip(i) for i in list(f)]
        classes = [i.split(',') for i in classes]
        cs = []
        for i in range(len(classes)):
            cs.append([int(j) for j in classes[i]])

        i = 0
        for g in dataset:
            g.graph['classes'] = cs[i]
            i += 1

    if os.path.exists(downloads / dataset_name / "raw" / (dataset_name + "_graph_attributes.txt")):
        with open(downloads / dataset_name / "raw" / (dataset_name + "_graph_attributes.txt"),
                  "r") as f:
            targets = [str.strip(i) for i in list(f)]

        targets = [i.split(',') for i in targets]
        ts = []
        for i in range(len(targets)):
            ts.append([float(j) for j in targets[i]])

        i = 0
        for g in dataset:
            g.graph['targets'] = ts[i]
            i += 1

    dataset = [separate_labels_and_attributes(graph) for graph in dataset]

    return dataset


@dataclass
class TUDatasetDescription:
    name: str
    node_labels: int
    edge_labels: int
    node_attributes: int
    edge_attributes: int


def discover_labels_and_attributes(dataset_name: str, graph: nx.Graph) -> TUDatasetDescription:
    node_labels = 0
    while len(nx.get_node_attributes(graph, f"node_label_{node_labels}")) > 0:
        node_labels += 1
    edge_labels = 0
    while len(nx.get_edge_attributes(graph, f"edge_label_{edge_labels}")) > 0:
        edge_labels += 1
    node_attributes = 0
    while len(nx.get_node_attributes(graph, f"node_attribute_{node_attributes}")) > 0:
        node_attributes += 1
    edge_attributes = 0
    while len(nx.get_edge_attributes(graph, f"node_attribute_{edge_attributes}")) > 0:
        edge_attributes += 1
    return TUDatasetDescription(
        name=dataset_name,
        node_labels=node_labels,
        edge_labels=edge_labels,
        node_attributes=node_attributes,
        edge_attributes=edge_attributes,
    )


def tudataset2persisted(
        dataset_name: str
) -> Tuple[List[datasets.PersistedClassificationSample], List[int], TUDatasetDescription]:
    TUDataset(constants.DOWNLOADS_DIR, dataset_name)
    dataset = tudataset2nx(dataset_name)
    tudataset_description = discover_labels_and_attributes(dataset_name, dataset[0])
    target_classes = [graph.graph['category'] for graph in dataset]
    converted_dataset = [
        graphs.edges_into_nodes(graph)
        for graph in tqdm(dataset, desc="Converting to a bipartite form")
    ]
    converted_dataset = [
        datasets.create_persisted_sample(graph, dataset_name, i, 'category')
        for i, graph in enumerate(tqdm(converted_dataset, desc="Creating persisted samples"))
    ]
    return converted_dataset, target_classes, tudataset_description
