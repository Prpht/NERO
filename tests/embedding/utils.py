from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import graph_tool as gt

from nero.embedding.digitisers import VertexOrigin
from nero.tools.datasets import PersistedClassificationSample


@dataclass
class GraphProperty:
    origin: VertexOrigin
    type: str
    values: List[Any]
    name: str


def create_test_graph(
        original_elements: Tuple[int, int],
        edges: Sequence[Tuple[int, int]] = (),
        graph_properties: Sequence[GraphProperty] = (),
) -> gt.Graph:
    graph = gt.Graph(directed=False)
    original_nodes, original_edges = original_elements
    graph.add_vertex(original_nodes + original_edges)
    for edge_from, edge_to in edges:
        graph.add_edge(graph.vertex(edge_from), graph.vertex(edge_to))
    original_node_values = original_nodes * [True] + original_edges * [False]
    original_node_property = graph.new_vertex_property('bool', vals=original_node_values)
    graph.vertex_properties['original_node'] = original_node_property

    for graph_property in graph_properties:
        if graph_property.origin == VertexOrigin.NODE:
            values = graph_property.values + original_edges * [0.0]
        else:
            values = original_nodes * [0.0] + graph_property.values
        materialised_property = graph.new_vertex_property(graph_property.type, vals=values)
        graph.vertex_properties[graph_property.name] = materialised_property

    return graph


def create_mocked_samples(mocker, graphs: Sequence[gt.Graph]) -> List[PersistedClassificationSample]:
    dataset = []
    for graph in graphs:
        mock_sample = mocker.Mock(PersistedClassificationSample)
        mock_sample.materialise().graph = graph
        dataset.append(mock_sample)
    return dataset
