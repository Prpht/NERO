from typing import Set, Union, Tuple

import networkx as nx


def materialize_subgraph(graph: nx.Graph, nodes_to_save: Union[nx.Graph, Set]) -> nx.Graph:
    """Returns the largest connected components of a given graph"""
    graph_class = graph.__class__
    subgraph = graph_class()
    subgraph.add_nodes_from((node, graph.nodes[node]) for node in nodes_to_save)
    subgraph.add_edges_from(
        (node, neighbour, data)
        for node, neighbours in graph.adj.items() if node in nodes_to_save
        for neighbour, data in neighbours.items() if neighbour in nodes_to_save
    )
    subgraph.graph.update(graph.graph)
    return subgraph


def edges_into_nodes(graph: nx.Graph) -> nx.Graph:
    """Transforms all edges in a graph into nodes connected to initial begin and end nodes."""

    def node_id_for_edge() -> Tuple:
        node_id: Tuple = (edge_from, edge_to)
        conflict_resolver = 0
        while node_id in graph:
            node_id = (edge_from, edge_to, conflict_resolver)
            conflict_resolver += 1
        return node_id

    result = nx.Graph(**graph.graph)
    result.add_nodes_from(graph.nodes(data=True))
    for node in result.nodes:
        result.nodes[node]['original_node'] = True
    for edge_from, edge_to, data in graph.edges(data=True):
        edge_node_id = node_id_for_edge()
        result.add_node(edge_node_id, **data)
        result.nodes[edge_node_id]['original_node'] = False
        result.add_edge(edge_from, edge_node_id)
        result.add_edge(edge_node_id, edge_to)
    return result
