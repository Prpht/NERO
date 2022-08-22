import networkx as nx
import graph_tool as gt


def get_property_type(value, key=None):
    """
    Performs typing and value conversion for the graph_tool PropertyMap class.
    If a key is provided, it also ensures the key is in a format that can be
    used with the PropertyMap. Returns a tuple, (type name, value, key)
    """
    # Deal with the value
    if isinstance(value, bool):
        type_name = 'bool'

    elif isinstance(value, int):
        type_name = 'float'
        value = float(value)

    elif isinstance(value, float):
        type_name = 'float'

    elif isinstance(value, dict):
        type_name = 'object'

    else:
        type_name = 'string'
        value = str(value)

    return type_name, value, key


def nx2gt(nx_graph: nx.Graph) -> gt.Graph:
    """
    Converts a networkx graph to a graph-tool graph.
    """
    # Phase 0: Create a directed or undirected graph-tool Graph
    gt_graph = gt.Graph(directed=nx_graph.is_directed())

    # Add the Graph properties as "internal properties"
    for key, value in nx_graph.graph.items():
        # Convert the value and key into a type for graph-tool
        type_name, value, key = get_property_type(value, key)

        graph_property = gt_graph.new_graph_property(type_name)  # Create the PropertyMap
        gt_graph.graph_properties[key] = graph_property  # Set the PropertyMap
        gt_graph.graph_properties[key] = value  # Set the actual value

    # Phase 1: Add the vertex and edge property maps
    # Go through all nodes and edges and add seen properties
    # Add the node properties first
    node_properties = set()  # cache keys to only add properties once
    for node, data in nx_graph.nodes(data=True):

        # Go through all the properties if not seen and add them.
        for key, value in data.items():
            if key in node_properties:
                continue  # Skip properties already added

            # Convert the value and key into a type for graph-tool
            type_name, _, key = get_property_type(value, key)

            graph_property = gt_graph.new_vertex_property(type_name)  # Create the PropertyMap
            gt_graph.vertex_properties[key] = graph_property  # Set the PropertyMap

            # Add the key to the already seen properties
            node_properties.add(key)

    # Also add the node id: in NetworkX a node can be any hashable type, but
    # in graph-tool node are defined as indices. So we capture any strings
    # in a special PropertyMap called 'id' -- modify as needed!
    gt_graph.vertex_properties['id'] = gt_graph.new_vertex_property('string')

    # Add the edge properties second
    edge_properties = set()  # cache keys to only add properties once
    for source, destination, data in nx_graph.edges(data=True):

        # Go through all the edge properties if not seen and add them.
        for key, value in data.items():
            if key in edge_properties:
                continue  # Skip properties already added

            # Convert the value and key into a type for graph-tool
            type_name, _, key = get_property_type(value, key)

            graph_property = gt_graph.new_edge_property(type_name)  # Create the PropertyMap
            gt_graph.edge_properties[key] = graph_property  # Set the PropertyMap

            # Add the key to the already seen properties
            edge_properties.add(key)

    # Phase 2: Actually add all the nodes and vertices with their properties
    # Add the nodes
    vertices = {}  # vertex mapping for tracking edges later
    for node, data in nx_graph.nodes(data=True):

        # Create the vertex and annotate for our edges later
        vertex = gt_graph.add_vertex()
        vertices[node] = vertex

        # Set the vertex properties, not forgetting the id property
        data['id'] = str(node)
        for key, value in data.items():
            gt_graph.vp[key][vertex] = value  # vp is short for vertex_properties

    # Add the edges
    for source, destination, data in nx_graph.edges(data=True):

        # Look up the vertex structs from our vertices mapping and add edge.
        edge = gt_graph.add_edge(vertices[source], vertices[destination])

        # Add the edge properties
        for key, value in data.items():
            gt_graph.ep[key][edge] = value  # ep is short for edge_properties

    # Done, finally!
    return gt_graph


if __name__ == '__main__':

    # Create the networkx graph
    nxG = nx.Graph(name="Undirected Graph")
    nxG.add_node("v1", name="alpha", color="red")
    nxG.add_node("v2", name="bravo", color="blue")
    nxG.add_node("v3", name="charlie", color="blue")
    nxG.add_node("v4", name="hub", color="purple")
    nxG.add_node("v5", name="delta", color="red")
    nxG.add_node("v6", name="echo", color="red")

    nxG.add_edge("v1", "v2", weight=0.5, label="follows")
    nxG.add_edge("v1", "v3", weight=0.25, label="follows")
    nxG.add_edge("v2", "v4", weight=0.05, label="follows")
    nxG.add_edge("v3", "v4", weight=0.35, label="follows")
    nxG.add_edge("v5", "v4", weight=0.65, label="follows")
    nxG.add_edge("v6", "v4", weight=0.53, label="follows")
    nxG.add_edge("v5", "v6", weight=0.21, label="follows")

    for item in nxG.edges(data=True):
        print(item)

    # Convert to graph-tool graph
    gtG = nx2gt(nxG)
    gtG.list_properties()
