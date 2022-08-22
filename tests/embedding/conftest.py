import graph_tool as gt
import pytest

from nero.embedding.digitisers import VertexOrigin
from tests.embedding.utils import GraphProperty, create_test_graph


@pytest.fixture(scope='package')
def mock_leaf_graph() -> gt.Graph:
    return create_test_graph(
        original_elements=(6, 8),
        edges=(
            (0, 6), (6, 1), (1, 8), (1, 7), (8, 4), (7, 2), (2, 9), (2, 10), (2, 12), (9, 4), (12, 3), (10, 5), (4, 13),
            (13, 3), (3, 11), (11, 5)
        ),
        graph_properties=[
            GraphProperty(
                origin=VertexOrigin.EDGE,
                type='float',
                values=[1.0, 1.0, 2.24, 1.41, 3.16, 3.0, 1.0, 1.0],
                name='edge_length',
            ),
            GraphProperty(
                origin=VertexOrigin.EDGE,
                type='float',
                values=[4.5, 4.1, 1.1, 2.2, 1.2, 0.5, 3.9, 0.6],
                name='edge_diameter',
            )
        ]
    )
