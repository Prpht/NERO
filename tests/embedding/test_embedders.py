import pytest

from nero.embedding.embedders import *
from nero.embedding.digitisers import *
from tests.embedding.utils import GraphProperty, create_mocked_samples, create_test_graph


@pytest.fixture(scope='module')
def embedding_slice() -> np.ndarray:
    return np.array([
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [0, 2, 0, 2, 0, 1, 0, 0, 0, 1],
        [2, 1, 1, 0, 2, 0, 0, 1, 2, 0],
        [0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
    ])


@pytest.fixture(scope='module')
def digitised_properties() -> np.ndarray:
    return np.array([
        [0, 1, 0, 0, 1, 1, 1],
        [4, 3, 2, 4, 3, 2, 4],
        [9, 9, 8, 8, 5, 6, 7],
    ])


class TestNeroEmbedder:
    # TODO: fails for empty graph
    def test_track_source_consistency(self, mocker, mock_leaf_graph):
        dataset = create_mocked_samples(mocker, [mock_leaf_graph])
        bin_generators = [
            EqualSizeBinGenerator(
                relevant_property=RelevantProperty('edge_length', VertexOrigin.EDGE),
                bins_no=10,
            ),
            EqualSizeBinGenerator(
                relevant_property=RelevantProperty('edge_diameter', VertexOrigin.EDGE),
                bins_no=20,
            ),
        ]
        embedder = NeroEmbedder(bin_generators=bin_generators, jobs_no=1)

        embedder.fit(dataset)
        result = embedder.transform_graph(mock_leaf_graph, track_sources=False)
        result_with_tracks = embedder.transform_graph(mock_leaf_graph, track_sources=True)
        result_with_tracks_compressed = np.sum(result_with_tracks, axis=3)

        np.testing.assert_array_equal(result * 2, result_with_tracks_compressed)

    @pytest.mark.parametrize("graphs, bin_generators, expected_result", [
        pytest.param(
            [
                create_test_graph(
                    original_elements=(0, 0),
                    graph_properties=[
                        GraphProperty(
                            origin=VertexOrigin.NODE,
                            type='float',
                            values=[],
                            name='a',
                        )
                    ]
                ),
            ],
            [
                EqualSizeBinGenerator(
                    relevant_property=RelevantProperty('a', VertexOrigin.NODE),
                    bins_no=2,
                ),
            ],
            [
                np.array([
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ],
                ]),
            ],
            id="zero node graph"
        ),
        pytest.param(
            [
                create_test_graph(
                    original_elements=(1, 0),
                    graph_properties=[
                        GraphProperty(
                            origin=VertexOrigin.NODE,
                            type='float',
                            values=[3.14],
                            name='a',
                        )
                    ]
                ),
            ],
            [
                EqualSizeBinGenerator(
                    relevant_property=RelevantProperty('a', VertexOrigin.NODE),
                    bins_no=2,
                ),
            ],
            [
                np.array([
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0],
                    ],
                ]),
            ],
            id="one node graph"
        ),
        pytest.param(
            [
                create_test_graph(
                    original_elements=(2, 1),
                    edges=((0, 2), (2, 1)),
                    graph_properties=[
                        GraphProperty(
                            origin=VertexOrigin.NODE,
                            type='float',
                            values=[3.14, 2.71],
                            name='a',
                        )
                    ]
                ),
            ],
            [
                EqualSizeBinGenerator(
                    relevant_property=RelevantProperty('a', VertexOrigin.NODE),
                    bins_no=2,
                ),
            ],
            [
                np.array([
                    [
                        [1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1],
                    ],
                    [
                        [0, 0, 1, 0, 1],
                        [0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                    ],
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                    ],
                ]),
            ],
            id="two node graph"
        ),
        pytest.param(
            [
                create_test_graph(
                    original_elements=(3, 3),
                    edges=((0, 3), (3, 1), (1, 4), (4, 2), (2, 5), (5, 0)),
                    graph_properties=[
                        GraphProperty(
                            origin=VertexOrigin.EDGE,
                            type='float',
                            values=[1.1, 1.2, 1.2],
                            name='b',
                        )
                    ]
                ),
            ],
            [
                EqualSizeBinGenerator(
                    relevant_property=RelevantProperty('b', VertexOrigin.EDGE),
                    bins_no=2,
                ),
            ],
            [
                np.array([
                    [
                        [3, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 2],
                    ],
                    [
                        [0, 0, 2, 0, 4],
                        [0, 0, 0, 0, 0],
                        [2, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [4, 0, 0, 0, 0],
                    ],
                    [
                        [6, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 2],
                        [0, 0, 0, 0, 0],
                        [0, 0, 2, 0, 2],
                    ],
                    [
                        [0, 0, 1, 0, 2],
                        [0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [2, 0, 0, 0, 0],
                    ],
                ]),
            ],
            id="three node graph"
        ),
        pytest.param(
            [
                create_test_graph(
                    original_elements=(4, 3),
                    edges=((0, 4), (4, 1), (1, 5), (5, 2), (1, 6), (6, 3)),
                    graph_properties=[
                        GraphProperty(
                            origin=VertexOrigin.NODE,
                            type='float',
                            values=[1.0, 2.5, 3.5, 4.0],
                            name='a',
                        ),
                        GraphProperty(
                            origin=VertexOrigin.EDGE,
                            type='float',
                            values=[1.1, 1.2, 1.2],
                            name='b',
                        )
                    ]
                ),
            ],
            [
                EqualSizeBinGenerator(
                    relevant_property=RelevantProperty('a', VertexOrigin.NODE),
                    bins_no=3,
                ),
                EqualSizeBinGenerator(
                    relevant_property=RelevantProperty('b', VertexOrigin.EDGE),
                    bins_no=2,
                ),
            ],
            [
                np.array([
                    [
                        [3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 1, 4, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                    ],
                    [
                        [0, 0, 1, 3, 1, 1, 6, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [6, 0, 0, 0, 0, 0, 0, 0, 2, 0, 4],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0, 2, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 2, 1, 1, 4, 0, 0, 0, 0],
                    ],
                    [
                        [6, 0, 0, 0, 0, 0, 0, 0, 2, 0, 4],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 1, 1, 3, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 3, 1, 1, 6, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [4, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2],
                    ],
                    [
                        [0, 0, 2, 0, 2, 2, 6, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                        [2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                        [6, 0, 0, 0, 0, 0, 0, 0, 2, 0, 4],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 2, 0, 1, 1, 4, 0, 0, 0, 0],
                    ],
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 1, 2, 0, 0, 0, 0],
                        [0, 0, 1, 0, 1, 0, 2, 0, 0, 0, 0],
                        [0, 0, 2, 0, 2, 2, 6, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                ])
            ],
            id="four node graph"
        )
    ])
    def test_fit_transform(
            self,
            mocker,
            graphs: List[gt.Graph],
            bin_generators: List[BinGenerator],
            expected_result: np.ndarray
    ) -> None:
        samples = create_mocked_samples(mocker, graphs)
        embedder = NeroEmbedder(bin_generators=bin_generators, jobs_no=1)

        result = embedder.fit_transform(samples)

        for embedding, expected_embedding in zip(result, expected_result):
            np.testing.assert_array_equal(embedding, expected_embedding)

    def test_leaf_fit_transform(self, mocker, mock_leaf_graph):
        dataset = create_mocked_samples(mocker, [mock_leaf_graph])
        bin_generators = [
            EqualSizeBinGenerator(
                relevant_property=RelevantProperty('edge_length', VertexOrigin.EDGE),
                bins_no=10,
            ),
            EqualSizeBinGenerator(
                relevant_property=RelevantProperty('edge_diameter', VertexOrigin.EDGE),
                bins_no=20,
            ),
        ]
        embedder = NeroEmbedder(bin_generators=bin_generators, jobs_no=1)

        result = embedder.fit_transform(dataset)

        assert result[0].shape == (8, 36, 36)

    def test_fill_slice(self, digitised_properties, embedding_slice):
        result_slice = np.zeros((4, 10), dtype='int')
        distances = np.array([0, 1, 2, 2, 1, 3, 2])

        NeroEmbedder.fill_slice(result_slice, distances, digitised_properties)

        np.testing.assert_array_equal(result_slice, embedding_slice)

    def test_update_embedding(self, digitised_properties, embedding_slice):
        embedding = np.zeros((1, 10, 10), dtype='int')
        node = 3

        result = NeroEmbedder.update_embedding(embedding, embedding_slice, digitised_properties, node)

        expected_result = np.array([
            [
                [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 2, 0, 2, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 2, 0, 2, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 2, 0, 2, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [2, 1, 1, 0, 2, 0, 0, 1, 2, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 1, 1, 0, 2, 0, 0, 1, 2, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 1, 1, 0, 2, 0, 0, 1, 2, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ])
        np.testing.assert_array_equal(result, expected_result)
