import pytest

from nero.tools.distributions import *
from nero.embedding.digitisers import *
from tests.embedding.utils import GraphProperty, create_mocked_samples, create_test_graph


class TestEqualProbabilityBinGenerator:
    @pytest.mark.parametrize("bins_no,insight_bins,insight_histogram,expected_result", [
        pytest.param(
            5,
            [0.0, 1.0, 2.0, 4.0, 7.0],
            [1, 10, 2, 7],
            [0.0, 1.3, 1.7, 3.0, 5.285714285714286, 7.0],
            id="simple bins"
        ),
        pytest.param(
            5,
            [0.0, 1.0],
            [13],
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            id="one initial bin"
        ),
        pytest.param(
            4,
            [10.0, 12.5, 20.0],
            [25, 75],
            [10.0, 12.5, 15.0, 17.5, 20.0],
            id="matching edges"
        ),
        pytest.param(
            1,
            [10.0, 13.0, 17.5, 18.5, 20.0],
            [13, 5, 7, 17],
            [10.0, 20.0],
            id="many to single"
        ),
    ])
    def test_equal_probability_bins(self, mocker, bins_no, insight_bins, insight_histogram, expected_result):
        generator = EqualProbabilityBinGenerator(
            bins_no=bins_no,
            insight_bins_generator=mocker.Mock(),
            relevant_property=mocker.Mock(),
        )
        insight_bins = np.array(insight_bins)
        insight_histogram = np.array(insight_histogram)

        result = generator.equal_probability_bins(insight_bins, insight_histogram)

        expected_result = np.array(expected_result)
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_fit(self, mocker):
        graphs = [
            create_test_graph(
                original_elements=(10, 3),
                edges=((0, 1), (1, 2), (0, 2)),
                graph_properties=[
                    GraphProperty(
                        origin=VertexOrigin.NODE,
                        type='float',
                        values=[0.0, 1.0, 2.0, 0.1, 0.2, 0.1, 0.13, 0.6, 1.7, 1.95],
                        name='a',
                    ),
                ]
            ),
            create_test_graph(
                original_elements=(5, 3),
                graph_properties=[
                    GraphProperty(
                        origin=VertexOrigin.NODE,
                        type='float',
                        values=[0.19, 0.20, 0.21, 0.22, 0.21],
                        name='a',
                    ),
                ]
            ),
        ]
        samples = create_mocked_samples(mocker, graphs)
        generator = EqualProbabilityBinGenerator(
            relevant_property=RelevantProperty(
                name='a',
                origin=VertexOrigin.NODE,
            ),
            bins_no=5,
            insight_bins_generator=InsightBinGenerator(
                calculate_bins=equal_size_bins,
                bins_no=10,
                bin_type='equal',
            ),
        )

        generator.fit(samples, lowest_index=0)

        assert generator.effective_bin_no == 8
        expected_bin_edges = np.array([np.NINF, 0.0, 0.12, 0.24, 0.36, 1.6, 2.0, np.PINF])
        np.testing.assert_array_almost_equal(generator.bin_edges_, expected_bin_edges)
        assert generator.id() == "a__5__equal_10__equal_probability"
