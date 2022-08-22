import pytest

from nero.pipeline.transform.normalise import *


class TestPercentOfMaxPerRelationsOrder:
    @pytest.mark.parametrize("embedding, expected", [
        pytest.param(
            [[1, 2, 3, 4],
             [5, 6, 7, 10],
             [1, 1, 1, 1]],
            [[0.25, 0.5, 0.75, 1.0],
             [0.5, 0.6, 0.7, 1.0],
             [1.0, 1.0, 1.0, 1.0]],
            id="two dimensional"
        ),
        pytest.param(
            [[[1, 2],
              [3, 4]],
             [[1, 2],
              [3, 5]],
             [[4, 4],
              [4, 4]]],
            [[[0.25, 0.5],
              [0.75, 1.0]],
             [[0.2, 0.4],
              [0.6, 1.0]],
             [[1.0, 1.0],
              [1.0, 1.0]]],
            id="three dimensional"
        ),
        pytest.param(
            [[1, 2, 3, 4],
             [5, 6, 7, 10],
             [0, 0, 0, 0]],
            [[0.25, 0.5, 0.75, 1.0],
             [0.5, 0.6, 0.7, 1.0],
             [0.0, 0.0, 0.0, 0.0]],
            id="support zeros"
        ),
    ])
    def test_correctly_normalises(self, embedding, expected):
        embedding = np.array(embedding)

        result = PercentOfMaxPerRelationOrder.normalise(embedding)

        expected = np.array(expected)
        np.testing.assert_array_almost_equal(result, expected)


class TestPercentOfRow:
    @pytest.mark.parametrize("embedding, expected", [
        pytest.param(
            [[[1, 2, 7],
              [5, 10, 5],
              [3, 4, 5]],
             [[1, 2, 7],
              [5, 5, 5],
              [3, 4, 5]],
             [[4, 4, 4],
              [4, 4, 4],
              [4, 4, 4]]],
            [[[0.1, 0.2, 0.7],
              [0.25, 0.5, 0.25],
              [0.25, 0.333333, 0.416666]],
             [[0.1, 0.2, 0.7],
              [0.333333, 0.333333, 0.333333],
              [0.25, 0.333333, 0.416666]],
             [[0.333333, 0.333333, 0.333333],
              [0.333333, 0.333333, 0.333333],
              [0.333333, 0.333333, 0.333333]]],
            id="three dimensional"
        ),
        pytest.param(
            [[[0, 0],
              [0, 0]],
             [[4, 4],
              [4, 4]]],
            [[[0.0, 0.0],
              [0.0, 0.0]],
             [[0.5, 0.5],
              [0.5, 0.5]]],
            id="support zeros"
        ),
    ])
    def test_correctly_normalises(self, embedding, expected):
        embedding = np.array(embedding)

        result = PercentOfRow.normalise(embedding)

        expected = np.array(expected)
        np.testing.assert_array_almost_equal(result, expected)
