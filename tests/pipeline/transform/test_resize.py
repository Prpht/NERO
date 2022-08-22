import pytest

from nero.pipeline.transform.resize import *


class TestSmallestCommonShape:
    def test_correctly_assess_common_shape(self):
        embeddings = [
            np.zeros((1, 1, 1, 7)),
            np.zeros((4, 3, 2, 1)),
            np.zeros((3, 3, 3, 3)),
            np.zeros((1, 1, 1, 20)),
        ]

        result = ToSmallestCommonShape.smallest_common_shape(embeddings)

        assert result == (4, 3, 3, 20)

    def test_not_allow_different_dimension_counts(self):
        with pytest.raises(ValueError):
            embeddings = [
                np.zeros((1, 2, 3)),
                np.zeros((2, 2, 2, 2, 2)),
            ]

            ToSmallestCommonShape.smallest_common_shape(embeddings)

    def test_not_allow_empty_embedding_list(self):
        with pytest.raises(ValueError):
            embeddings = []

            ToSmallestCommonShape.smallest_common_shape(embeddings)


class TestResizeWithOuterZeros:
    @pytest.mark.parametrize("embedding, target_shape, expected", [
        pytest.param(
            [[1]],
            (2, 3),
            [[1, 0, 0],
             [0, 0, 0]],
            id="only padding"
        ),
        pytest.param(
            [[1, 2, 3, 4],
             [5, 6, 7, 8]],
            (1, 1),
            [[1]],
            id="only cutting"
        ),
        pytest.param(
            [[1, 2, 3, 4],
             [5, 6, 7, 8]],
            (4, 3),
            [[1, 2, 3],
             [5, 6, 7],
             [0, 0, 0],
             [0, 0, 0]],
            id="one padding, one cutting"
        ),
        pytest.param(
            [[1.0, 2.0, 3.0, 4.0],
             [5.0, 6.0, 7.0, 8.0]],
            (4, 3),
            [[1.0, 2.0, 3.0],
             [5.0, 6.0, 7.0],
             [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]],
            id="floating point support"
        ),
        pytest.param(
            [[[1, 2, 3, 4],
              [5, 6, 7, 8]],
             [[9, 10, 11, 12],
              [13, 14, 15, 16]]],
            (1, 4, 3),
            [[[1, 2, 3],
              [5, 6, 7],
              [0, 0, 0],
              [0, 0, 0]]],
            id="more than two dimensions"
        ),
    ])
    def test_correctly_resizes(self, embedding, target_shape, expected):
        embedding = np.array(embedding)

        result = ToSmallestCommonShape.resize_with_outer_zeros(embedding, target_shape)

        expected = np.array(expected)
        np.testing.assert_array_almost_equal(result, expected)


class TestFlattenUpperTriangles:
    @pytest.mark.parametrize("embedding, expected", [
        pytest.param(
            [[[1, 2, 7],
              [2, 10, 4],
              [7, 4, 5]],
             [[1, 0, 0],
              [0, 0, 4],
              [0, 4, 0]],
             [[4, 4, 4],
              [4, 4, 4],
              [4, 4, 4]]],
            [1, 2, 7, 10, 4, 5, 1, 0, 0, 0, 4, 0, 4, 4, 4, 4, 4, 4],
            id="simple 3D"
        ),
    ])
    def test_correctly_flattens(self, embedding, expected):
        embedding = np.array(embedding)

        result = FlattenUpperTriangles.flatten(embedding)

        expected = np.array(expected)
        np.testing.assert_array_almost_equal(result, expected)
