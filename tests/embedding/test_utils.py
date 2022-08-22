from nero.embedding.utils import *


class TestAddEmbeddings:
    def test_addition_without_resize(self):
        a = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ])
        b = np.array([
            [9.0, 7.0],
            [5.0, 3.0]
        ])

        result = add_embeddings(a, b)

        expected_result = np.array([
            [10.0, 9.0],
            [8.0, 7.0],
        ])
        np.testing.assert_array_equal(result, expected_result)

    def test_addition_with_first_larger(self):
        a = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ])
        b = np.array([
            [9.0, 7.0],
            [5.0, 3.0],
        ])

        result = add_embeddings(a, b)

        expected_result = np.array([
            [10.0, 9.0],
            [8.0, 7.0],
            [5.0, 6.0],
        ])
        np.testing.assert_array_equal(result, expected_result)

    def test_addition_with_second_larger(self):
        a = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ])
        b = np.array([
            [9.0, 7.0, 1.0],
            [5.0, 3.0, 1.0],
        ])

        result = add_embeddings(a, b)

        expected_result = np.array([
            [10.0, 9.0, 1.0],
            [8.0, 7.0, 1.0],
        ])
        np.testing.assert_array_equal(result, expected_result)

    def test_addition_with_both_different(self):
        a = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ])
        b = np.array([
            [9.0, 7.0, 1.0],
            [5.0, 3.0, 1.0],
        ])

        result = add_embeddings(a, b)

        expected_result = np.array([
            [10.0, 9.0, 1.0],
            [8.0, 7.0, 1.0],
            [5.0, 6.0, 0.0],
        ])
        np.testing.assert_array_equal(result, expected_result)
