import pytest

from nero.embedding.compressors import *
from tests.embedding.utils import create_mocked_samples


class TestNoCompressionROC:
    def test_identity(self):
        distances = np.array([1, 3, 6, 7, 3, 3, 1, 0, 1, 17, 42])
        roc = NoCompressionROC()

        result = roc.compress(distances)

        np.testing.assert_almost_equal(distances, result)


class TestBinBasedROC:
    def test_leading_bins(self, mocker):
        mocker.patch.object(BinBasedRoc, '__abstractmethods__', set())
        leading_bins = [1, 3, 5]
        samples = mocker.Mock()
        bins = [2, 4, 8, 20]
        roc = BinBasedRoc(bins_no=len(bins), leading_bins=leading_bins)
        mocker.patch.object(roc, '_calculate_bins', return_value=bins)

        roc.fit(samples)

        expected_bin_edges = np.array([0, 1, 3, 5, 8, np.PINF])
        np.testing.assert_almost_equal(expected_bin_edges, roc.bin_edges_)

    def test_not_allow_wrong_leading_bins(self, mocker):
        with pytest.raises(ValueError):
            mocker.patch.object(BinBasedRoc, '__abstractmethods__', set())
            leading_bins = [1, 5, 3]
            samples = mocker.Mock()
            roc = BinBasedRoc(bins_no=mocker.Mock(), leading_bins=leading_bins)

            roc.fit(samples)


class TestDistanceBasedROC:
    @pytest.mark.parametrize("distance_histogram, bins_no, expected_bins", [
        pytest.param(
            [56, 14, 15, 10, 3, 0, 1, 0, 0, 1],
            10,
            [0, 1, 2, 3, 4, 5, 7, np.PINF],
            id="shrink early bins, skip late bins"
        ),
        pytest.param(
            [3, 3, 3, 3, 3, 3, 3, 3],
            3,
            [0, 3, 6, np.PINF],
            id="divide non-equal"
        ),
        pytest.param(
            [2, 2, 2, 2, 2, 2, 2, 2],
            4,
            [0, 2, 4, 6, np.PINF],
            id="divide equal"
        ),
    ])
    def test_compressed_distance_bins(self, mocker, distance_histogram, bins_no, expected_bins):
        mocker.patch.object(DistanceBasedROC, '__abstractmethods__', set())
        samples = mocker.Mock()
        distance_histogram = np.array(distance_histogram)
        roc = DistanceBasedROC(bins_no=bins_no)
        mocker.patch.object(roc, '_sample_distances', return_value=distance_histogram)

        roc.fit(samples)

        expected_bins = np.array(expected_bins)
        np.testing.assert_almost_equal(roc.bin_edges_, expected_bins)

    def test_compress(self, mocker):
        mocker.patch.object(DistanceBasedROC, '__abstractmethods__', set())
        roc = DistanceBasedROC(bins_no=mocker.Mock())
        roc.bin_edges_ = np.array([0, 1, 2, 3, 4, 5, 7, 11, np.PINF])
        distances = np.array([0, 1, 3, 6, 9, 300, 6, 1, 1, 2])

        compressed = roc.compress(distances)

        expected_compressed = np.array([0, 1, 3, 6, 7, 8, 6, 1, 1, 2])
        np.testing.assert_almost_equal(compressed, expected_compressed)


class TestEqualProbabilityROC:
    def test_fit(self, mocker, mock_leaf_graph):
        dataset = create_mocked_samples(mocker, [mock_leaf_graph])
        roc = EqualProbabilityROC(bins_no=4)

        roc.fit(dataset)

        expected_bins = np.array([0, 3, 4, 5, np.PINF])
        np.testing.assert_almost_equal(roc.bin_edges_, expected_bins)

    def test_sample(self, mocker, mock_leaf_graph):
        mocker.patch('numpy.random.choice', return_value=np.array([0, 1]))
        dataset = create_mocked_samples(mocker, [mock_leaf_graph])
        roc = EqualProbabilityROC(bins_no=mocker.Mock(), vertex_sample_size=2)

        histogram = roc._sample_distances(dataset)

        expected_histogram = np.array([2, 4, 4, 6, 4, 5, 2, 1])
        np.testing.assert_almost_equal(histogram, expected_histogram)


class TestFlattenedEqualProbabilityROC:
    def test_fit(self, mocker, mock_leaf_graph):
        dataset = create_mocked_samples(mocker, [mock_leaf_graph])
        roc = FlattenedEqualProbabilityROC(bins_no=6)

        roc.fit(dataset)

        expected_bins = np.array([0, 2, 3, 4, 5, 6, np.PINF])
        np.testing.assert_almost_equal(roc.bin_edges_, expected_bins)


class TestFibonacciROC:
    def test_fit(self, mocker):
        dataset = mocker.Mock()
        roc = FibonacciROC(bins_no=6)

        roc.fit(dataset)

        expected_bins = np.array([0, 1, 2, 3, 5, 8, np.PINF])
        np.testing.assert_almost_equal(roc.bin_edges_, expected_bins)
