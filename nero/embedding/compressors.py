from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import itertools
from typing import Iterator, List, Optional

import graph_tool.topology as gt_topology
import numpy as np
from tqdm import tqdm

import nero.tools.datasets as datasets
import nero.tools.logging as logging

logger = logging.get_configured_logger()


class RelationOrderCompressor(ABC):
    @abstractmethod
    def id(self) -> str:
        pass

    @abstractmethod
    def compress(self, distances: np.ndarray) -> np.ndarray:
        pass

    def fit(self, samples: List[datasets.PersistedClassificationSample]) -> RelationOrderCompressor:
        return self


class NoCompressionROC(RelationOrderCompressor):
    def id(self) -> str:
        return "no_relation_oder_compression"

    def compress(self, distances: np.ndarray) -> np.ndarray:
        return distances


# noinspection PyAttributeOutsideInit
@dataclass
class BinBasedRoc(RelationOrderCompressor, ABC):
    bins_no: int
    leading_bins: List[int] = None

    def fit(self, samples: List[datasets.PersistedClassificationSample]) -> RelationOrderCompressor:
        if self.leading_bins:
            if not (np.all(np.diff(self.leading_bins) > 0)):
                error_message = "leading bins not ordered"
                logger.error(error_message)
                raise ValueError(error_message)
        else:
            self.leading_bins = []
        self.bin_edges_ = self._wrap_bins(self._calculate_bins(samples))
        return self

    def compress(self, distances: np.ndarray) -> np.ndarray:
        return np.digitize(distances, self.bin_edges_, right=True)

    def _wrap_bins(self, bins: List[int]) -> np.ndarray:
        if self.leading_bins:
            i = 0
            for i, value in enumerate(bins):
                if value > self.leading_bins[-1]:
                    break
            bins = bins[i:]
        helper_result = [0] + self.leading_bins + bins
        helper_result[-1] = np.PINF
        return np.array(helper_result)

    @abstractmethod
    def _calculate_bins(self, samples: List[datasets.PersistedClassificationSample]) -> List[int]:
        pass


@dataclass
class DistanceBasedROC(BinBasedRoc):
    vertex_sample_size: Optional[int] = None
    disable_tqdm: bool = False

    def _compressed_distance_bins(self, distance_histogram: np.ndarray) -> List[int]:
        def _compressed_histogram_helper(i: int, remaining_bins: int) -> List[int]:
            if i == distance_histogram.size or not remaining_bins:
                return []
            else:
                bin_sum = np.sum(distance_histogram[i:]) / remaining_bins
                covered_sum = 0
                while covered_sum < bin_sum and i < distance_histogram.size:
                    covered_sum += distance_histogram[i]
                    i += 1
                return [i] + _compressed_histogram_helper(i, remaining_bins - 1)

        return _compressed_histogram_helper(0, self.bins_no)

    def _sample_distances(self, samples: List[datasets.PersistedClassificationSample]) -> np.ndarray:
        distance_histogram = np.zeros(1, dtype=np.int64)
        for sample in tqdm(samples, desc=f"Fitting the compressor to training data", disable=self.disable_tqdm):
            graph = sample.materialise().graph

            vertices = graph.get_vertices()
            if self.vertex_sample_size:
                sample_size = min(self.vertex_sample_size, vertices.shape[0])
                vertices = np.random.choice(vertices, sample_size, replace=False)

            for vertex in vertices:
                distances = gt_topology.shortest_distance(graph, vertex).a
                distances[distances == 2147483647] = 0
                partial_histogram = self._partial_histogram(distances)

                if distance_histogram.shape[0] < partial_histogram.shape[0]:
                    distance_histogram, partial_histogram = partial_histogram, distance_histogram
                distance_histogram[:partial_histogram.shape[0]] += partial_histogram
        return distance_histogram

    def _calculate_bins(self, samples: List[datasets.PersistedClassificationSample]) -> List[int]:
        distance_histogram = self._sample_distances(samples)
        return self._compressed_distance_bins(distance_histogram)

    @abstractmethod
    def _partial_histogram(self, distances: np.ndarray) -> np.ndarray:
        pass


class EqualProbabilityROC(DistanceBasedROC):
    def id(self) -> str:
        return f"{tuple(self.bin_edges_)}__relation_oder_probability_compression"

    def _partial_histogram(self, distances: np.ndarray) -> np.ndarray:
        return np.bincount(distances)


class FlattenedEqualProbabilityROC(DistanceBasedROC):
    def id(self) -> str:
        return f"{tuple(self.bin_edges_)}__flattened_relation_oder_probability_compression"

    def _partial_histogram(self, distances: np.ndarray) -> np.ndarray:
        return np.ones(np.max(distances) + 1)


# noinspection PyAttributeOutsideInit
class FibonacciROC(BinBasedRoc):
    def id(self) -> str:
        return f"{self.bins_no}__fibonacci_compression"

    def _calculate_bins(self, samples: List[datasets.PersistedClassificationSample]) -> List[int]:
        return list(itertools.islice(self._fibonaccis(), self.bins_no))

    @staticmethod
    def _fibonaccis() -> Iterator[int]:
        a = 1
        b = 2
        while True:
            yield a
            a, b = b, a + b


@dataclass
class VGGInspiredROC(BinBasedRoc):
    a: float = 1.07431
    b: float = 1.25690766
    c: float = -3.17819495

    def id(self) -> str:
        return f"{self.bins_no}__vgg_ins_compression"

    def _calculate_bins(self, samples: List[datasets.PersistedClassificationSample]) -> List[int]:
        return [1] + [int(self.linearised_exponent(x + 3, self.a, self.b, self.c)) for x in range(self.bins_no - 1)]

    @staticmethod
    def linearised_exponent(x: int, a: float, b: float, c: float) -> float:
        return (a ** x) * x + (b ** x) + c
