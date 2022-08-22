from abc import abstractmethod, ABC
import collections
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, List, Sequence, Tuple

import graph_tool as gt
import numpy as np
from tqdm import tqdm

import nero.tools.datasets as datasets
import nero.tools.logging as logging

logger = logging.get_configured_logger()


# noinspection PyArgumentList
class VertexOrigin(str, Enum):
    NODE = 'node'
    EDGE = 'edge'


@dataclass(frozen=True)
class RelevantProperty:
    name: str
    origin: VertexOrigin


# noinspection PyAttributeOutsideInit
@dataclass()
class BinGenerator(ABC):

    @abstractmethod
    def fit(self, samples: List[datasets.PersistedClassificationSample], lowest_index: int) -> None:
        self.lowest_index_: int = lowest_index

    @abstractmethod
    def digitise(self, graph: gt.Graph) -> List[np.ndarray]:
        pass

    @abstractmethod
    def signature(self) -> Tuple[str, Tuple]:
        pass

    @abstractmethod
    def id(self) -> str:
        pass

    @abstractmethod
    def not_use_tqdm(self) -> bool:
        pass

    @property
    def effective_bin_no(self) -> int:
        return self.__effective_bin_no

    @effective_bin_no.setter
    def effective_bin_no(self, effective_bin_no: int) -> None:
        self.__effective_bin_no = effective_bin_no

    def _assess_attributes_bounds(
            self,
            samples: List[datasets.PersistedClassificationSample],
            relevant_properties: List[RelevantProperty],
    ) -> Dict[RelevantProperty, Tuple[float, float]]:
        values = collections.defaultdict(list)
        for sample in tqdm(
                samples,
                desc=f"Fitting the {self.id()} to training data - assessing bounds",
                disable=self.not_use_tqdm(),
        ):
            graph = sample.materialise().graph
            for prop in relevant_properties:
                property_values, mask = self._unpack_property(graph, prop)
                values[(prop, 'min')].append(np.min(property_values, where=mask, initial=float('+inf')))
                values[(prop, 'max')].append(np.max(property_values, where=mask, initial=float('-inf')))
        return {prop: (min(values[(prop, 'min')]), max(values[(prop, 'max')])) for prop in relevant_properties}

    @staticmethod
    def _unpack_property(graph: gt.Graph, relevant_property: RelevantProperty) -> Tuple[np.ndarray, np.ndarray]:
        property_values = graph.vp[relevant_property.name].a
        mask = graph.vp['original_node'].a.astype('bool')
        mask = np.logical_not(mask) if relevant_property.origin == VertexOrigin.EDGE else mask
        return property_values, mask

    @staticmethod
    def wrap_with_infinities(bin_edges: Sequence[float]):
        return np.concatenate(([np.NINF], bin_edges, [np.PINF]))

    @staticmethod
    def digitise_property(
            graph: gt.Graph,
            relevant_property: RelevantProperty,
            bin_edges: np.ndarray,
            index_correction: int
    ) -> np.ndarray:
        digitized_property = np.digitize(graph.vertex_properties[relevant_property.name].a, bin_edges)
        member_mask = graph.vp['original_node'].a.astype('bool')
        member_mask = np.logical_not(member_mask) if relevant_property.origin == VertexOrigin.NODE else member_mask
        digitized_property[member_mask] = 0
        digitized_property += index_correction
        return digitized_property


# noinspection PyAttributeOutsideInit
@dataclass
class LabelBinGenerator(BinGenerator):
    relevant_property: RelevantProperty
    disable_tqdm: bool = False

    def fit(self, samples: List[datasets.PersistedClassificationSample], lowest_index: int) -> None:
        _, max_value = self._assess_attribute_bounds(samples)
        self.max_label_value_: int = int(max_value)
        self.effective_bin_no = self.max_label_value_ + 1
        BinGenerator.fit(self, samples, lowest_index)

    def signature(self) -> Tuple[str, Tuple]:
        return self.id(), tuple([self.max_label_value_])

    def digitise(self, graph: gt.Graph) -> List[np.ndarray]:
        digitised_property = graph.vertex_properties[self.relevant_property.name].a.astype('int')
        digitised_property[digitised_property > self.max_label_value_] = 0
        return [digitised_property]

    def id(self) -> str:
        return f"{self.relevant_property.name}__label"

    def not_use_tqdm(self) -> bool:
        return self.disable_tqdm

    def _assess_attribute_bounds(self, samples: List[datasets.PersistedClassificationSample]) -> Tuple[float, float]:
        return self._assess_attributes_bounds(samples, [self.relevant_property])[self.relevant_property]


# noinspection PyAttributeOutsideInit
@dataclass
class UnivariateBinGenerator(BinGenerator, ABC):
    relevant_property: RelevantProperty

    def fit(self, samples: List[datasets.PersistedClassificationSample], lowest_index: int) -> None:
        self.bin_edges_: np.ndarray = self._fit_bin_edges(samples)
        self.effective_bin_no = len(self.bin_edges_)
        BinGenerator.fit(self, samples, lowest_index)

    def signature(self) -> Tuple[str, Tuple]:
        return self.id(), tuple(self.bin_edges_)

    def digitise(self, graph: gt.Graph) -> List[np.ndarray]:
        return [self._digitise_single_property(graph)]

    def id(self) -> str:
        return f"{self.relevant_property.name}__default"

    def _digitise_single_property(self, graph: gt.Graph) -> np.ndarray:
        return self.digitise_property(graph, self.relevant_property, self.bin_edges_, self.lowest_index_)

    @abstractmethod
    def _fit_bin_edges(self, samples: List[datasets.PersistedClassificationSample]) -> np.ndarray:
        pass

    def _assess_attribute_bounds(self, samples: List[datasets.PersistedClassificationSample]) -> Tuple[float, float]:
        return self._assess_attributes_bounds(samples, [self.relevant_property])[self.relevant_property]


@dataclass
class EqualSizeBinGenerator(UnivariateBinGenerator):
    bins_no: int
    disable_tqdm: bool = False

    def id(self) -> str:
        return f"{self.relevant_property.name}__{self.bins_no}__equal_size"

    def not_use_tqdm(self) -> bool:
        return self.disable_tqdm

    def _fit_bin_edges(self, samples: List[datasets.PersistedClassificationSample]) -> np.ndarray:
        min_value, max_value = self._assess_attribute_bounds(samples)
        try:
            bin_edges = np.histogram_bin_edges(0, self.bins_no, (min_value, max_value))
        except ValueError:
            logger.error(f"Generator '{self.id()}' was unable to generate bins from {(min_value, max_value)}")
            bin_edges = [np.NINF] * (self.bins_no + 1)
        return self.wrap_with_infinities(bin_edges)


@dataclass
class InsightBinGenerator:
    calculate_bins: Callable[[float, float, int], Sequence[float]]
    bins_no: int
    bin_type: str

    def id(self) -> str:
        return f"{self.bin_type}_{self.bins_no}"

    def calculate(self, min_value: float, max_value: float) -> np.ndarray:
        return np.array(self.calculate_bins(min_value, max_value, self.bins_no))


@dataclass
class EqualProbabilityBinGenerator(UnivariateBinGenerator):
    bins_no: int
    insight_bins_generator: InsightBinGenerator
    disable_tqdm: bool = False

    def id(self) -> str:
        return f"{self.relevant_property.name}__{self.bins_no}__{self.insight_bins_generator.id()}__equal_probability"

    def not_use_tqdm(self) -> bool:
        return self.disable_tqdm

    def equal_probability_bins(self, insight_bins, insight_histogram) -> np.ndarray:
        histogram_sum = np.sum(insight_histogram)
        target_bin_weight = histogram_sum / self.bins_no
        bins = [insight_bins[0]]
        acquired_sum = 0
        current_insight_bin = 0
        current_bin_percentage = 1.0
        while len(bins) < self.bins_no:
            updated_sum = acquired_sum + insight_histogram[current_insight_bin] * current_bin_percentage
            if updated_sum > target_bin_weight:
                surplus_percentage = (updated_sum - target_bin_weight) / insight_histogram[current_insight_bin]
                edge_percentage = 1.0 - surplus_percentage
                insight_bin_width = insight_bins[current_insight_bin + 1] - insight_bins[current_insight_bin]
                edge_position = insight_bin_width * edge_percentage + insight_bins[current_insight_bin]
                bins.append(edge_position)
                acquired_sum = 0
                current_bin_percentage = surplus_percentage
            else:
                acquired_sum = updated_sum
                current_insight_bin += 1
                current_bin_percentage = 1.0
        bins.append(insight_bins[-1])
        return np.array(bins)

    def _fit_bin_edges(self, samples: List[datasets.PersistedClassificationSample]) -> np.ndarray:
        min_value, max_value = self._assess_attribute_bounds(samples)
        insight_bins = self.insight_bins_generator.calculate(min_value, max_value)
        insight_histogram = self._calculate_insight_histogram(samples, insight_bins)
        bin_edges = self.equal_probability_bins(insight_bins, insight_histogram)
        return self.wrap_with_infinities(bin_edges)

    def _calculate_insight_histogram(
            self,
            samples: List[datasets.PersistedClassificationSample],
            insight_bins: np.ndarray
    ) -> np.ndarray:
        insight_histogram = np.zeros(insight_bins.shape[0] - 1)
        for sample in tqdm(
                samples,
                desc=f"Fitting the {self.id()} to training data: sampling bins",
                disable=self.not_use_tqdm(),
        ):
            graph = sample.materialise().graph
            property_values, mask = self._unpack_property(graph, self.relevant_property)
            partial_histogram, _ = np.histogram(property_values, bins=insight_bins, weights=mask.astype('int'))
            insight_histogram += partial_histogram
        return insight_histogram
