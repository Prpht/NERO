from __future__ import annotations

from dataclasses import dataclass
import functools
import itertools
import pathlib
import pickle
from typing import Iterable, List, Tuple, Optional

import graph_tool as gt
import graph_tool.topology as gt_topology
from joblib import delayed, Memory, Parallel
import numpy as np
import sklearn.base as skl_base
from tqdm import tqdm

import nero.embedding.digitisers as trait_digitisers
import nero.embedding.compressors as compressors
from nero.embedding.utils import add_embeddings_3d, add_embeddings_4d
import nero.tools.datasets as datasets
import nero.tools.logging as logging

logger = logging.get_configured_logger()


# noinspection PyAttributeOutsideInit
# noinspection PyPep8Naming
@dataclass
class NeroEmbedder(skl_base.TransformerMixin, skl_base.BaseEstimator):
    bin_generators: List[trait_digitisers.BinGenerator]
    relation_order_compressor: compressors.RelationOrderCompressor = compressors.NoCompressionROC()
    jobs_no: int = 8
    node_batch_size: int = 50
    memory: Optional[Memory] = None
    disable_tqdm: bool = False

    def __hash__(self) -> int:
        return hash(self._id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NeroEmbedder) and self.id == other.id

    @property
    def id(self) -> Tuple[Tuple[Tuple[str, Tuple]], str]:
        if not hasattr(self, '_id'):
            bin_generator_signatures = [g.signature() for g in self.bin_generators]
            embedder_id = tuple(sorted(bin_generator_signatures)), self.relation_order_compressor.id()
            self._id = embedder_id
        return self._id

    # noinspection PyUnusedLocal
    def fit(self, X: List[datasets.PersistedClassificationSample], y: None = None) -> NeroEmbedder:
        lowest_index = 0
        for generator in self.bin_generators:
            generator.fit(X, lowest_index)
            lowest_index += generator.effective_bin_no
        self.cumulative_bin_no_ = sum(generator.effective_bin_no for generator in self.bin_generators)
        self.relation_order_compressor.fit(X)
        if self.memory:
            self.transform_graph = self.memory.cache(self.transform_graph)
        return self

    def transform(self, X: List[datasets.PersistedClassificationSample]) -> List[np.ndarray]:
        result = []
        for x in tqdm(X, desc="Creating attribute histogram embeddings", disable=self.disable_tqdm):
            graph = x.materialise().graph
            embedding = self.transform_graph(graph)
            result.append(embedding)
        return result

    def transform_graph(
            self,
            graph: gt.Graph,
            track_sources: bool = False,
            persist_partial_results_to: Optional[pathlib.Path] = None,
    ) -> np.ndarray:
        vertex_batches = self.prepare_vertex_batches(graph, self.node_batch_size)
        digitised_properties = self._digitise_properties(graph)
        if self.jobs_no == 1:
            partial_results = [
                self._embed(graph, digitised_properties, batch, track_sources) for batch in vertex_batches
            ]
        else:
            partial_results = Parallel(n_jobs=self.jobs_no)(
                delayed(self._embed)(graph, digitised_properties, batch, track_sources) for batch in vertex_batches
            )
        if persist_partial_results_to is not None:
            with open(persist_partial_results_to, 'wb') as file:
                pickle.dump(partial_results, file, pickle.HIGHEST_PROTOCOL)
        return self._reduce_partial_results(graph, partial_results, track_sources)

    def _reduce_partial_results(
            self,
            graph: gt.Graph,
            partial_results: List[np.array],
            track_sources: bool = False
    ) -> np.ndarray:
        reduce_function = add_embeddings_3d if not track_sources else add_embeddings_4d
        embedding_block = self._embedding_block() if not track_sources else self._embedding_block(graph.num_vertices())
        return functools.reduce(reduce_function, partial_results, embedding_block)

    @staticmethod
    def prepare_vertex_batches(graph: gt.Graph, node_batch_size: int) -> Iterable[np.ndarray]:
        vertices = graph.get_vertices()
        np.random.shuffle(vertices)
        return (vertices[i:i + node_batch_size] for i in range(0, len(vertices), node_batch_size))

    def _digitise_properties(self, graph: gt.Graph) -> np.ndarray:
        digitised_properties = [generator.digitise(graph) for generator in self.bin_generators]
        digitised_properties = list(itertools.chain.from_iterable(digitised_properties))
        return np.array(digitised_properties)

    def _embedding_block(self, track_sources_size: Optional[int] = None) -> np.ndarray:
        if track_sources_size is None:
            return np.zeros((1, self.cumulative_bin_no_, self.cumulative_bin_no_), dtype='int')
        else:
            return np.zeros((1, self.cumulative_bin_no_, self.cumulative_bin_no_, track_sources_size), dtype='int')

    def _slice(self, max_distance: int, track_sources_size: Optional[int] = None) -> np.ndarray:
        if track_sources_size is None:
            return np.zeros((max_distance, self.cumulative_bin_no_), dtype='int')
        else:
            return np.zeros((max_distance, self.cumulative_bin_no_, track_sources_size), dtype='int')

    def _embed(
            self,
            graph: gt.Graph,
            digitised_properties: np.ndarray,
            nodes: np.ndarray,
            track_sources: bool = False,
    ) -> np.array:
        if not track_sources:
            embedding = self._embedding_block()
        else:
            embedding = self._embedding_block(graph.num_vertices())
        for node in nodes:
            embedding_slice = self._embed_from_source(graph, digitised_properties, node, track_sources)
            embedding = self.update_embedding(embedding, embedding_slice, digitised_properties, node, track_sources)
        return embedding

    def _embed_from_source(
            self,
            graph: gt.Graph,
            digitised_properties: np.ndarray,
            node_index: int,
            track_sources: bool = False,
    ) -> np.ndarray:
        distances = gt_topology.shortest_distance(graph, node_index).a
        distances[distances == 2147483647] = 0
        distances = self.relation_order_compressor.compress(distances)
        max_distance = np.max(distances) + 1
        if not track_sources:
            embedding_slice = self._slice(max_distance)
            self.fill_slice(embedding_slice, distances, digitised_properties)
        else:
            embedding_slice = self._slice(max_distance, graph.num_vertices())
            self.fill_slice(embedding_slice, distances, digitised_properties, node_index)
        return embedding_slice

    @staticmethod
    def fill_slice(
            embedding_slice: np.ndarray,
            distances: np.ndarray,
            digitised_properties: np.ndarray,
            track_sources_node_index: Optional[int] = None,
    ) -> None:
        ravelled_properties = digitised_properties.ravel()
        tiles_no = ravelled_properties.shape[0] // distances.shape[0]
        tiled_distances = np.tile(distances, tiles_no)
        if track_sources_node_index is None:
            np.add.at(embedding_slice, (tiled_distances, ravelled_properties), 1)
        else:
            neighbour_indexes = np.arange(distances.shape[0])
            tiled_neighbour_indexes = np.tile(neighbour_indexes, tiles_no)
            np.add.at(embedding_slice, (tiled_distances, ravelled_properties, tiled_neighbour_indexes), 1)
            tiled_node_index = np.repeat(track_sources_node_index, tiled_distances.shape[0])
            np.add.at(embedding_slice, (tiled_distances, ravelled_properties, tiled_node_index), 1)

    @staticmethod
    def update_embedding(
            embedding: np.ndarray,
            embedding_slice: np.ndarray,
            digitised_properties: np.ndarray,
            node_index: int,
            track_sources: bool = False,
    ) -> np.ndarray:
        if not track_sources:
            max_order = embedding_slice.shape[0]
            current_max_order, source_bin_no, target_bin_no = embedding.shape
            if max_order > current_max_order:
                new_embedding = np.zeros((max_order, source_bin_no, target_bin_no), dtype='int')
                new_embedding[:embedding.shape[0], :, :] = embedding
                embedding = new_embedding

            node_bins = digitised_properties[:, node_index]
            embedding[:embedding_slice.shape[0], node_bins, :] += embedding_slice[:, np.newaxis, :]
        else:
            max_order = embedding_slice.shape[0]
            current_max_order, source_bin_no, target_bin_no, node_indexes = embedding.shape
            if max_order > current_max_order:
                new_embedding = np.zeros((max_order, source_bin_no, target_bin_no, node_indexes), dtype='int')
                new_embedding[:embedding.shape[0], :, :, :] = embedding
                embedding = new_embedding

            node_bins = digitised_properties[:, node_index]
            embedding[:embedding_slice.shape[0], node_bins, :, :] += embedding_slice[:, np.newaxis, :, :]
        return embedding
