from __future__ import annotations
from typing import List, Tuple

import numpy as np
import scipy.sparse as scp_sparse
import sklearn.random_projection as skl_random_projection

from nero.pipeline.transform.utils import ReplacingTransformer
import nero.tools.logging as logging

logger = logging.get_configured_logger()


# noinspection PyAttributeOutsideInit
# noinspection PyPep8Naming
class ToSmallestCommonShape(ReplacingTransformer):  # TODO: write tests for whole classes
    fit_parameters: List[str] = ['smallest_common_shape_']
    progress_message: str = "Resizing to smallest common shape"

    @staticmethod
    def smallest_common_shape(embeddings: List[np.ndarray]) -> Tuple[int, ...]:
        if len(embeddings) < 1:
            raise ValueError("Need at least one embedding to assess the common shape!")
        if not all(len(embeddings[0].shape) == len(embedding.shape) for embedding in embeddings[1:]):
            raise ValueError("Embeddings should have equal number of dimensions!")
        return tuple(max(sizes) for sizes in zip(*(embedding.shape for embedding in embeddings)))

    @staticmethod
    def resize_with_outer_zeros(embedding: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        padding = tuple((0, max(0, target_size - size)) for target_size, size in zip(target_shape, embedding.shape))
        target_slice = tuple(slice(None, target_size) for target_size in target_shape)
        return np.pad(embedding, padding)[target_slice]

    # noinspection PyUnusedLocal
    def fit(self, X: List[np.ndarray], y: None = None) -> ToSmallestCommonShape:
        self.smallest_common_shape_: Tuple[int, ...] = self.smallest_common_shape(X)
        return self

    def transform_element(self, x: np.ndarray) -> np.ndarray:
        return self.resize_with_outer_zeros(x, self.smallest_common_shape_)

    def transform_element_with_tracks(self, x: np.ndarray) -> np.ndarray:
        shape = self.smallest_common_shape_ + (x.shape[-1],)
        return self.resize_with_outer_zeros(x, shape)


class FlattenSamples(ReplacingTransformer):
    progress_message: str = "Flattening"

    def transform_element(self, x: np.ndarray) -> np.ndarray:
        return x.flatten()


class FlattenUpperTriangles(ReplacingTransformer):
    progress_message: str = "Flattening 3D samples"

    @staticmethod
    def flatten(embedding: np.ndarray) -> np.ndarray:
        triu_x, triu_y = np.triu_indices(embedding.shape[1])
        return embedding[:, triu_x, triu_y].flatten()

    @staticmethod
    def flatten_with_tracks(embedding: np.ndarray) -> np.ndarray:
        triu_x, triu_y = np.triu_indices(embedding.shape[1])
        embedding_upper_triangle = embedding[:, triu_x, triu_y, :]
        return np.reshape(embedding_upper_triangle, (-1, embedding_upper_triangle.shape[-1]))

    def transform_element(self, x: np.ndarray) -> np.ndarray:
        return self.flatten(x)


# noinspection PyAttributeOutsideInit
# noinspection PyPep8Naming
class RandomProjection(ReplacingTransformer):
    fit_parameters: List[str] = ['random_projection_']
    progress_message: str = "Casting a random projection"
    percentage_of_components_to_keep = 0.25

    # noinspection PyUnusedLocal
    def fit(self, X: List[np.ndarray], y: None = None) -> RandomProjection:
        sparse_proxy = scp_sparse.csr_matrix((len(X), len(X[0])))
        try:
            self.random_projection_ = skl_random_projection.SparseRandomProjection()  # TODO: more generic?
            self.random_projection_.fit(sparse_proxy)
        except ValueError:
            logger.warning(f"Failed to create an automatic random projection for data with shape{len(X), len(X[0])}")
            n_components = int(self.percentage_of_components_to_keep * len(X[0]))
            self.random_projection_ = skl_random_projection.SparseRandomProjection(n_components=n_components)
            self.random_projection_.fit(sparse_proxy)
        return self

    def transform_element(self, x: np.ndarray) -> np.ndarray:
        return self.random_projection_.transform([x])[0]
