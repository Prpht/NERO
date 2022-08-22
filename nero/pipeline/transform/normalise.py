import numpy as np

from nero.pipeline.transform.utils import ReplacingTransformer


class PercentOfMaxPerRelationOrder(ReplacingTransformer):
    progress_message: str = "Calculating percent of row maximum normalisation"

    def __init__(self, disable_tqdm: bool = False):
        super().__init__(disable_tqdm)

    @staticmethod
    def normalise(embedding: np.ndarray) -> np.ndarray:
        np.seterr(divide='ignore', invalid='ignore')
        dims = len(embedding.shape)
        axis = tuple(ax for ax in range(1, dims))
        indexes = tuple(slice(None) if ax == 0 else np.newaxis for ax in range(dims))
        return np.nan_to_num(embedding / np.amax(embedding, axis=axis)[indexes])

    def transform_element(self, x: np.ndarray) -> np.ndarray:
        return self.normalise(x)


class PercentOfRow(ReplacingTransformer):
    progress_message: str = "Calculating percent of row normalisation"

    def __init__(self, disable_tqdm: bool = False):
        super().__init__(disable_tqdm)

    @staticmethod
    def normalise(embedding: np.ndarray) -> np.ndarray:
        np.seterr(divide='ignore', invalid='ignore')
        return np.nan_to_num(embedding / np.sum(embedding, axis=2)[:, :, np.newaxis])

    def transform_element(self, x: np.ndarray) -> np.ndarray:
        return self.normalise(x)


class DoNothing(ReplacingTransformer):
    progress_message: str = "Calculating ain't no normalisation"

    def __init__(self, disable_tqdm: bool = False):
        super().__init__(disable_tqdm)

    def transform_element(self, x: np.ndarray) -> np.ndarray:
        return x
