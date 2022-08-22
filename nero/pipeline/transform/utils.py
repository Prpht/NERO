from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import sklearn.base as skl_base
import sklearn.utils.validation as skl_validation
from tqdm import tqdm


# noinspection PyPep8Naming
class ReplacingTransformer(skl_base.TransformerMixin, skl_base.BaseEstimator, ABC):  # TODO: mock based tests?
    fit_parameters: List[str] = []
    progress_message: str = "Doing an unknown kind of transformation"

    def __init__(self, disable_tqdm: bool = False):
        self.disable_tqdm = disable_tqdm

    @abstractmethod
    def transform_element(self, x: np.ndarray) -> np.ndarray:
        pass

    # noinspection PyUnusedLocal
    def fit(self, X: List[np.ndarray], y: None = None) -> ReplacingTransformer:
        return self

    def transform(self, X: List[np.ndarray]) -> List[np.ndarray]:
        for parameter in self.fit_parameters:
            skl_validation.check_is_fitted(self, parameter)
        for i in tqdm(range(len(X)), desc=self.progress_message, disable=self.disable_tqdm):
            X[i] = self.transform_element(X[i])
        return X
