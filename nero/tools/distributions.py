from dataclasses import dataclass
import itertools
import math
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def aggregate_histograms(
        df: pd.DataFrame,
        attribute: str,
        bins: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    histograms = np.array([np.histogram(d, bins=bins, density=True)[0] for d in df[attribute]])
    histograms_mean = np.mean(histograms, axis=0)
    histograms_std = np.std(histograms, axis=0)
    upper_limit = histograms_mean + histograms_std
    lower_limit = np.array([value if value > 0.0 else 0.0 for value in (histograms_mean - histograms_std)])
    return histograms_mean, lower_limit, upper_limit


def attribute_global_range(df: pd.DataFrame, attribute: str) -> Tuple[float, float]:
    min_val = np.min([np.min(val) for val in df[attribute]])
    max_val = np.max([np.max(val) for val in df[attribute]])
    return min_val, max_val


@dataclass
class DistributionApproximation:
    attribute: str
    histograms_mean: np.array
    lower_limit: np.array
    upper_limit: np.array
    bins_centers: np.array
    bins_width: np.array


def calculate_distribution(
        df: pd.DataFrame,
        attribute: str,
        calculate_bins: Callable[..., Sequence[float]],
        bins_no: Optional[int] = None,
) -> DistributionApproximation:
    min_val, max_val = attribute_global_range(df, attribute)
    bins = calculate_bins(min_val, max_val, bins_no) if bins_no else calculate_bins(min_val, max_val)
    histograms_mean, lower_limit, upper_limit = aggregate_histograms(df, attribute, bins)
    bins_centers = np.array([np.mean(margins) for margins in pairwise(bins)])
    bins_width = np.array([math.fabs(right - left) for left, right in pairwise(bins)])
    return DistributionApproximation(attribute, histograms_mean, lower_limit, upper_limit, bins_centers, bins_width)


def log_bins(min_val: float, max_val: float, bins_no: int) -> Sequence[float]:
    return np.logspace(np.log10(min_val), np.log10(max_val), bins_no + 1)


def equal_size_bins(min_val: float, max_val: float, bins_no: int) -> Sequence[float]:
    return np.histogram_bin_edges([], bins_no, (min_val, max_val))


def discrete_bins(min_val: float, max_val: float) -> Sequence[float]:
    return [x - 0.5 for x in range(int(min_val), int(max_val) + 2)]


def pairwise(iterable: Iterable) -> Iterable:
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)
