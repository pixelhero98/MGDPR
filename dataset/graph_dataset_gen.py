"""Utilities to generate temporal graph datasets from raw OHLCV data."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

GraphSample = Dict[str, Tensor]

_EPS = 1e-8
_NUM_FEATURES = 5  # Open, High, Low, Close, Volume


def _ensure_sequence(name: str, values: Iterable[str]) -> List[str]:
    values = list(values)
    if not values:
        raise ValueError(f"{name} must contain at least one element.")
    return values


class GraphDataset(Dataset[GraphSample]):
    """Dataset of sliding-window graph snapshots for stock prediction."""

    def __init__(
        self,
        root: str | Path,
        destination: str | Path,
        market: str,
        companies: Sequence[str],
        start: str,
        end: str,
        window: int,
        dataset_type: str,
        sparsification_threshold: float = 1.0,
    ) -> None:
        super().__init__()

        if window <= 1:
            raise ValueError("`window` must be greater than 1 to compute labels.")
        if sparsification_threshold <= 0:
            raise ValueError("`sparsification_threshold` must be positive.")

        self.root = Path(root)
        self.destination = Path(destination)
        self.market = market
        self.companies = _ensure_sequence("companies", companies)
        self.start = start
        self.end = end
        self.window = window
        self.dataset_type = dataset_type
        self.sparsification_threshold = float(sparsification_threshold)
        self.feature_dim = self.window * _NUM_FEATURES

        self._dataframes: Dict[str, pd.DataFrame] = {
            company: self._load_company_frame(company) for company in self.companies
        }

        self.dates, self.next_day = self._find_common_dates()
        self._num_windows = len(self.dates) - self.window + 1
        if self._num_windows <= 0:
            raise ValueError(
                "Not enough common trading days to build even a single window. "
                "Reduce `window` or adjust the date range."
            )

        actual_start = self.dates[0].strftime("%Y-%m-%d")
        actual_end = self.dates[-1].strftime("%Y-%m-%d")
        self._output_dir = self.destination / (
            f"{self.market}_{self.dataset_type}_{actual_start}_{actual_end}_{self.window}"
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)

        if len(list(self._output_dir.glob("graph_*.pt"))) < self._num_windows:
            self._create_graphs()

    # ------------------------------------------------------------------
    # Dataset protocol
    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return self._num_windows

    def __getitem__(self, index: int) -> GraphSample:
        path = self._graph_path(index)
        if not path.exists():
            raise FileNotFoundError(f"Missing graph sample at index {index} ({path})")
        sample: GraphSample = torch.load(path, map_location="cpu")
        return sample

    # ------------------------------------------------------------------
    # Data loading helpers
    def _graph_path(self, index: int) -> Path:
        return self._output_dir / f"graph_{index}.pt"

    def _load_company_frame(self, company: str) -> pd.DataFrame:
        csv_path = self.root / f"{self.market}_{company}_30Y.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV file for {company!r}: {csv_path}")

        df = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
        if df.empty:
            raise ValueError(f"CSV file {csv_path} is empty.")
        if df.shape[1] < _NUM_FEATURES:
            raise ValueError(
                f"CSV file {csv_path} must contain at least {_NUM_FEATURES} columns "
                "for the OHLCV features."
            )

        df.index = df.index.normalize()
        return df

    def _find_common_dates(self) -> Tuple[List[pd.Timestamp], pd.Timestamp]:
        start_ts = pd.Timestamp(self.start)
        requested_end = pd.Timestamp(self.end)

        max_dates = [frame.index.max() for frame in self._dataframes.values()]
        min_dates = [frame.index.min() for frame in self._dataframes.values()]
        latest_common_end = min(max_dates)
        earliest_common_start = max(min_dates)

        if earliest_common_start > requested_end:
            raise ValueError(
                "The requested end date precedes the earliest shared trading day "
                f"({earliest_common_start.date()})."
            )

        effective_end = min(requested_end, latest_common_end)
        if effective_end < start_ts:
            raise ValueError(
                "The requested window does not overlap with the available data. "
                f"Earliest shared date is {earliest_common_start.date()}"
                f" while start date is {start_ts.date()}."
            )

        valid_sets: List[set[pd.Timestamp]] = []
        future_sets: List[set[pd.Timestamp]] = []
        for frame in self._dataframes.values():
            index = frame.index
            valid_mask = (index >= start_ts) & (index <= effective_end)
            future_mask = index > effective_end
            valid_sets.append(set(index[valid_mask]))
            future_sets.append(set(index[future_mask]))

        if not valid_sets:
            raise RuntimeError("No dataframes were loaded for the requested companies.")

        common = sorted(set.intersection(*valid_sets))
        if not common:
            raise ValueError("No common trading days found in the requested range.")

        next_candidates = set.intersection(*future_sets)
        if not next_candidates:
            raise ValueError(
                "No common trading day is available after "
                f"{effective_end.date()}. Ensure the requested end date leaves "
                "at least one shared trading session for label computation."
            )

        next_day = min(next_candidates)
        return [pd.Timestamp(ts) for ts in common], pd.Timestamp(next_day)

    # ------------------------------------------------------------------
    # Feature preparation helpers
    @staticmethod
    def _shannon_entropy(matrix: np.ndarray) -> np.ndarray:
        entropies = np.empty(matrix.shape[0], dtype=np.float64)
        for idx, row in enumerate(matrix):
            values, counts = np.unique(row, return_counts=True)
            probabilities = counts.astype(np.float64) / counts.sum()
            entropies[idx] = -np.sum(probabilities * np.log(probabilities + _EPS))
        return entropies

    def _compute_adjacency(self, node_features: np.ndarray) -> Tensor:
        node_features = node_features.astype(np.float64, copy=False)
        energy = np.sum(np.square(node_features), axis=1) + _EPS
        entropy = self._shannon_entropy(node_features)

        energy_ratio = energy[:, None] / energy[None, :]
        entropy_ratio = np.exp(entropy[:, None] - entropy[None, :])
        combined = energy_ratio * entropy_ratio
        combined = 0.5 * (combined + combined.T)
        combined = np.clip(combined, self.sparsification_threshold, None)

        return torch.from_numpy(np.log(combined)).float()

    @lru_cache(maxsize=None)
    def _get_window(self, company: str, dates: Tuple[pd.Timestamp, ...]) -> np.ndarray:
        frame = self._dataframes[company]
        window_df = frame.loc[list(dates)]
        features = window_df.iloc[:, :_NUM_FEATURES].to_numpy(dtype=np.float32, copy=True)
        return np.log1p(features)

    # ------------------------------------------------------------------
    # Graph generation
    def _create_graphs(self) -> None:
        all_dates = self.dates + [self.next_day]

        for index in tqdm(
            range(self._num_windows),
            desc="Building graph dataset",
            unit="win",
        ):
            path = self._graph_path(index)
            if path.exists():
                continue

            window_dates = tuple(all_dates[index : index + self.window + 1])
            feature_windows = np.stack(
                [self._get_window(company, window_dates[:-1]) for company in self.companies],
                axis=0,
            )  # (num_companies, window, num_features)

            features = torch.from_numpy(
                feature_windows.reshape(len(self.companies), -1)
            ).float()

            feature_matrices = np.transpose(feature_windows, (2, 0, 1))
            adjacency = torch.stack(
                [
                    self._compute_adjacency(feature_matrices[feat_idx])
                    for feat_idx in range(feature_matrices.shape[0])
                ]
            )

            price_changes = np.stack(
                [
                    self._dataframes[company]
                    .loc[list(window_dates[-2:]), "Close"]
                    .to_numpy(dtype=np.float32, copy=True)
                    for company in self.companies
                ],
                axis=0,
            )
            labels = torch.from_numpy(
                (price_changes[:, 1] > price_changes[:, 0]).astype(np.int64)
            )

            torch.save({"X": features, "A": adjacency, "Y": labels}, path)


MyDataset = GraphDataset

__all__ = ["GraphDataset", "MyDataset"]
