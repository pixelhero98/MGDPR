import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from functools import lru_cache
from torch.utils.data import Dataset
from tqdm import tqdm

class MyDataset(Dataset):
    """
    Dataset of graph snapshots built on sliding windows of stock‐time‐series.

    Each graph contains node features (X), adjacency matrices (A), and labels (Y).
    Normalize features by log1p then z-score per feature.
    """

    def __init__(
        self,
        root: str,
        desti: str,
        market: str,
        comlist: list[str],
        start: str,
        end: str,
        window: int,
        dataset_type: str,
        sparsification_threshold: float = 1.0
    ):
        super().__init__()
        self.root = root
        self.desti = desti
        self.market = market
        self.comlist = comlist
        self.start = start
        self.end = end
        self.window = window
        self.dataset_type = dataset_type
        self.sparsification_threshold = sparsification_threshold

        # Pre-load all dataframes
        self._dataframes = {comp: self._load_csv(comp) for comp in self.comlist}

        # Find dates and next common day
        self.dates, self.next_day = self._find_common_dates()
        if not self.next_day:
            raise ValueError(f"No common next day after {self.end} for all companies.")

        # Prepare output directory
        self._out_dir = os.path.join(
            self.desti,
            f"{self.market}_{self.dataset_type}_{self.start}_{self.end}_{self.window}"
        )
        os.makedirs(self._out_dir, exist_ok=True)

        # Generate graphs if missing or incomplete
        total_windows = len(self.dates) - self.window + 1
        existing = sorted([f for f in os.listdir(self._out_dir) if f.startswith('graph_')])
        if len(existing) < total_windows:
            self._create_graphs()

    def __len__(self) -> int:
        return len(self.dates) - self.window + 1

    def __getitem__(self, idx: int) -> dict:
        path = os.path.join(self._out_dir, f'graph_{idx}.pt')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing graph file for index {idx}")
        return torch.load(path)

    def _load_csv(self, comp: str) -> pd.DataFrame:
        """Load company CSV, index by date."""
        path = os.path.join(self.root, f"{self.market}_{comp}_30Y.csv")
        df = pd.read_csv(path, parse_dates=[0], index_col=0)
        df.index = df.index.normalize()
        return df

    def _find_common_dates(self) -> tuple[list[str], str | None]:
        """Find common trading dates between start/end and next common date after end."""
        start_dt = pd.to_datetime(self.start).date()
        end_dt = pd.to_datetime(self.end).date()
        max_dates = [df.index.max().date() for df in self._dataframes.values()]
        bound_dt = min(max_dates)

        valid_sets, after_sets = [], []
        for df in self._dataframes.values():
            dates = df.index.date
            valid = {d.isoformat() for d in dates if start_dt <= d <= end_dt}
            after = {d.isoformat() for d in dates if end_dt < d <= bound_dt}
            valid_sets.append(valid)
            after_sets.append(after)

        common = sorted(set.intersection(*valid_sets))
        after_common = set.intersection(*after_sets)
        next_day = min(after_common) if after_common else None
        return common, next_day

    def _compute_adjacency(self, features: np.ndarray) -> torch.Tensor:
        """
        Build adjacency matrix A where
        A_ij = log(max((E_i/E_j) * exp(H_i - H_j), threshold))
        """
        energy = np.sum(features**2, axis=1) + 1e-8
        entropy = np.apply_along_axis(
            lambda row: float(-np.sum(
                (np.unique(row, return_counts=True)[1] / row.size) *
                np.log((np.unique(row, return_counts=True)[1] / row.size) + 1e-12)
            )),
            1,
            features
        )

        E_ratio = energy[:, None] / energy[None, :]
        H_diff = np.exp(entropy[:, None] - entropy[None, :])
        A_np = E_ratio * H_diff

        A_clamped = np.maximum(A_np, self.sparsification_threshold)
        logA = np.log(A_clamped)
        return torch.from_numpy(logA).float()

    @lru_cache(maxsize=None)
    def _get_window(self, comp: str, dates_tuple: tuple[str, ...]) -> np.ndarray:
        """Return normalized windowed feature matrix: log1p + z-score of first 5 cols."""
        df = self._dataframes[comp].loc[dates_tuple]
        arr = df.iloc[:, :5].to_numpy()
        # Log transform
        log_arr = np.log1p(arr)
        # Z-score per column
        mean = log_arr.mean(axis=0, keepdims=True)
        std = log_arr.std(axis=0, keepdims=True) + 1e-8
        norm_arr = (log_arr - mean) / std
        return norm_arr

    def _create_graphs(self):
        """Generate and save graph tensors for each sliding window."""
        all_dates = self.dates + [self.next_day]

        for t in tqdm(range(len(self.dates) - self.window + 1)):
            out_path = os.path.join(self._out_dir, f'graph_{t}.pt')
            if os.path.exists(out_path):
                continue

            window_dates = tuple(all_dates[t:t + self.window + 1])
            X_list = [self._get_window(c, window_dates[:-1]) for c in self.comlist]
            X_np = np.stack(X_list, axis=1)
            X = torch.from_numpy(X_np.transpose(2,1,0)).float()

            last_prices = [
                self._dataframes[c].loc[window_dates[-2:], 'Close'].to_numpy()
                for c in self.comlist
            ]
            last_arr = np.stack(last_prices, axis=0)
            Y = torch.from_numpy((last_arr[:,1] - last_arr[:,0]) > 0).float()

            A = torch.stack([
                self._compute_adjacency(X[f].numpy()) for f in range(X.shape[0])
            ])

            torch.save({'X': X, 'A': A, 'Y': Y}, out_path)
