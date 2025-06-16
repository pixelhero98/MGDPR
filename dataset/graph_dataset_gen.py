import os
import csv
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
        dataset_type: str
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

        # Pre-load all dataframes
        self._dataframes = {
            comp: self._load_csv(comp)
            for comp in self.comlist
        }

        # Find dates and next common day
        self.dates, self.next_day = self._find_common_dates()

        # Ensure output directory exists
        self._out_dir = os.path.join(
            self.desti,
            f"{self.market}_{self.dataset_type}_{self.start}_{self.end}_{self.window}"
        )
        os.makedirs(self._out_dir, exist_ok=True)

        # Generate graphs if missing
        expected = len(self.dates) - self.window + 1
        exists = len([name for name in os.listdir(self._out_dir) if name.startswith('graph_')])
        if exists < expected:
            self._create_graphs()

    def __len__(self) -> int:
        return len(self.dates) - self.window + 1

    def __getitem__(self, idx: int) -> dict:
        path = os.path.join(self._out_dir, f'graph_{idx}.pt')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing graph file for index {idx}")
        return torch.load(path)

    def _load_csv(self, comp: str) -> pd.DataFrame:
        """Load company CSV, index by date string."""
        path = os.path.join(self.root, f"{self.market}_{comp}_30Y.csv")
        df = pd.read_csv(path, parse_dates=[0], index_col=0)
        df.index = df.index.normalize()
        return df

    def _find_common_dates(self) -> tuple[list[str], str | None]:
        """Find common trading dates in [start, end] and the next day after end."""
        start_dt = datetime.fromisoformat(self.start).date()
        end_dt = datetime.fromisoformat(self.end).date()
        today_str = datetime.today().strftime("%Y-%m-%d")
        after_dt = datetime.today().date()

        valid_sets, after_sets = [], []
        for comp, df in self._dataframes.items():
            dates = df.index.date
            valid = {d.isoformat() for d in dates if start_dt <= d <= end_dt}
            after = {d.isoformat() for d in dates if end_dt < d <= after_dt}
            valid_sets.append(valid)
            after_sets.append(after)

        common = sorted(set.intersection(*valid_sets))
        after_common = set.intersection(*after_sets)
        next_day = min(after_common) if after_common else None
        return common, next_day

    @staticmethod
    def _signal_energy(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    @staticmethod
    def _information_entropy(x: np.ndarray) -> float:
        unique, counts = np.unique(x, return_counts=True)
        p = counts / counts.sum()
        return float(-(p * np.log(p + 1e-12)).sum())

    def _adjacency(self, features: np.ndarray) -> torch.Tensor:
        """
        Build adjacency matrix A where
          A_ij = log( max( (E_i/E_j) * exp(H_i - H_j), 1 ) )
        using vectorized numpy operations.
        """
        # features: shape (num_nodes, num_features)
        energy = np.array([self._signal_energy(row) for row in features]) + 1e-8
        entropy = np.array([self._information_entropy(row) for row in features])

        E_ratio = energy[:, None] / energy[None, :]
        H_diff = np.exp(entropy[:, None] - entropy[None, :])
        A_np = E_ratio * H_diff
        A_np[A_np < 1] = 1.0

        return torch.from_numpy(np.log(A_np)).float()

    @lru_cache(maxsize=None)
    def _get_window(self, comp: str, dates_tuple: tuple[str, ...]) -> np.ndarray:
        """Return windowed feature matrix for one company."""
        df = self._dataframes[comp].loc[dates_tuple]
        # Select first 5 columns
        arr = df.iloc[:, :5].to_numpy()
        return np.log1p(arr)

    def _create_graphs(self):
        """Generate and save graph tensors for each sliding window."""
        # include next_day at end to shape labels
        all_dates = self.dates + ([self.next_day] if self.next_day else [])
        for t in tqdm(range(len(self.dates) - self.window + 1)):
            idx = t
            out_path = os.path.join(self._out_dir, f'graph_{idx}.pt')
            if os.path.exists(out_path):
                continue

            window_dates = tuple(all_dates[t : t + self.window + 1])
            # Build node features: shape (num_features, num_nodes, window)
            X_list = []
            for comp in self.comlist:
                mat = self._get_window(comp, window_dates[:-1])  # last date reserved for labels
                X_list.append(mat)
            X_np = np.stack(X_list, axis=1)  # shape (window, num_nodes, features)
            X = torch.from_numpy(X_np.transpose(2,1,0)).float()

            # Labels Y: price change from penultimate to last day
            last0 = np.stack([self._get_window(c, window_dates[-2:]) for c in self.comlist], axis=0)
            Y = torch.tensor((last0[:,1,3] - last0[:,0,3]) > 0, dtype=torch.float32)

            # Adjacency per feature slice
            A = torch.stack([self._adjacency(X[f].numpy()) for f in range(X.shape[0])])

            torch.save({ 'X': X, 'A': A, 'Y': Y }, out_path)
