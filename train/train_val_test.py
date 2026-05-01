"""Training, validation and testing entrypoint for MGDPR."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.market_data import (  # noqa: E402
    default_download_range,
    default_ticker_path,
    load_ticker_file,
    prepare_market_data,
)
from dataset.graph_dataset_gen import GraphDataset, GraphSample  # noqa: E402
from model.multi_gdn import MGDPR  # noqa: E402


# ─── Reproducibility ────────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _iterate_dataset(dataset: GraphDataset, shuffle: bool = False) -> Iterator[GraphSample]:
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)
    for idx in indices:
        yield dataset[idx]


def _move_sample(sample: GraphSample, device: torch.device) -> GraphSample:
    return {key: tensor.to(device) for key, tensor in sample.items()}


# ─── Training / Validation / Test Loops ────────────────────────────────────────
def train_one_epoch(
    model: MGDPR,
    dataset: GraphDataset,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for sample in _iterate_dataset(dataset, shuffle=True):
        batch = _move_sample(sample, device)
        features = batch["X"]  # (N, F)
        adjacencies = batch["A"]  # (R, N, N)
        labels = batch["Y"].long()  # (N,)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features, adjacencies)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    acc = total_correct / max(total_samples, 1)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model: MGDPR, dataset: GraphDataset, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for sample in _iterate_dataset(dataset):
        batch = _move_sample(sample, device)
        logits = model(batch["X"], batch["A"])
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(batch["Y"].cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    acc = float((y_pred == y_true).mean())
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    return acc, f1, mcc


def _load_tickers(paths: Sequence[Path]) -> Tuple[List[str], List[str], List[str], List[str]]:
    if len(paths) != 4:
        raise ValueError("Expected four CSV paths: NASDAQ, NYSE, NYSE-missing, SSE.")

    tickers = [[] for _ in range(4)]
    for csv_path, target in zip(paths, tickers):
        with csv_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                ticker = line.strip().split(",")[0]
                if ticker:
                    target.append(ticker)

    nasdaq, nyse, nyse_missing, sse = tickers
    nyse = [ticker for ticker in nyse if ticker not in nyse_missing]
    return nasdaq, nyse, nyse_missing, sse


def _missing_raw_files(raw_dir: str | Path, market: str, tickers: Sequence[str]) -> List[Path]:
    root = Path(raw_dir)
    return [root / f"{market}_{ticker}_30Y.csv" for ticker in tickers if not (root / f"{market}_{ticker}_30Y.csv").exists()]


def _missing_data_message(args: argparse.Namespace, missing: Sequence[Path]) -> str:
    preview = ", ".join(path.name for path in missing[:5])
    if len(missing) > 5:
        preview += f", ... ({len(missing)} missing total)"
    command = (
        "python -m data.prepare_market_data "
        f"--market {args.market} --raw_dir {args.raw_dir}"
    )
    if args.ticker_path:
        command += f" --ticker_path {args.ticker_path}"
    return (
        f"Missing raw OHLCV files: {preview}. "
        f"Prepare them with `{command}` or rerun training with `--fetch_data`."
    )


def _resolve_sp500_tickers(args: argparse.Namespace) -> List[str]:
    if args.fetch_data:
        data_start, data_end = default_download_range(args.train_dates[0], args.test_dates[1])
        result = prepare_market_data(
            market=args.market,
            raw_dir=args.raw_dir,
            ticker_path=args.ticker_path,
            output_ticker_path=args.ticker_path or default_ticker_path(args.raw_dir, args.market),
            start=args.data_start or data_start,
            end=args.data_end or data_end,
            force_download=args.force_download,
            allow_partial=args.allow_partial,
        )
        if result.failed:
            failed = ", ".join(ticker for ticker, _ in result.failed[:10])
            print(f"Skipping failed downloads: {failed}")
        failed_tickers = {ticker for ticker, _ in result.failed}
        return [ticker for ticker in result.tickers if ticker not in failed_tickers]

    if args.ticker_path is None:
        command = (
            "python -m data.prepare_market_data "
            f"--market SP500 --raw_dir {args.raw_dir}"
        )
        raise ValueError(
            "--ticker_path is optional for SP500 only when --fetch_data is set. "
            f"Provide --ticker_path, run `{command}`, or rerun training with --fetch_data."
        )
    return load_ticker_file(args.ticker_path)


# ─── Main ──────────────────────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    if args.early_stop_patience is not None and args.early_stop_patience <= 0:
        raise ValueError("--early_stop_patience must be positive when supplied.")
    if args.early_stop_min_delta < 0:
        raise ValueError("--early_stop_min_delta must be non-negative.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_range = args.train_dates
    val_range = args.val_dates
    test_range = args.test_dates

    args.market = args.market.upper()

    if args.market == "SP500":
        company_list = _resolve_sp500_tickers(args)
    else:
        if args.fetch_data:
            raise ValueError("--fetch_data currently supports --market SP500 only.")
        if args.com_paths is None:
            raise ValueError("--com_paths is required for NASDAQ, NYSE and SSE runs.")
        com_paths = [Path(path) for path in args.com_paths]
        nasdaq, nyse, _, sse = _load_tickers(com_paths)
        market_map = {
            "NASDAQ": nasdaq,
            "NYSE": nyse,
            "SSE": sse,
        }
        if args.market not in market_map:
            supported = ", ".join(sorted(market_map))
            raise ValueError(f"Unsupported market {args.market!r}. Choose from: {supported}, SP500.")
        company_list = market_map[args.market]

    if not company_list:
        raise ValueError(f"No tickers found for market {args.market!r}.")

    missing = _missing_raw_files(args.raw_dir, args.market, company_list)
    if missing:
        raise FileNotFoundError(_missing_data_message(args, missing))

    num_nodes = len(company_list)
    dataset_kwargs = {
        "representation_mode": args.representation_mode,
        "graph_mode": args.graph_mode,
        "graph_feature_mode": args.graph_feature_mode,
        "mi_neighbors": args.mi_neighbors,
        "seed": args.seed,
    }

    train_ds = GraphDataset(
        args.raw_dir,
        args.generated_dir,
        args.market,
        company_list,
        *train_range,
        args.time_steps,
        "Train",
        **dataset_kwargs,
    )
    val_ds = GraphDataset(
        args.raw_dir,
        args.generated_dir,
        args.market,
        company_list,
        *val_range,
        args.time_steps,
        "Validation",
        **dataset_kwargs,
    )
    test_ds = GraphDataset(
        args.raw_dir,
        args.generated_dir,
        args.market,
        company_list,
        *test_range,
        args.time_steps,
        "Test",
        **dataset_kwargs,
    )

    if args.num_relation != train_ds.num_relations:
        raise ValueError(
            "The number of relations must match the dataset feature relations "
            f"({train_ds.num_relations})."
        )

    if args.diffusion_dims[0] != train_ds.feature_dim:
        print(
            "Adjusting diffusion input dimension "
            f"from {args.diffusion_dims[0]} to {train_ds.feature_dim}."
        )
        args.diffusion_dims[0] = train_ds.feature_dim

    model = MGDPR(
        num_nodes,
        args.diffusion_dims,
        args.ret_in_dim,
        args.ret_inter_dim,
        args.ret_hidden_dim,
        args.ret_out_dim,
        args.post_pro,
        args.num_relation,
        args.diffusion_steps,
        args.zeta,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    ckpt_dir = Path(args.save_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_model.pt"

    best_val_acc = float("-inf")
    best_val_f1 = 0.0
    best_val_mcc = 0.0
    best_epoch = 0
    epochs_completed = 0
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        epochs_completed = epoch
        train_loss, train_acc = train_one_epoch(model, train_ds, optimizer, device)
        val_acc, val_f1, val_mcc = evaluate(model, val_ds, device)

        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} | "
            f"Val: acc={val_acc:.4f}, f1={val_f1:.4f}, mcc={val_mcc:.4f}"
        )

        if val_acc > best_val_acc + args.early_stop_min_delta:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_val_mcc = val_mcc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"→ Saved new best model (acc={val_acc:.4f}) to {ckpt_path}")
        elif args.early_stop_patience is not None:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop_patience:
                print(
                    "Early stopping after "
                    f"{epochs_without_improvement} epochs without validation "
                    f"accuracy improvement. Best epoch: {best_epoch}."
                )
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_acc, test_f1, test_mcc = evaluate(model, test_ds, device)
    print(f"\nTest Results — acc={test_acc:.4f}, f1={test_f1:.4f}, mcc={test_mcc:.4f}")

    if args.results_path:
        results_path = Path(args.results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not results_path.exists()
        with results_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "run_name",
                    "market",
                    "tickers",
                    "representation_mode",
                    "graph_feature_mode",
                    "graph_mode",
                    "mi_neighbors",
                    "feature_dim",
                    "num_nodes",
                    "time_steps",
                    "epochs",
                    "epochs_completed",
                    "best_epoch",
                    "best_val_acc",
                    "best_val_f1",
                    "best_val_mcc",
                    "test_acc",
                    "test_f1",
                    "test_mcc",
                    "checkpoint",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "run_name": args.run_name or args.market,
                    "market": args.market,
                    "tickers": " ".join(company_list),
                    "representation_mode": args.representation_mode,
                    "graph_feature_mode": train_ds.graph_feature_mode,
                    "graph_mode": train_ds.graph_mode,
                    "mi_neighbors": train_ds._resolved_mi_neighbors(),
                    "feature_dim": train_ds.feature_dim,
                    "num_nodes": num_nodes,
                    "time_steps": args.time_steps,
                    "epochs": args.epochs,
                    "epochs_completed": epochs_completed,
                    "best_epoch": best_epoch,
                    "best_val_acc": f"{best_val_acc:.6f}",
                    "best_val_f1": f"{best_val_f1:.6f}",
                    "best_val_mcc": f"{best_val_mcc:.6f}",
                    "test_acc": f"{test_acc:.6f}",
                    "test_f1": f"{test_f1:.6f}",
                    "test_mcc": f"{test_mcc:.6f}",
                    "checkpoint": str(ckpt_path),
                }
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--time_steps", type=int, default=21)
    parser.add_argument("--num_relation", type=int, default=5)
    parser.add_argument("--diffusion_steps", type=int, default=7)
    parser.add_argument("--zeta", type=float, default=1.001)
    parser.add_argument("--market", type=str, default="SP500")
    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Base directory for raw stock CSV files.",
    )
    parser.add_argument(
        "--generated_dir",
        type=str,
        required=True,
        help="Directory where generated graph samples are stored.",
    )
    parser.add_argument(
        "--com_paths",
        nargs=4,
        help="CSVs listing NASDAQ, NYSE, NYSE missing, and SSE symbols.",
    )
    parser.add_argument(
        "--ticker_path",
        type=str,
        default=None,
        help="Single ticker CSV used for SP500-style experiment runs.",
    )
    parser.add_argument(
        "--fetch_data",
        action="store_true",
        help="Download missing SP500 OHLCV CSVs before graph construction.",
    )
    parser.add_argument("--data_start", type=str, default=None)
    parser.add_argument("--data_end", type=str, default=None)
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--allow_partial", action="store_true")
    parser.add_argument("--train_dates", nargs=2, default=["2013-01-01", "2014-12-31"])
    parser.add_argument("--val_dates", nargs=2, default=["2015-01-01", "2015-06-30"])
    parser.add_argument("--test_dates", nargs=2, default=["2015-07-01", "2017-12-31"])
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument(
        "--representation_mode",
        choices=["log_ohlcv", "ratio_ohlcv"],
        default="ratio_ohlcv",
    )
    parser.add_argument(
        "--graph_mode",
        choices=["continuous_mi", "energy_entropy", "energy_entropy_log"],
        default="continuous_mi",
    )
    parser.add_argument(
        "--graph_feature_mode",
        choices=["log_ohlcv", "raw_ohlcv"],
        default="log_ohlcv",
    )
    parser.add_argument("--mi_neighbors", type=int, default=None)
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument(
        "--diffusion_dims",
        nargs="+",
        type=int,
        default=[105, 128, 256, 512, 512, 512, 256, 128, 64],
    )
    parser.add_argument(
        "--ret_in_dim",
        nargs="+",
        type=int,
        default=[128, 256, 512, 512, 512, 256, 128, 64],
    )
    parser.add_argument(
        "--ret_inter_dim",
        nargs="+",
        type=int,
        default=[512] * 8,
    )
    parser.add_argument(
        "--ret_hidden_dim",
        nargs="+",
        type=int,
        default=[1024] * 8,
    )
    parser.add_argument(
        "--ret_out_dim",
        nargs="+",
        type=int,
        default=[256] * 8,
    )
    parser.add_argument("--post_pro", nargs="+", type=int, default=[256, 64, 2])
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args())
