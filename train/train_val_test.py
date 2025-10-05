"""Training, validation and testing entrypoint for MGDPR."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dataset.graph_dataset_gen import GraphDataset, GraphSample  # noqa: E402
from model.multi_gdn import MGDPR  # noqa: E402


# ─── Reproducibility ────────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
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


# ─── Main ──────────────────────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.time_steps <= 1:
        raise ValueError("`time_steps` must be greater than 1 to create labels.")
    if args.num_relation <= 0:
        raise ValueError("`num_relation` must be positive.")
    if args.diffusion_steps <= 0:
        raise ValueError("`diffusion_steps` must be positive.")
    if not args.ret_in_dim:
        raise ValueError("At least one retention layer must be specified.")
    if len(args.diffusion_dims) != len(args.ret_in_dim) + 1:
        raise ValueError(
            "`diffusion_dims` must be exactly one element longer than the retention dimension lists."
        )
    if not (
        len(args.ret_in_dim)
        == len(args.ret_inter_dim)
        == len(args.ret_hidden_dim)
        == len(args.ret_out_dim)
    ):
        raise ValueError("All retention dimension lists must have equal length.")
    if len(args.post_pro) < 2:
        raise ValueError("`post_pro` must contain at least input and output dimensions.")
    if args.post_pro[0] != args.ret_out_dim[-1]:
        raise ValueError(
            "`post_pro` must start with the last retention output dimension to ensure shape continuity."
        )

    train_range = args.train_dates
    val_range = args.val_dates
    test_range = args.test_dates

    com_paths = [Path(path) for path in args.com_paths]
    nasdaq, nyse, _, sse = _load_tickers(com_paths)

    market_map = {
        "NASDAQ": nasdaq,
        "NYSE": nyse,
        "SSE": sse,
    }
    if args.market not in market_map:
        supported = ", ".join(sorted(market_map))
        raise ValueError(f"Unsupported market {args.market!r}. Choose from: {supported}.")

    company_list = market_map[args.market]
    if not company_list:
        raise ValueError(f"No tickers found for market {args.market!r}.")

    num_nodes = len(company_list)

    train_ds = GraphDataset(
        args.raw_dir,
        args.generated_dir,
        args.market,
        company_list,
        *train_range,
        args.time_steps,
        "Train",
    )
    val_ds = GraphDataset(
        args.raw_dir,
        args.generated_dir,
        args.market,
        company_list,
        *val_range,
        args.time_steps,
        "Validation",
    )
    test_ds = GraphDataset(
        args.raw_dir,
        args.generated_dir,
        args.market,
        company_list,
        *test_range,
        args.time_steps,
        "Test",
    )

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
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_ds, optimizer, device)
        val_acc, val_f1, val_mcc = evaluate(model, val_ds, device)

        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} | "
            f"Val: acc={val_acc:.4f}, f1={val_f1:.4f}, mcc={val_mcc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"→ Saved new best model (acc={val_acc:.4f}) to {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_acc, test_f1, test_mcc = evaluate(model, test_ds, device)
    print(f"\nTest Results — acc={test_acc:.4f}, f1={test_f1:.4f}, mcc={test_mcc:.4f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--time_steps", type=int, default=21)
    parser.add_argument("--num_relation", type=int, default=5)
    parser.add_argument("--diffusion_steps", type=int, default=7)
    parser.add_argument("--zeta", type=float, default=1.001)
    parser.add_argument("--market", type=str, default="NASDAQ")
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
        required=True,
        help="CSVs listing NASDAQ, NYSE, NYSE missing, and SSE symbols.",
    )
    parser.add_argument("--train_dates", nargs=2, default=["2013-01-01", "2014-12-31"])
    parser.add_argument("--val_dates", nargs=2, default=["2015-01-01", "2015-06-30"])
    parser.add_argument("--test_dates", nargs=2, default=["2015-07-01", "2017-12-31"])
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
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
