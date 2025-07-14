import os
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, matthews_corrcoef
from graph_dataset_gen import MyDataset
from multi_gdn import MGDPR

# ─── Reproducibility ────────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ─── Training / Validation / Test Loops ────────────────────────────────────────
def train_one_epoch(model, dataset, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for sample in dataset:
        X = sample['X'].to(device)           # (N, F)
        A = sample['A'].to(device)           # (N, N)
        C = sample['Y'].long().to(device)    # (N,) labels

        optimizer.zero_grad()
        logits = model(X, A)                 # → (N, num_classes)
        loss = F.cross_entropy(logits, C)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * C.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == C).sum().item()
        total_samples += C.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc

@torch.no_grad()
def evaluate(model, dataset, device):
    model.eval()
    all_preds = []
    all_labels = []

    for sample in dataset:
        X = sample['X'].to(device)
        A = sample['A'].to(device)
        C = sample['Y'].long().to(device)

        logits = model(X, A)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(C.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    acc = (y_pred == y_true).mean()
    f1  = f1_score(y_true, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)
    return acc, f1, mcc

# ─── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    set_seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Date ranges
    train_range = args.train_dates
    val_range   = args.val_dates
    test_range  = args.test_dates

    # Company lists
    # (you already populate these as before)
    NASDAQ_list, NYSE_list, NYSE_missing, SSE_list = [], [], [], []
    for path, target in zip(args.com_paths, [NASDAQ_list, NYSE_list, NYSE_missing, SSE_list]):
        with open(path, 'r') as f:
            for line in f:
                target.append(line.strip().split(',')[0])
    NYSE_list = [c for c in NYSE_list if c not in NYSE_missing]

    # 1) select the tickers for the chosen market
    if args.market == 'NASDAQ':
        company_list = NASDAQ_list
    elif args.market == 'NYSE':
        company_list = NYSE_list
    elif args.market == 'SSE':
        company_list = SSE_list
    else:
        raise ValueError(f"Unsupported market {args.market!r}")

    # 2) derive number of nodes
    num_nodes = len(company_list)
    time_steps      = args.time_steps
    num_relation    = args.num_relation
    zeta            = args.zeta
    diffusion_steps = args.diffusion_steps

    # 3) instantiate datasets using company_list
    train_ds = MyDataset(args.raw_dir,
                         args.generated_dir,
                         args.market,
                         company_list,
                         *train_range,
                         time_steps,
                         'Train')
    val_ds   = MyDataset(args.raw_dir,
                         args.generated_dir,
                         args.market,
                         company_list,
                         *val_range,
                         time_steps,
                         'Validation')
    test_ds  = MyDataset(args.raw_dir,
                         args.generated_dir,
                         args.market,
                         company_list,
                         *test_range,
                         time_steps,
                         'Test')

    # 4) build model using the dynamic num_nodes
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
        args.zeta
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_ds, optimizer, device)
        val_acc, val_f1, val_mcc = evaluate(model, val_ds, device)

        print(f"[Epoch {epoch}/{args.epochs}] "
              f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} | "
              f"Val: acc={val_acc:.4f}, f1={val_f1:.4f}, mcc={val_mcc:.4f}")

        # Checkpoint on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.save_dir, 'best_model.pt')
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"→ Saved new best model (acc={val_acc:.4f}) to {ckpt_path}")

    # Final test
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_acc, test_f1, test_mcc = evaluate(model, test_ds, device)
    print(f"\nTest Results — acc={test_acc:.4f}, f1={test_f1:.4f}, mcc={test_mcc:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--seed',           type=int,   default=42)
    p.add_argument('--epochs',         type=int,   default=3000)
    p.add_argument('--lr',             type=float, default=2e-4)
    p.add_argument('--time_steps',     type=int,   default=21)
    p.add_argument('--num_relation',   type=int,   default=5)
    p.add_argument('--diffusion_steps',type=int,   default=7)
    p.add_argument('--zeta',           type=float, default=1.001)
    p.add_argument('--market',         type=str,   default='NASDAQ')
    p.add_argument('--raw_dir',         type=str, required=True,
                   help='base directory for raw stock CSV files')
    p.add_argument('--generated_dir',   type=str, required=True,
                   help='directory where generated feature data is stored (was “des”)')
    p.add_argument('--com_paths',     nargs=3,    required=True,
                   help='CSVs listing NASDAQ, NYSE, SSE, missing symbols')
    p.add_argument('--train_dates',   nargs=2,    default=['2013-01-01','2014-12-31'])
    p.add_argument('--val_dates',     nargs=2,    default=['2015-01-01','2015-06-30'])
    p.add_argument('--test_dates',    nargs=2,    default=['2015-07-01','2017-12-31'])
    p.add_argument('--save_dir',      type=str,   default='./checkpoints')
    # model dims
    p.add_argument('--diffusion_dims', nargs='+', type=int, default=[105,128,256,512,512,512,256,128,64])
    p.add_argument('--ret_in_dim',    nargs='+', type=int, default=[128,256,512,512,512,256,128,64])
    p.add_argument('--ret_inter_dim', nargs='+', type=int, default=[512]*8)
    p.add_argument('--ret_hidden_dim',nargs='+', type=int, default=[1024]*8)
    p.add_argument('--ret_out_dim',   nargs='+', type=int, default=[256]*8)
    p.add_argument('--post_pro',      nargs='+', type=int, default=[256,64,2])

    args = p.parse_args()
    main(args)
