import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
import nibabel as nib
import argparse
import sys

# Try to import custom dataset
try:
    from utils.dataset_coarse import Random3DPatchDataset
except ImportError:
    print("Warning: datasets.online_reader not found.")

# Import model (keeping original path)
from net.coarse_net import UNet3d_Weighted_wa


# =======================================================
# 1. Argument Parsing (Argparse)
# =======================================================
def get_args():
    parser = argparse.ArgumentParser(description="Train Coarse Restoration Net ")

    # Experiment related
    parser.add_argument("--exp_name", type=str, default="Coarse", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of loader workers")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=2e-4, help="Base learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Total number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--accum_steps", type=int, default=8, help="Gradient accumulation steps")

    # Data related
    # Note: patch_size accepts 3 integers, e.g. --patch_size 128 192 192
    parser.add_argument("--patch_size", type=int, nargs='+', default=[128, 192, 192], help="Input patch size (D, H, W)")
    parser.add_argument("--l_win", type=float, default=200.0, help="Window Level Low")
    parser.add_argument("--h_win", type=float, default=3000.0, help="Window Level High")
    parser.add_argument("--data_root", type=str, default="./Data", help="Root directory for input/label folders")
    parser.add_argument("--split_json", type=str, default="./Data/splits.json", help="Path to splits json")

    args = parser.parse_args()
    return args


# =======================================================
# 2. Loss & Metrics
# =======================================================
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()


def compute_psnr(x, y):
    mse = F.mse_loss(x.float(), y.float()).item()
    if mse == 0: return 100.0
    return 10 * torch.log10(1.0 / torch.tensor(mse)).item()


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =======================================================
# 3. Dataset
# =======================================================
class NiftiValDataset(Dataset):
    def __init__(self, splits_json, L, H, data_root="./Data"):
        with open(splits_json, 'r') as f:
            self.uids = json.load(f)['val']
        self.L, self.H = L, H
        self.D = H - L
        self.data_root = data_root

    def __len__(self): return len(self.uids)

    def __getitem__(self, i):
        uid = self.uids[i]
        # Use os.path.join for robustness, still compatible with original logic
        p_lbl = os.path.join(self.data_root, "label", f"{uid}.nii.gz")
        p_inp = os.path.join(self.data_root, "input", f"{uid}.nii.gz")

        x = nib.load(p_inp).get_fdata().astype(np.float32)
        y = nib.load(p_lbl).get_fdata().astype(np.float32)

        x = (np.clip(x, self.L, self.H) - self.L) / max(self.D, 1e-6)
        y = (np.clip(y, self.L, self.H) - self.L) / max(self.D, 1e-6)

        return {
            "input": torch.from_numpy(np.transpose(x, (2, 1, 0))[None, ...]),
            "label": torch.from_numpy(np.transpose(y, (2, 1, 0))[None, ...]),
            "uid": uid
        }


# =======================================================
# 4. Train & Valid Functions
# =======================================================
def train(train_loader, model, criterion, optimizer, scaler, writer, epoch, accum_steps):
    model.train()
    losses = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")
    optimizer.zero_grad()

    for i, data in enumerate(pbar):
        inp = data['input'].cuda()
        lbl = data['label'].cuda()

        with autocast(enabled=True):
            pred = model(inp)
            loss = criterion(pred, lbl)
            loss_norm = loss / accum_steps

        scaler.scale(loss_norm).backward()

        if (i + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        losses += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = losses / len(train_loader)
    writer.add_scalar('train_loss', avg_loss, epoch + 1)


def valid(valid_loader, model, criterion, writer, epoch):
    model.eval()
    losses = 0.0
    mae_sum = 0.0
    psnr_sum = 0.0

    with torch.no_grad():
        pbar = tqdm(valid_loader, desc=f"Epoch {epoch + 1} [Valid]")
        for data in pbar:
            inp = data['input'].cuda()
            lbl = data['label'].cuda()
            pred = model(inp)

            loss = criterion(pred, lbl)
            losses += loss.item()

            pred_clamped = torch.clamp(pred, 0, 1)
            lbl_clamped = torch.clamp(lbl, 0, 1)

            # Compute metrics
            current_mae = F.l1_loss(pred_clamped, lbl_clamped).item()
            current_psnr = compute_psnr(pred_clamped, lbl_clamped)

            mae_sum += current_mae
            psnr_sum += current_psnr

            pbar.set_postfix(MAE=f"{current_mae:.4f}", PSNR=f"{current_psnr:.2f}")

    avg_loss = losses / len(valid_loader)
    avg_mae = mae_sum / len(valid_loader)
    avg_psnr = psnr_sum / len(valid_loader)

    writer.add_scalar('Valid/Loss', avg_loss, epoch + 1)
    writer.add_scalar('Valid/MAE', avg_mae, epoch + 1)
    writer.add_scalar('Valid/PSNR', avg_psnr, epoch + 1)

    print(f"\n>>> Epoch {epoch + 1}: MAE={avg_mae:.6f} | PSNR={avg_psnr:.2f} dB\n")
    return avg_mae, avg_psnr


# =======================================================
# 5. Main Execution
# =======================================================
def main():
    # 1. Get arguments
    args = get_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    set_global_seed(args.seed)

    # Path setup
    save_dir = f"./runs/{args.exp_name}/checkpoints"
    log_dir = f"./runs/{args.exp_name}/logs"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"Initializing Model for Exp: {args.exp_name}")
    print(
        f"Hyperparameters: LR={args.lr}, BS={args.batch_size}, Accum={args.accum_steps}, Patch={tuple(args.patch_size)}")

    # 2. Model definition
    # (★) Keep the original model selection
    model = UNet3d_Weighted_wa(in_chl=1, out_chl=1, model_chl=16).cuda()
    # model = UNet3d_Weighted_wa_false(in_chl=1, out_chl=1, model_chl=16).cuda()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)
    criterion = CharbonnierLoss().cuda()
    scaler = GradScaler()
    writer = SummaryWriter(log_dir)

    # 3. Data loading
    print("Loading Data...")

    # Training set
    train_ds = Random3DPatchDataset(
        patch_size=tuple(args.patch_size),  # convert list to tuple
        roi_bias=0.6,
        patches_per_vol=24,
        cache_vols=26
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        persistent_workers=True
    )

    # Validation set
    valid_ds = NiftiValDataset(args.split_json, args.l_win, args.h_win, data_root=args.data_root)
    valid_loader = DataLoader(
        valid_ds,
        batch_size=1,
        num_workers=args.num_workers
    )

    # 4. Resume logic
    best_psnr = 0.0
    min_mae = float('inf')
    start_epoch = 0

    latest_path = os.path.join(save_dir, "latest.pth")
    if os.path.exists(latest_path):
        print(f"[Info] Found checkpoint. Resuming...")
        checkpoint = torch.load(latest_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']

        best_psnr = checkpoint.get('best_psnr', 0.0)
        min_mae = checkpoint.get('min_mae', float('inf'))

        print(f"[Info] Resumed from Epoch {start_epoch}. Best PSNR: {best_psnr:.2f}, Min MAE: {min_mae:.6f}")
    else:
        print("[Info] Starting from scratch.")

    print(f"Start Training {args.exp_name} from Epoch {start_epoch + 1}...")

    # 5. Main loop
    for epoch in range(start_epoch, args.epochs):
        # Training
        train(train_loader, model, criterion, optimizer, scaler, writer, epoch, args.accum_steps)

        # Validation
        val_mae, val_psnr = valid(valid_loader, model, criterion, writer, epoch)

        scheduler.step()

        state = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_psnr': best_psnr,
            'min_mae': min_mae
        }

        # (★) Saving strategy (logic remains unchanged)

        # 1. Save champion model: lowest MAE (most accurate)
        if val_mae < min_mae:
            min_mae = val_mae
            state['min_mae'] = min_mae
            torch.save(state, os.path.join(save_dir, "best_mae_model.pth"))
            print(f"★ [Saved] Best MAE: {min_mae:.6f} (PSNR: {val_psnr:.2f})")

        # 2. Save reference model: highest PSNR (for comparison/backup)
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            state['best_psnr'] = best_psnr
            torch.save(state, os.path.join(save_dir, "best_psnr_model.pth"))
            print(f"☆ [Saved] Best PSNR: {best_psnr:.2f} (MAE: {val_mae:.6f})")

        # 3. Save latest model (only for resuming interrupted training)
        torch.save(state, os.path.join(save_dir, "latest.pth"))


if __name__ == "__main__":
    main()