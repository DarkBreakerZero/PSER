import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import math
from torch.cuda.amp import autocast, GradScaler
from utils.AverageMeter import AverageMeter
from utils.dataset_fine import npzFileReader
from net.fine_net import SnakeDenseUnet2d

# =================================================================
#  window para
# =================================================================
L_WIN = 200.0
H_WIN = 3000.0
WIN_DEN = H_WIN - L_WIN


# =================================================================
# 1. Loss
# =================================================================
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        kernel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
        kernel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, pred, target):
        pred_grad_x = F.conv2d(pred, self.kernel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.kernel_y, padding=1)
        target_grad_x = F.conv2d(target, self.kernel_x, padding=1)
        target_grad_y = F.conv2d(target, self.kernel_y, padding=1)
        loss = F.l1_loss(torch.abs(pred_grad_x), torch.abs(target_grad_x)) + \
               F.l1_loss(torch.abs(pred_grad_y), torch.abs(target_grad_y))
        return loss


# =================================================================
# 2. data progress
# =================================================================
class NPZUnitWrapper(torch.utils.data.Dataset):
    def __init__(self, base_ds, use_norm: bool = True, is_train: bool = True):
        self.base = base_ds
        self.use_norm = use_norm
        self.is_train = is_train

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i: int):
        d = self.base[i]
        inp, lbl = d['input'], d['label']
        if not isinstance(inp, torch.Tensor): inp = torch.from_numpy(inp)
        if not isinstance(lbl, torch.Tensor): lbl = torch.from_numpy(lbl)

        # 归一化
        if self.use_norm:
            inp = torch.clamp(inp.float(), 0.0, 1.0)
            # 使用全局变量 L_WIN, H_WIN, WIN_DEN
            lbl = torch.clamp(lbl.float(), L_WIN, H_WIN)
            lbl = (lbl - L_WIN) / max(WIN_DEN, 1e-6)

        if inp.ndim == 4 and inp.shape[0] == 1: inp = inp.squeeze(0)
        if lbl.dim() == 2: lbl = lbl.unsqueeze(0)

        # 数据增强 (Augmentation)
        if self.is_train:
            if random.random() > 0.5:  # H-Flip
                inp = torch.flip(inp, dims=[-1])
                lbl = torch.flip(lbl, dims=[-1])
            if random.random() > 0.5:  # V-Flip
                inp = torch.flip(inp, dims=[-2])
                lbl = torch.flip(lbl, dims=[-2])
            k = random.randint(0, 3)  # Rotate
            if k > 0:
                inp = torch.rot90(inp, k, dims=[-2, -1])
                lbl = torch.rot90(lbl, k, dims=[-2, -1])

        return {'input': inp.to(torch.float32), 'label': lbl.to(torch.float32)}


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =================================================================
# 3. Train &valid
# =================================================================
def train(train_loader, model, optimizer, scaler, writer, epoch, criterion_char, criterion_grad):
    model.train()
    loss_meter = AverageMeter()
    w_char = 1.0
    w_grad = 0.5

    pbar = tqdm(train_loader, desc=f"Ep {epoch + 1} [Train]", unit="batch")
    for data in pbar:
        gt = data["label"].cuda(non_blocking=True)
        inp = data["input"].cuda(non_blocking=True)

        optimizer.zero_grad()
        with autocast(dtype=torch.float16):
            pred = model(inp)
            pred = torch.clamp(pred, 0.0, 1.0)

            l_char = criterion_char(pred, gt)
            l_grad = criterion_grad(pred, gt)
            total_loss = w_char * l_char + w_grad * l_grad

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(total_loss.item())
        pbar.set_postfix(Loss=f"{loss_meter.avg:.4f}")

    writer.add_scalar('Loss/Train', loss_meter.avg, epoch + 1)


def valid(valid_loader, model, writer, epoch):
    model.eval()
    mae_meter = AverageMeter()
    psnr_meter = AverageMeter()

    with torch.no_grad():
        for data in tqdm(valid_loader, desc=f"Ep {epoch + 1} [Valid]", leave=False):
            gt = data["label"].cuda(non_blocking=True)
            inp = data["input"].cuda(non_blocking=True)

            with autocast(dtype=torch.float16):
                pred = model(inp)
                pred = torch.clamp(pred.float(), 0, 1)

                # Metrics
                mae = F.l1_loss(pred, gt).item()
                mse = F.mse_loss(pred, gt).item()
                psnr = 10 * math.log10(1 / (mse + 1e-8))

                mae_meter.update(mae)
                psnr_meter.update(psnr)

    writer.add_scalar('Metric/Valid_MAE', mae_meter.avg, epoch + 1)
    writer.add_scalar('Metric/Valid_PSNR', psnr_meter.avg, epoch + 1)

    print(f"Ep {epoch + 1}: MAE={mae_meter.avg:.6f} | PSNR={psnr_meter.avg:.2f} dB")
    return mae_meter.avg, psnr_meter.avg


# =================================================================
# 4. 主程序
# =================================================================
if __name__ == "__main__":
    # Argparse
    parser = argparse.ArgumentParser(description="Train 2D Baseline")
    parser.add_argument('--exp_name', type=str, default="Fine", help='Experiment name')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # train
    parser.add_argument('--epochs', type=int, default=120, help='Total epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Base learning rate')
    parser.add_argument('--workers', type=int, default=16, help='Num workers')

    # argument
    parser.add_argument('--l_win', type=float, default=200.0, help='Window Low')
    parser.add_argument('--h_win', type=float, default=3000.0, help='Window High')

    args = parser.parse_args()


    EXP_NAME = args.exp_name
    SEED = args.seed
    BASE_LR = args.lr
    TOTAL_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.workers
    L_WIN = args.l_win
    H_WIN = args.h_win
    WIN_DEN = H_WIN - L_WIN

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    set_global_seed(SEED)
    cudnn.benchmark = True

    # path
    result_path = f'./runs/{EXP_NAME}/logs/'
    save_dir = f'./runs/{EXP_NAME}/checkpoints/'
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    scaler = GradScaler()
    print(f"[Info] Start 2D Training: {EXP_NAME} | GPU: {args.gpu}")
    print(f"[Info] Window: {L_WIN} ~ {H_WIN}")

    # Loss
    criterion_char = CharbonnierLoss().cuda()
    criterion_grad = GradientLoss().cuda()

    # Dataset
    base_train_reader = npzFileReader('./txt/train_2d_img_list.txt')
    base_valid_reader = npzFileReader('./txt/valid_2d_img_list.txt')
    train_ds = NPZUnitWrapper(base_train_reader, use_norm=True, is_train=True)
    valid_ds = NPZUnitWrapper(base_valid_reader, use_norm=True, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, pin_memory=True,
                              drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, pin_memory=True)

    print("[Info] Building Model...")

    model = SnakeDenseUnet2d(in_chl=3, out_chl=1, model_chl=32).cuda()
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6)

    writer = SummaryWriter(log_dir=result_path)
    best_psnr = 0.0
    min_mae = float('inf')
    start_epoch = 0

    # Resume
    latest_path = os.path.join(save_dir, 'latest.pth')
    if os.path.exists(latest_path):
        print(f"[Resume] Loading checkpoint from {latest_path} ...")
        checkpoint = torch.load(latest_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        best_psnr = checkpoint.get('best_psnr', 0.0)
        min_mae = checkpoint.get('min_mae', float('inf'))

    print(f"Training start from epoch {start_epoch + 1}...")

    for epoch in range(start_epoch, TOTAL_EPOCHS):
        train(train_loader, model, optimizer, scaler, writer, epoch, criterion_char, criterion_grad)

        val_mae, val_psnr = valid(valid_loader, model, writer, epoch)
        scheduler.step()

        state = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_psnr': best_psnr,
            'min_mae': min_mae
        }

        # 1. save Best MAE
        if val_mae < min_mae:
            min_mae = val_mae
            torch.save(state, os.path.join(save_dir, 'best_mae_model.pth'))
            print(f"★ Saved Best MAE: {min_mae:.6f}")

        # 2. save Best PSNR
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(state, os.path.join(save_dir, 'best_psnr_model.pth'))
            print(f"☆ Saved Best PSNR: {best_psnr:.2f}")

        # 3. save Latest
        torch.save(state, os.path.join(save_dir, 'latest.pth'))

    print("Training Finished.")