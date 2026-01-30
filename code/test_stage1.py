import os
import json
import time
import argparse
import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm
from utils.dataset_coarse import win01_np, win_denorm

# Assume your model definition is imported here; adjust import if needed
try:
    from net.coarse_net import UNet3d_Weighted_wa
except ImportError:
    pass  # Prevent immediate crash if file is missing; ensure correct import at runtime


# ================ Utility Functions (unchanged) ================
def _ckpt_path(ckpt_dir: str, epoch):
    mapping = {
        'latest': os.path.join(ckpt_dir, 'latest.pth'),
        'best_mae': os.path.join(ckpt_dir, 'best_mae_model.pth'),
        'best_psnr': os.path.join(ckpt_dir, 'best_psnr_model.pth'),
        'best_model': os.path.join(ckpt_dir, 'best_model.pth'),
    }
    key = epoch.strip().lower()
    target_path = mapping.get(key)

    if target_path is None or not os.path.exists(target_path):
        if os.path.exists(ckpt_dir):
            avail = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
            print(f"[Info] Available checkpoints: {avail}")

            # Auto-correction logic
            if key == 'best_psnr' and 'best_psnr_model.pth' in avail:
                return os.path.join(ckpt_dir, 'best_psnr_model.pth')
            if key == 'best_mae' and 'best_mae_model.pth' in avail:
                return os.path.join(ckpt_dir, 'best_mae_model.pth')
            if key == 'best_mae' and 'best_loss.pth' in avail:
                return os.path.join(ckpt_dir, 'best_loss.pth')

        raise FileNotFoundError(f"Cannot find checkpoint for key '{key}' in {ckpt_dir}")

    return target_path


def _uid_from_name(name: str) -> str:
    return name[:-7] if name.endswith('.nii.gz') else (name[:-4] if name.endswith('.nii') else name)


def _load_patient_list_from_json(input_dir):
    SPLITS_JSON = './Data/splits.json'
    # If input_dir is not default, try looking one level up for splits.json (optional fallback)
    if not os.path.isfile(SPLITS_JSON):
        SPLITS_JSON = os.path.join(os.path.dirname(input_dir.rstrip('/')), 'splits.json')

    if not os.path.isfile(SPLITS_JSON):
        raise FileNotFoundError(f'Missing {SPLITS_JSON}')

    with open(SPLITS_JSON, 'r', encoding='utf-8') as f:
        s = json.load(f)
    uids = []
    for key in ['train', 'val', 'test']:
        if key in s and isinstance(s[key], list):
            uids.extend(s[key])
    names = []
    for uid in uids:
        for ext in ('.nii.gz', '.nii'):
            p = os.path.join(input_dir, uid + ext)
            if os.path.exists(p):
                names.append(uid + ext)
                break
    return names


def _hann_1d_soft(L: int, floor: float = 0.05) -> np.ndarray:
    if L <= 1: return np.ones((L,), dtype=np.float32)
    n = np.arange(L, dtype=np.float32)
    w = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (L - 1)))
    w = floor + (1.0 - floor) * w
    return w.astype(np.float32)


def _hann_3d_soft(sz: int, sy: int, sx: int, floor: float = 0.05) -> np.ndarray:
    wz = _hann_1d_soft(sz, floor)
    wy = _hann_1d_soft(sy, floor)
    wx = _hann_1d_soft(sx, floor)
    w = wz[:, None, None] * wy[None, :, None] * wx[None, None, :]
    m = np.max(w)
    if m > 0: w = w / m
    return w.astype(np.float32)


def _make_starts(L, P, S):
    starts = np.arange(0, max(L - P, 0) + 1, S, dtype=int)
    if len(starts) == 0 or starts[-1] + P < L:
        starts = np.append(starts, L - P)
    return starts


# ================ Main Workflow ================
if __name__ == '__main__':
    # 1. Parse arguments
    parser = argparse.ArgumentParser(description="Stage 1 Inference")

    # Basic configuration
    parser.add_argument('--gpu', type=str, default='cuda:0', help='GPU ID')
    parser.add_argument('--model_name', type=str, default='Coarse', help='Run folder name')
    parser.add_argument('--epoch', type=str, default='best_mae',
                        choices=['best_mae', 'best_psnr', 'latest'], help='Checkpoint type')

    # Path configuration
    parser.add_argument('--input_dir', type=str, default='./Data/input/', help='Input data directory')
    parser.add_argument('--save_dir', type=str, default='./Data/Stage1Results/', help='Output directory')

    # Window and patch configuration
    parser.add_argument('--window', type=float, nargs=2, default=[200.0, 3000.0], help='Window level (Min Max)')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[128, 192, 192], help='Patch size (Z X Y)')

    # Other settings
    parser.add_argument('--save_domain', type=str, default='unit', help='Output domain')

    args = parser.parse_args()

    # 2. Assign arguments to original variable names (keep downstream logic identical)
    gpu = args.gpu
    epoch = args.epoch
    model_name = args.model_name
    input_data_dir = args.input_dir
    save_dir = args.save_dir

    # Derived variables
    ckpt_dir = f'./runs/{model_name}/checkpoints/'
    WINDOW = tuple(args.window)
    SAVE_DOMAIN = args.save_domain
    CLIP_UNIT_BEFORE_SAVE = True
    use_amp = True

    patchZ, patchX, patchY = args.patch_size
    strideZ = patchZ // 2
    strideX = patchX // 2
    strideY = patchY // 2

    os.makedirs(save_dir, exist_ok=True)

    # Start execution
    torch.backends.cudnn.benchmark = True

    # Note: Adjusted parameter passing here since original function relied on global input_data_dir
    patient_list = _load_patient_list_from_json(input_data_dir)
    print(f'[Info] Total cases: {len(patient_list)}')

    # Load weights
    ckpt_path = _ckpt_path(ckpt_dir, epoch)
    print(f'[Info] Loading checkpoint: {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location=gpu)

    # Instantiate model (keeping your original logic)
    # Note: Your original code had two model= lines; the last one takes effect
    model = UNet3d_Weighted_wa(model_chl=16, den=0.8).cuda()

    state = checkpoint.get('model', checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()

    hann3d = _hann_3d_soft(patchZ, patchX, patchY, floor=0.05)

    # Inference loop
    for index, patient in enumerate(tqdm(patient_list, desc='S1 Inference', unit='case'), 1):
        tic = time.time()
        uid = _uid_from_name(patient)

        in_path = os.path.join(input_data_dir, patient)
        inp_xyz = nib.load(in_path).get_fdata(dtype=np.float32)
        inp_zyx = np.transpose(inp_xyz, (2, 1, 0)).copy()

        Z, Y, X = inp_zyx.shape
        inp_zyx_unit = win01_np(inp_zyx, *WINDOW)

        out_sum = np.zeros_like(inp_zyx_unit, dtype=np.float32)
        weight_sum = np.zeros_like(inp_zyx_unit, dtype=np.float32)

        z_starts = _make_starts(Z, patchZ, strideZ)
        y_starts = _make_starts(Y, patchX, strideX)
        x_starts = _make_starts(X, patchY, strideY)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            for z0 in tqdm(z_starts, desc=f'{uid}', unit='z', leave=False):
                z1 = z0 + patchZ
                for y0 in y_starts:
                    y1 = y0 + patchX
                    for x0 in x_starts:
                        x1 = x0 + patchY

                        patch_unit = inp_zyx_unit[z0:z1, y0:y1, x0:x1]
                        tin = torch.from_numpy(patch_unit[None, None, ...]).to(gpu)

                        pred_unit = model(tin)
                        pred_unit = pred_unit.float().squeeze().detach().cpu().numpy()

                        w = hann3d
                        out_sum[z0:z1, y0:y1, x0:x1] += pred_unit * w
                        weight_sum[z0:z1, y0:y1, x0:x1] += w

        eps = 1e-6
        out_unit = out_sum / np.clip(weight_sum, eps, None)

        if CLIP_UNIT_BEFORE_SAVE:
            np.clip(out_unit, 0.0, 1.0, out=out_unit)

        if SAVE_DOMAIN == 'unit':
            out_zyx = out_unit
        else:
            out_zyx = win_denorm(out_unit, *WINDOW)

        out_xyz = np.transpose(out_zyx, (2, 1, 0)).astype(np.float32)
        np.save(os.path.join(save_dir, uid + '.npy'), out_xyz)

        toc = time.time()
        print(f'[Done] {uid} ({index}/{len(patient_list)})')