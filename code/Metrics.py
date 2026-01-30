import os
import json
import numpy as np
import nibabel as nib
import pandas as pd
from typing import List, Optional, Dict
from tqdm import tqdm
import warnings
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ==================== Configurable Items ====================
WINDOW_L = 200.0  # Normalization lower bound
WINDOW_H = 3000.0  # Normalization upper bound
SPLIT = 'test'  # Test set
ROOT_DIR = './Data'

# Path configuration
GT_DIR = os.path.join(ROOT_DIR, 'label')
FDK_DIR = os.path.join(ROOT_DIR, 'input')
S1_DIR = os.path.join(ROOT_DIR, 'Stage1Results')  # Stage 1: .npy
S2_DIR = os.path.join(ROOT_DIR, 'Stage2Results')  # Stage 2: .nii (newly added)

SPLITS_JSON = os.path.join(ROOT_DIR, 'splits.json')
SSIM_BASE_WIN = 11  # SSIM window size


# =================================================

# ----------------- I/O Helpers -----------------
def _resolve(uid: str, folder: str, exts: List[str]) -> Optional[str]:
    if not os.path.isdir(folder): return None
    for ext in exts:
        p = os.path.join(folder, uid + ext)
        if os.path.exists(p): return p
    return None


def _load_nii_xyz(path: str) -> Optional[np.ndarray]:
    try:
        return nib.load(path).get_fdata(dtype=np.float32)
    except Exception:
        return None


def _load_npy_xyz(path: str) -> Optional[np.ndarray]:
    try:
        return np.load(path).astype(np.float32)
    except Exception:
        return None


# ----------------- Preprocessing -----------------
def win01_xyz(arr: np.ndarray, L: float, H: float) -> np.ndarray:
    """Fixed window [L,H] → [0,1]"""
    x = arr.astype(np.float32, copy=False)
    x = np.clip(x, L, H)
    x = (x - L) / max(H - L, 1e-6)
    return x


def crop_to_min_shape(arrs: List[np.ndarray]) -> List[np.ndarray]:
    """Multi-modal alignment: crop to the smallest shape"""
    shapes = np.array([a.shape for a in arrs])
    s = shapes.min(axis=0)
    return [a[:s[0], :s[1], :s[2]] for a in arrs]


# ----------------- Metric Calculation -----------------
def _pick_win_size_2d(h: int, w: int, base: int) -> int:
    md = min(h, w)
    wsize = min(base, md if md % 2 == 1 else md - 1)
    return wsize if wsize >= 3 else (3 if md >= 3 else 1)


def ssim_slice_mean(gt_norm_xyz: np.ndarray, pred_norm_xyz: np.ndarray) -> float:
    Z = gt_norm_xyz.shape[2]
    scores = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(Z):
            g, p = gt_norm_xyz[..., k], pred_norm_xyz[..., k]
            if g.max() < 1e-6:
                continue
            win = _pick_win_size_2d(g.shape[0], g.shape[1], SSIM_BASE_WIN)
            try:
                val = structural_similarity(g, p, data_range=1.0, win_size=win)
                if np.isfinite(val):
                    scores.append(val)
            except ValueError:
                pass
    return float(np.mean(scores)) if len(scores) else 0.0


def calculate_metrics(gt_norm_xyz: np.ndarray, pred_norm_xyz: np.ndarray) -> Dict[str, float]:
    # All metrics are computed in [0,1] range
    psnr = peak_signal_noise_ratio(gt_norm_xyz, pred_norm_xyz, data_range=1.0)
    ssim = ssim_slice_mean(gt_norm_xyz, pred_norm_xyz)
    mae = float(np.mean(np.abs(gt_norm_xyz - pred_norm_xyz)))
    return {"MAE": mae, "SSIM": ssim * 100.0, "PSNR": float(psnr)}


# ----------------- Formatting & Output -----------------
def fmt_mean_std(vals: np.ndarray) -> str:
    if vals.size == 0: return "N/A"
    return f"{np.mean(vals):.2f}±{np.std(vals, ddof=0):.2f}"


def mark_best(entries: List[str], greater_is_better: bool) -> List[str]:
    return entries


# ----------------- Main Workflow -----------------
def main():
    if not os.path.exists(SPLITS_JSON):
        raise FileNotFoundError(f"Missing {SPLITS_JSON}")

    with open(SPLITS_JSON, 'r', encoding='utf-8') as f:
        uids: List[str] = json.load(f)[SPLIT]

    print(f"[INFO] Split={SPLIT} | Cases={len(uids)}")
    print(f"[INFO] Evaluating: FDK, CR-Net (Stage1), FT-Net (Stage2)")

    per_case_rows = []
    methods_found = set()

    # (★) 1. Modified: define Stage 2 (FT-Net) data domain as 'unit' ([0,1])
    DOMAIN = {
        "FDK": "raw",     # needs windowing
        "CR-Net": "unit", # already in [0,1]
        "FT-Net": "unit"
    }

    for uid in tqdm(uids, desc="Evaluating", unit="case"):
        gt = _load_nii_xyz(_resolve(uid, GT_DIR, ['.nii.gz', '.nii']))
        if gt is None:
            continue

        # Stage 2 results are in .nii / .nii.gz format
        all_preds = {
            "FDK": _load_nii_xyz(_resolve(uid, FDK_DIR, ['.nii.gz', '.nii'])),
            "CR-Net": _load_npy_xyz(_resolve(uid, S1_DIR, ['.npy'])),
            "FT-Net": _load_nii_xyz(_resolve(uid, S2_DIR, ['.nii.gz', '.nii']))
        }

        # Filter out missing files
        valid_preds = {tag: pred for tag, pred in all_preds.items() if pred is not None}
        if not valid_preds:
            continue

        methods_found.update(valid_preds.keys())

        # Crop to align shapes
        all_arrs = [gt] + list(valid_preds.values())
        gt_c, *preds_c = crop_to_min_shape(all_arrs)

        # Normalize GT to [0,1]
        gt_n = win01_xyz(gt_c, WINDOW_L, WINDOW_H)

        for (tag, _), pred_c in zip(valid_preds.items(), preds_c):
            # Process input according to data domain
            dom = DOMAIN.get(tag, "raw")

            if dom == "raw":
                # Raw CT values → Windowing → [0,1]
                pred_n = win01_xyz(pred_c, WINDOW_L, WINDOW_H)
            else:
                # Already in [0,1] → just clip for safety
                pred_n = np.clip(pred_c.astype(np.float32, copy=False), 0.0, 1.0)

            # Compute metrics
            metrics = calculate_metrics(gt_n, pred_n)

            per_case_rows.append({
                "uid": uid,
                "method": tag,
                "MAE": metrics["MAE"],
                "MAE_HUlike": metrics["MAE"] * (WINDOW_H - WINDOW_L),  # scale back to HU-like magnitude for intuition
                "SSIM": metrics["SSIM"],
                "PSNR": metrics["PSNR"]
            })

    if not per_case_rows:
        print("[ERROR] No valid cases found.")
        return

    # Save per-case results to CSV
    df = pd.DataFrame(per_case_rows)
    save_csv = os.path.join(ROOT_DIR, "evaluation_results.csv")
    df.to_csv(save_csv, index=False)
    print(f"[OK] Saved CSV: {save_csv}")

    # (★) 3. Modified: include FT-Net in the printed table
    # Ensure FT-Net appears in the print order
    methods_order = ["FDK", "CR-Net", "FT-Net"]
    methods_to_print = [m for m in methods_order if m in methods_found]

    # Table printing logic
    def get_entries(key, greater_is_better):
        vals = [df[df["method"] == m][key].values for m in methods_to_print]
        entries = [fmt_mean_std(v) for v in vals]

        # Mark best value with **
        nums = [v.mean() if v.size > 0 else np.nan for v in vals]
        if not np.all(np.isnan(nums)):
            idx = np.nanargmax(nums) if greater_is_better else np.nanargmin(nums)
            entries[idx] = f"**{entries[idx]}**"
        return entries

    mae_hu_entries = get_entries("MAE_HUlike", greater_is_better=False)
    ssim_entries = get_entries("SSIM", greater_is_better=True)
    psnr_entries = get_entries("PSNR", greater_is_better=True)

    col_width = 20
    header = f"{'Metrics':<15}" + "".join([f"{m:<{col_width}}" for m in methods_to_print])
    separator = "-" * len(header)

    print("\n" + "=" * len(header))
    print(" " * 15 + "Final Evaluation Results")
    print("=" * len(header))
    print(header)
    print(separator)
    print(f"{'MAE (HU)':<15}" + "".join([f"{v:<{col_width}}" for v in mae_hu_entries]))
    print(f"{'SSIM (%)':<15}" + "".join([f"{v:<{col_width}}" for v in ssim_entries]))
    print(f"{'PSNR (dB)':<15}" + "".join([f"{v:<{col_width}}" for v in psnr_entries]))
    print(separator)
    print("=" * len(header) + "\n")


if __name__ == "__main__":
    main()