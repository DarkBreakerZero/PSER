from pathlib import Path
import warnings

import nibabel as nib
import numpy as np
from tqdm import tqdm
from skimage.morphology import skeletonize
from skimage.measure import euler_number
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ==================== Configurable Items ====================
GT_DIR = Path('./Data/label')
PRED_DIR = Path('./Data/prediction')

# raw: normalize using the fixed intensity window
# unit: values are already in [0, 1]
GT_DOMAIN = 'raw'
PRED_DOMAIN = 'unit'

WINDOW = (200.0, 3000.0)
THRESHOLD = 0.10
EULER_CONNECTIVITY = 3
SSIM_WIN_SIZE = 11

EXTENSIONS = ('.nii.gz', '.nii', '.npy')
# ============================================================


def case_id(path: Path) -> str:
    if path.name.endswith('.nii.gz'):
        return path.name[:-7]

    return path.stem


def find_cases(folder: Path) -> dict[str, Path]:
    files = {}

    for ext in EXTENSIONS:
        for path in folder.glob(f'*{ext}'):
            files.setdefault(case_id(path), path)

    return files


def load_volume(path: Path) -> np.ndarray:
    if path.suffix == '.npy':
        return np.load(path).astype(np.float32)

    return nib.load(str(path)).get_fdata(dtype=np.float32)


def normalize(
    volume: np.ndarray,
    domain: str
) -> np.ndarray:
    if domain == 'unit':
        return np.clip(volume, 0.0, 1.0)

    if domain == 'raw':
        low, high = WINDOW
        volume = np.clip(volume, low, high)
        return (volume - low) / (high - low)

    raise ValueError(
        "domain must be either 'raw' or 'unit'"
    )


def centerline(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)

    try:
        return np.asarray(
            skeletonize(mask, method='lee'),
            dtype=bool
        )

    except TypeError:
        return np.asarray(
            skeletonize(mask),
            dtype=bool
        )


def pick_ssim_win_size(
    height: int,
    width: int
) -> int:
    min_size = min(height, width)

    win_size = min(
        SSIM_WIN_SIZE,
        min_size if min_size % 2 == 1 else min_size - 1
    )

    return win_size if win_size >= 3 else 1


def calculate_ssim(
    gt_norm: np.ndarray,
    pred_norm: np.ndarray
) -> float:
    scores = []

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        for index in range(gt_norm.shape[2]):
            gt_slice = gt_norm[..., index]
            pred_slice = pred_norm[..., index]

            if gt_slice.max() < 1e-6:
                continue

            win_size = pick_ssim_win_size(
                gt_slice.shape[0],
                gt_slice.shape[1]
            )

            if win_size < 3:
                continue

            try:
                score = structural_similarity(
                    gt_slice,
                    pred_slice,
                    data_range=1.0,
                    win_size=win_size
                )

                if np.isfinite(score):
                    scores.append(score)

            except ValueError:
                continue

    return float(np.mean(scores)) if scores else 0.0


def calculate_metrics(
    gt: np.ndarray,
    pred: np.ndarray
) -> dict[str, float]:
    # Crop GT and prediction to their common minimum shape
    shape = np.minimum(gt.shape, pred.shape)
    slices = tuple(
        slice(0, int(size))
        for size in shape
    )

    gt_norm = normalize(
        gt[slices],
        GT_DOMAIN
    )

    pred_norm = normalize(
        pred[slices],
        PRED_DOMAIN
    )

    # ---------------- Image Metrics ----------------
    # First calculate MAE in the normalized [0, 1] range,
    # then multiply by the fixed window width
    normalized_mae = float(
        np.mean(
            np.abs(gt_norm - pred_norm)
        )
    )

    mae = normalized_mae * (
        WINDOW[1] - WINDOW[0]
    )

    psnr = peak_signal_noise_ratio(
        gt_norm,
        pred_norm,
        data_range=1.0
    )

    ssim = calculate_ssim(
        gt_norm,
        pred_norm
    )

    # ---------------- Vessel Masks ----------------
    gt_mask = gt_norm >= THRESHOLD
    pred_mask = pred_norm >= THRESHOLD

    # ---------------- Centerlines ----------------
    gt_line = centerline(gt_mask)
    pred_line = centerline(pred_mask)

    n_gt = int(gt_line.sum())
    n_pred = int(pred_line.sum())

    if n_gt == 0 and n_pred == 0:
        precision = 1.0
        recall = 1.0

    elif n_gt == 0 or n_pred == 0:
        precision = 0.0
        recall = 0.0

    else:
        precision = (
            np.logical_and(
                pred_line,
                gt_mask
            ).sum()
            / n_pred
        )

        recall = (
            np.logical_and(
                gt_line,
                pred_mask
            ).sum()
            / n_gt
        )

    cldice = (
        0.0
        if precision + recall == 0
        else (
            2.0
            * precision
            * recall
            / (precision + recall)
        )
    )

    # ---------------- Centerline-based Euler ----------------
    gt_euler = euler_number(
        gt_line,
        connectivity=EULER_CONNECTIVITY
    )

    pred_euler = euler_number(
        pred_line,
        connectivity=EULER_CONNECTIVITY
    )

    ece = abs(
        pred_euler - gt_euler
    )

    return {
        'MAE': float(mae),
        'SSIM': float(ssim * 100.0),
        'PSNR': float(psnr),
        'CenterlinePrecision': float(
            precision * 100.0
        ),
        'CenterlineRecall': float(
            recall * 100.0
        ),
        'clDice': float(
            cldice * 100.0
        ),
        'ECE': float(ece),
    }


def mean_std(
    values: np.ndarray
) -> str:
    return (
        f'{values.mean():.2f} '
        f'+/- {values.std(ddof=0):.2f}'
    )


def main() -> None:
    gt_cases = find_cases(GT_DIR)
    pred_cases = find_cases(PRED_DIR)

    uids = sorted(
        gt_cases.keys()
        & pred_cases.keys()
    )

    if not uids:
        raise RuntimeError(
            'No matching case names were found '
            'in GT_DIR and PRED_DIR'
        )

    rows = []

    for uid in tqdm(
        uids,
        desc='Evaluating',
        unit='case'
    ):
        gt = load_volume(
            gt_cases[uid]
        )

        pred = load_volume(
            pred_cases[uid]
        )

        metrics = calculate_metrics(
            gt,
            pred
        )

        rows.append(metrics)

        print(
            f'\n{uid} | '
            f'MAE={metrics["MAE"]:.2f} | '
            f'SSIM={metrics["SSIM"]:.2f} | '
            f'PSNR={metrics["PSNR"]:.2f} | '
            f'Precision='
            f'{metrics["CenterlinePrecision"]:.2f} | '
            f'Recall='
            f'{metrics["CenterlineRecall"]:.2f} | '
            f'clDice={metrics["clDice"]:.2f} | '
            f'ECE={metrics["ECE"]:.0f}'
        )

    print('\nMean metrics:')

    metric_names = (
        'MAE',
        'SSIM',
        'PSNR',
        'CenterlinePrecision',
        'CenterlineRecall',
        'clDice',
        'ECE'
    )

    for key in metric_names:
        values = np.asarray(
            [row[key] for row in rows],
            dtype=np.float64
        )

        print(
            f'{key}: {mean_std(values)}'
        )


if __name__ == '__main__':
    main()