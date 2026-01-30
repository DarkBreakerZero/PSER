import os, json, random
from typing import Tuple, List, Optional, Dict, Tuple as Tup
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import scipy.ndimage as ndimage

# ===== path =====
ROOT = '../Data'
LABEL = os.path.join(ROOT, 'label')
INPUT = os.path.join(ROOT, 'input')
SPLITS_JSON = os.path.join(ROOT, 'splits.json')

# ===== 窗口配置 =====
L_WIN = 200.0
H_WIN = 3000.0


def win01_np(x: np.ndarray, L: float = L_WIN, H: float = H_WIN) -> np.ndarray:
    """固定窗口 [L,H] -> [0,1]"""
    y = np.clip(x.astype(np.float32, copy=False), L, H)
    y = (y - L) / (H - L + 1e-6)
    return y


def win_denorm(x: np.ndarray, L: float = L_WIN, H: float = H_WIN) -> np.ndarray:
    """把 [0,1] 反归一化回原窗口 [L,H]"""
    return x * (H - L) + L


def _resolve(uid: str) -> Tup[str, str]:
    for ext in ('.nii.gz', '.nii'):
        lp = os.path.join(LABEL, uid + ext)
        ip = os.path.join(INPUT, uid + ext)
        if os.path.exists(lp) and os.path.exists(ip):
            return lp, ip
    raise FileNotFoundError(f'{uid}.nii(.gz) not found')


def _load_train_uids() -> List[str]:
    if not os.path.isfile(SPLITS_JSON):
        raise FileNotFoundError(f'缺少 {SPLITS_JSON}')
    with open(SPLITS_JSON, 'r', encoding='utf-8') as f:
        s = json.load(f)
    u = s.get('train', [])
    if not isinstance(u, list) or len(u) == 0:
        raise ValueError('splits.json 中 "train" 列表为空')
    return u


# (★) FIX 1: 修复版弹性变形 (坐标系正确，参数增强)
def elastic_transform_3d(image: np.ndarray, label: np.ndarray, alpha=500, sigma=10, random_state=None):
    """
    弹性变形：模拟血管的自然扭曲
    alpha: 变形强度 (建议 500-800)
    sigma: 平滑程度 (建议 10-15)
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    # 生成平滑的随机位移场
    dx = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    # (★) 关键修正: 使用 meshgrid(indexing='ij') 确保维度顺序正确
    z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = (z + dz, y + dy, x + dx)

    # 插值应用 (Image: Linear, Label: Nearest)
    distorted_image = ndimage.map_coordinates(image, indices, order=1, mode='reflect')
    distorted_label = ndimage.map_coordinates(label, indices, order=0, mode='nearest').astype(label.dtype)

    return distorted_image, distorted_label


def random_scale_3d(image: np.ndarray, label: np.ndarray, scale_range=(0.85, 1.15)):
    """
    随机缩放：模拟不同粗细的血管，并保持图像居中
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])

    if abs(scale - 1.0) < 1e-3:
        return image, label

    zoomed_img = ndimage.zoom(image, scale, order=1, mode='reflect')
    zoomed_lbl = ndimage.zoom(label, scale, order=0, mode='nearest')

    orig_shape = image.shape
    new_shape = zoomed_img.shape

    scaled_img = np.zeros(orig_shape, dtype=image.dtype)
    scaled_lbl = np.zeros(orig_shape, dtype=label.dtype)

    start_z = (new_shape[0] - orig_shape[0]) // 2
    start_y = (new_shape[1] - orig_shape[1]) // 2
    start_x = (new_shape[2] - orig_shape[2]) // 2

    sz0 = max(0, start_z);
    sz1 = min(new_shape[0], start_z + orig_shape[0])
    sy0 = max(0, start_y);
    sy1 = min(new_shape[1], start_y + orig_shape[1])
    sx0 = max(0, start_x);
    sx1 = min(new_shape[2], start_x + orig_shape[2])

    tz0 = max(0, -start_z);
    tz1 = min(orig_shape[0], new_shape[0] - start_z)
    ty0 = max(0, -start_y);
    ty1 = min(orig_shape[1], new_shape[1] - start_y)
    tx0 = max(0, -start_x);
    tx1 = min(orig_shape[2], new_shape[2] - start_x)

    if (sz1 > sz0) and (sy1 > sy0) and (sx1 > sx0) and \
            (tz1 > tz0) and (ty1 > ty0) and (tx1 > tx0):
        scaled_img[tz0:tz1, ty0:ty1, tx0:tx1] = zoomed_img[sz0:sz1, sy0:sy1, sx0:sx1]
        scaled_lbl[tz0:tz1, ty0:ty1, tx0:tx1] = zoomed_lbl[sz0:sz1, sy0:sy1, sx0:sx1]

    return scaled_img, scaled_lbl


class Random3DPatchDataset(Dataset):
    def __init__(
            self,
            patch_size: Tuple[int, int, int] = (64, 128, 128),
            roi_bias: float = 0.7,
            attempts: int = 16,
            patches_per_vol: int = 12,
            cache_vols: int = 32,
    ):
        super().__init__()
        self.uids: List[str] = _load_train_uids()
        self.pz, self.py, self.px = map(int, patch_size)
        self.roi_bias = float(roi_bias)
        self.attempts = int(attempts)
        self.ppv = max(1, int(patches_per_vol))
        self.cache_vols = max(0, int(cache_vols))

        self.cache: Dict[str, np.ndarray] = {}
        self.cache_keys: List[str] = []
        self.roi_lin_cache: Dict[str, Tup[np.ndarray, Tup[int, int, int]]] = {}

    def __len__(self) -> int:
        return len(self.uids) * self.ppv

    def _cache_put(self, key: str, arr: np.ndarray):
        if self.cache_vols <= 0: return
        if key in self.cache: return
        self.cache[key] = arr
        self.cache_keys.append(key)
        if len(self.cache_keys) > self.cache_vols:
            old = self.cache_keys.pop(0)
            self.cache.pop(old, None)
            self.roi_lin_cache.pop(old, None)

    def _load_zyx(self, path: str) -> np.ndarray:
        if path in self.cache:
            return self.cache[path]

        # 加载数据并确保是 float32
        arr = nib.load(path).get_fdata(dtype=np.float32)
        arr = np.transpose(arr, (2, 1, 0)).copy()  # (X,Y,Z) -> (Z,Y,X)
        self._cache_put(path, arr)
        return arr

    def _get_roi_lin(self, path: str, arr_zyx: np.ndarray):
        if path in self.roi_lin_cache:
            return self.roi_lin_cache[path]

        roi = (arr_zyx > L_WIN)
        lin = np.flatnonzero(roi.ravel(order='C'))
        out = (lin.astype(np.int64, copy=False), arr_zyx.shape)
        self.roi_lin_cache[path] = out
        return out

    def _choose_window(self, Z: int, Y: int, X: int, center=None):
        pz, py, px = self.pz, self.py, self.px
        if center is not None:
            zc, yc, xc = center
            z0 = int(np.clip(zc - pz // 2, 0, max(0, Z - pz)))
            y0 = int(np.clip(yc - py // 2, 0, max(0, Y - py)))
            x0 = int(np.clip(xc - px // 2, 0, max(0, X - px)))
            return z0, y0, x0
        z0 = 0 if pz >= Z else np.random.randint(0, Z - pz + 1)
        y0 = 0 if py >= Y else np.random.randint(0, Y - py + 1)
        x0 = 0 if px >= X else np.random.randint(0, X - px + 1)
        return z0, y0, x0

    @staticmethod
    def _center_crop_to(arr: np.ndarray, tz: int, ty: int, tx: int):
        z, y, x = arr.shape
        if (z, y, x) == (tz, ty, tx): return arr
        cz0 = max(0, (z - tz) // 2);
        cz1 = min(z, cz0 + tz)
        cy0 = max(0, (y - ty) // 2);
        cy1 = min(y, cy0 + ty)
        cx0 = max(0, (x - tx) // 2);
        cx1 = min(x, cx0 + tx)
        return arr[cz0:cz1, cy0:cy1, cx0:cx1]

    def __getitem__(self, index: int):
        uid = self.uids[index % len(self.uids)]
        lp, ip = _resolve(uid)
        lbl = self._load_zyx(lp)
        inp = self._load_zyx(ip)

        Z, Y, X = lbl.shape
        pz, py, px = self.pz, self.py, self.px

        center = None
        if random.random() < self.roi_bias:
            lin, shape = self._get_roi_lin(lp, lbl)
            if lin.size > 0:
                lin_idx = int(lin[np.random.randint(lin.size)])
                center = np.unravel_index(lin_idx, shape, order='C')

        ok = False
        for _ in range(self.attempts):
            z0, y0, x0 = self._choose_window(Z, Y, X, center)
            z1, y1, x1 = z0 + pz, y0 + py, x0 + px
            if z1 <= Z and y1 <= Y and x1 <= X:
                ok = True
                break
            center = None

        if not ok:
            z0 = 0 if pz >= Z else np.random.randint(0, Z - pz + 1)
            y0 = 0 if py >= Y else np.random.randint(0, Y - py + 1)
            x0 = 0 if px >= X else np.random.randint(0, X - px + 1)
            z1, y1, x1 = min(Z, z0 + pz), min(Y, y0 + py), min(X, x0 + px)

        lbl_p = lbl[z0:z1, y0:y1, x0:x1]
        inp_p = inp[z0:z1, y0:y1, x0:x1]
        lbl_p = win01_np(lbl_p)
        inp_p = win01_np(inp_p)

        if random.random() < 0.5:
            lbl_p = np.ascontiguousarray(np.flip(lbl_p, axis=0))
            inp_p = np.ascontiguousarray(np.flip(inp_p, axis=0))
        if random.random() < 0.5:
            lbl_p = np.ascontiguousarray(np.flip(lbl_p, axis=1))
            inp_p = np.ascontiguousarray(np.flip(inp_p, axis=1))
        if random.random() < 0.5:
            lbl_p = np.ascontiguousarray(np.flip(lbl_p, axis=2))
            inp_p = np.ascontiguousarray(np.flip(inp_p, axis=2))

        k = random.randint(0, 3)
        if k > 0:
            lbl_p = np.ascontiguousarray(np.rot90(lbl_p, k=k, axes=(1, 2)))
            inp_p = np.ascontiguousarray(np.rot90(inp_p, k=k, axes=(1, 2)))

        if random.random() < 0.3:
            try:
                inp_p, lbl_p = random_scale_3d(inp_p, lbl_p, scale_range=(0.8, 1.25))
            except:
                pass
        if random.random() < 0.3:
            inp_p, lbl_p = elastic_transform_3d(inp_p, lbl_p, alpha=500, sigma=10)

        lbl_p = self._center_crop_to(lbl_p, pz, py, px)
        inp_p = self._center_crop_to(inp_p, pz, py, px)

        return {
            "label": torch.from_numpy(lbl_p[None, ...].copy()),
            "input": torch.from_numpy(inp_p[None, ...].copy()),
        }