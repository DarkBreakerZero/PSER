# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class npzFileReader(Dataset):
    """
    只负责把 .npz 读成张量，不做 (a,b) 归一化。
    - 支持 2D: (H,W) -> (1,H,W)
    - 支持 3D: (D,H,W) -> (1,D,H,W)
    - 不修改原始强度；是否把负值清零由 clamp_neg 控制（默认 False）
    """
    def __init__(self, paired_data_txt: str, clamp_neg: bool = False):
        super().__init__()
        if not os.path.isfile(paired_data_txt):
            raise FileNotFoundError(f'缺少列表文件: {paired_data_txt}')
        with open(paired_data_txt, 'r', encoding='utf-8') as f:
            self.paired_files = [ln.strip() for ln in f if ln.strip()]
        self.clamp_neg = bool(clamp_neg)

    def __len__(self):
        return len(self.paired_files)

    @staticmethod
    def _to_tensor(arr: np.ndarray) -> torch.Tensor:
        # 2D -> (1,H,W) ; 3D -> (1,D,H,W)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        elif arr.ndim == 3:
            arr = arr[np.newaxis, ...]
        else:
            raise ValueError(f'仅支持 2D/3D，收到形状: {arr.shape}')
        return torch.from_numpy(arr.astype(np.float32, copy=False))

    def __getitem__(self, index: int):
        path = self.paired_files[index]
        if not os.path.isfile(path):
            raise FileNotFoundError(f'样本不存在: {path}')

        # 只读、禁止pickle更安全
        d = np.load(path, allow_pickle=False, mmap_mode='r')

        if 'label' not in d or 'input' not in d:
            raise KeyError(f'.npz 缺少 "label"/"input" 键: {path}')

        img   = np.asarray(d['label'], dtype=np.float32)
        scout = np.asarray(d['input'], dtype=np.float32)

        if self.clamp_neg:
            # 可选：清负值（与外层 clip(0,T) 等价下限，默认不做，避免重复）
            np.maximum(img,   0.0, out=img)
            np.maximum(scout, 0.0, out=scout)

        return {
            "label": self._to_tensor(img),
            "input": self._to_tensor(scout),
        }
