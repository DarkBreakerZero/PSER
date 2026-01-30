# -*- coding: utf-8 -*-
import os
import numpy as np
import time
import nibabel as nib
import json
import random
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ---------------- 1. Basic Path Configuration ----------------
root_dir = './Data/'
label_dir = os.path.join(root_dir, 'label')
stage1_dir = os.path.join(root_dir, 'Stage1Results')

# Output directories (keeping original names)
TrainNPZSaveDir = os.path.join(root_dir, 'TrainSlice2D/')
ValidNPZSaveDir = os.path.join(root_dir, 'ValidSlice2D/')
TRAIN_TXT = './txt/train_2d_img_list.txt'
VALID_TXT = './txt/valid_2d_img_list.txt'


# Native replacement for pytools.make_dirs
def check_and_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


check_and_make_dir(stage1_dir)
check_and_make_dir(TrainNPZSaveDir)
check_and_make_dir(ValidNPZSaveDir)
check_and_make_dir('./txt/')

# Clear list files
open(TRAIN_TXT, 'w').close()
open(VALID_TXT, 'w').close()

# ---------------- 2. JSON Reading Logic ----------------
SPLITS_JSON = os.path.join(root_dir, 'splits.json')


def _resolve_filename(uid: str) -> str:
    for ext in ('.nii.gz', '.nii'):
        p = os.path.join(label_dir, uid + ext)
        if os.path.exists(p): return uid + ext
    raise FileNotFoundError(f'Label not found for: {uid}')


def _uid_from_name(name: str) -> str:
    return name[:-7] if name.endswith('.nii.gz') else (name[:-4] if name.endswith('.nii') else name)


def _load_split(which: str):
    with open(SPLITS_JSON, 'r', encoding='utf-8') as f:
        s = json.load(f)
    names = [_resolve_filename(uid) for uid in s[which]]
    return sorted(names)


train_list = _load_split('train')
validation_list = _load_split('val')

print(f'[Info] Train: {len(train_list)}, Val: {len(validation_list)}')


# ---------------- 3. Core Logic: Extract 2.5D Stack ----------------
def get_stack_slice(volume_zyx, z_idx, max_z):
    # Clamp indices to prevent out-of-bound access
    prev_idx = max(0, z_idx - 1)
    curr_idx = z_idx
    next_idx = min(max_z - 1, z_idx + 1)

    # Extract (3, H, W) stack
    return np.stack([
        volume_zyx[prev_idx],
        volume_zyx[curr_idx],
        volume_zyx[next_idx]
    ], axis=0)


# ---------------- 4. Training Set Processing ----------------
print("Processing Train Set...")
train_lines_buffer = []  # Buffer write content to reduce I/O operations

for index, patient in enumerate(tqdm(train_list, desc='Train 2.5D', unit='case'), 1):
    # Load Label (HU)
    labelFile = nib.load(os.path.join(label_dir, patient))
    labelImgData = np.float32(labelFile.get_fdata())
    labelImgData = np.transpose(labelImgData, [2, 1, 0])  # (Z, Y, X)

    # Load Input (already in [0,1])
    uid = _uid_from_name(patient)
    npy_path = os.path.join(stage1_dir, uid + '.npy')
    inputImgData = np.load(npy_path).astype(np.float32)
    inputImgData = np.transpose(inputImgData, (2, 1, 0))  # (Z, Y, X)

    if inputImgData.shape != labelImgData.shape:
        raise ValueError(f"Shape mismatch: {uid}")

    Z_depth = labelImgData.shape[0]

    for indexZ in range(Z_depth):
        # Get 3-channel input stack
        inputStack = get_stack_slice(inputImgData, indexZ, Z_depth)
        # Get 1-channel label slice
        labelSlice = labelImgData[indexZ, :, :]

        save_stem = f'{uid}_slice{indexZ}'
        save_path = os.path.join(TrainNPZSaveDir, save_stem + '.npz')

        np.savez(save_path, input=inputStack, label=labelSlice)
        train_lines_buffer.append(save_path + '\n')

# Write training list at once (and shuffle)
with open(TRAIN_TXT, 'w') as f:
    random.shuffle(train_lines_buffer)  # Shuffle directly here
    f.writelines(train_lines_buffer)

# ---------------- 5. Validation Set Processing ----------------
print("Processing Val Set...")
val_lines_buffer = []

for index, patient in enumerate(tqdm(validation_list, desc='Val 2.5D', unit='case'), 1):
    labelFile = nib.load(os.path.join(label_dir, patient))
    labelImgData = np.float32(labelFile.get_fdata())
    labelImgData = np.transpose(labelImgData, [2, 1, 0])

    uid = _uid_from_name(patient)
    npy_path = os.path.join(stage1_dir, uid + '.npy')
    inputImgData = np.load(npy_path).astype(np.float32)
    inputImgData = np.transpose(inputImgData, (2, 1, 0))

    Z_depth = labelImgData.shape[0]

    for indexZ in range(Z_depth):
        inputStack = get_stack_slice(inputImgData, indexZ, Z_depth)
        labelSlice = labelImgData[indexZ, :, :]

        save_stem = f'{uid}_slice{indexZ}'
        save_path = os.path.join(ValidNPZSaveDir, save_stem + '.npz')

        np.savez(save_path, input=inputStack, label=labelSlice)
        val_lines_buffer.append(save_path + '\n')

with open(VALID_TXT, 'w') as f:
    f.writelines(val_lines_buffer)

print("[Done] 2.5D Preprocessing Finished.")