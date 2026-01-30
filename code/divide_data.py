# -*- coding: utf-8 -*-
import os, glob, json, random, time

# ===== Keep your original configuration unchanged =====
ROOT = './Data'
IN_DIR = os.path.join(ROOT, 'input')
GT_DIR = os.path.join(ROOT, 'label')
OUT_JSON = os.path.join(ROOT, 'splits.json')

# Fix random seed for reproducibility
SEED = 985
random.seed(SEED)


def list_nii(folder):
    """List all .nii or .nii.gz files in the folder."""
    nii = glob.glob(os.path.join(folder, '*.nii'))
    niigz = glob.glob(os.path.join(folder, '*.nii.gz'))
    return sorted(nii + niigz)


def stem(p):
    """Extract filename stem (remove .nii or .nii.gz)."""
    b = os.path.basename(p)
    return b[:-7] if b.endswith('.nii.gz') else b[:-4]


def pick_name_map(files):
    """Map stem to full path, prioritizing .nii.gz if duplicates exist."""
    m = {}
    for p in files:
        s = stem(p)
        if s in m:
            if p.endswith('.nii.gz'):
                m[s] = p  # Prefer .nii.gz
        else:
            m[s] = p
    return m


def main():
    # Check if paths exist to prevent runtime errors
    if not os.path.exists(IN_DIR) or not os.path.exists(GT_DIR):
        print(f"[Error] Path not found: {IN_DIR} or {GT_DIR}")
        print("Please ensure you are running this from the PSER root directory, or the Data folder is correct.")
        return

    in_map = pick_name_map(list_nii(IN_DIR))
    gt_map = pick_name_map(list_nii(GT_DIR))

    # Find intersection of cases (filenames must match)
    uids = sorted(set(in_map.keys()) & set(gt_map.keys()))
    n = len(uids)

    # Simple sanity check
    if n < 3:
        print(f'[Warning] Too few paired samples found: {n}')
        # You can choose to return here if strictly required

    print(f'[INFO] Available cases: {n}')

    # Target quotas (Specifically for your 37 cases)
    target_train, target_val, target_test = 25, 6, 6

    if target_train + target_val + target_test != n:
        print(f"[Error] Target sum ({target_train + target_val + target_test}) does not match total cases ({n})")
        return

    # Shuffle and split
    random.shuffle(uids)
    train_uids = uids[:target_train]
    val_uids = uids[target_train:target_train + target_val]
    test_uids = uids[target_train + target_val:]

    print(f'[OK] Split results: train={len(train_uids)}, val={len(val_uids)}, test={len(test_uids)}')

    splits = {
        "train": train_uids,
        "val": val_uids,
        "test": test_uids,
        "meta": {
            "total": n,
            "counts": {"train": len(train_uids), "val": len(val_uids), "test": len(test_uids)},
            "seed": SEED,
            "strategy": "fixed counts for 37 cases: 25/6/6"
        }
    }

    os.makedirs(ROOT, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)

    print(f'[DONE] Split configuration saved to {OUT_JSON}')


# Entry point
if __name__ == '__main__':
    main()