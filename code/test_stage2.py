import os
import json
import time
import argparse
import torch
import numpy as np
import nibabel as nib
from net.fine_net import SnakeDenseUnet2d

def _ckpt_path(ckpt_dir, epoch):
    return os.path.join(ckpt_dir, epoch)


def _resolve_filename(uid: str, label_dir: str) -> str:
    for ext in ('.nii.gz', '.nii'):
        p = os.path.join(label_dir, uid + ext)
        if os.path.exists(p): return uid + ext
    raise FileNotFoundError(f'Label not found: {uid} in {label_dir}')


def _uid_from_name(name: str) -> str:
    return name[:-7] if name.endswith('.nii.gz') else name[:-4]


if __name__ == '__main__':
    # =================================================================
    # Argument Parsing
    # =================================================================
    parser = argparse.ArgumentParser(description="Test 2D Model (Stage-2)")

    # Basic configuration
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--model_name', type=str, default='Fine', help='Experiment Name (e.g., Snake, Dense)')
    parser.add_argument('--epoch_name', type=str, default='best_psnr_model.pth', help='Checkpoint filename')
    parser.add_argument('--model_chl', type=int, default=32, help='Model channel width (must match training)')

    # Data and paths
    parser.add_argument('--root_dir', type=str, default='./Data/', help='Data root directory')
    parser.add_argument('--split', type=str, default='test', help='Dataset split (test/val)')
    parser.add_argument('--clip', action='store_true', default=True, help='Clip output to 0-1')

    args = parser.parse_args()

    # =================================================================
    # Variable assignment
    # =================================================================
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    gpu_device = 'cuda:0'  # After setting env var, code usually uses cuda:0 directly

    model_name = args.model_name
    epoch_name = args.epoch_name
    MODEL_CHL = args.model_chl
    SPLIT = args.split
    CLIP01 = args.clip

    # Path configuration
    ckpt_dir = f'./runs/{model_name}/checkpoints/'
    root_dir = args.root_dir
    label_dir = os.path.join(root_dir, 'label')
    stage1_dir = os.path.join(root_dir, 'Stage1Results')  # Coarse restoration results
    splits_json = os.path.join(root_dir, 'splits.json')
    save_dir = os.path.join(root_dir, 'Stage2Results')

    os.makedirs(save_dir, exist_ok=True)

    print(f"========================================")
    print(f"[Info] Model: {model_name} | Epoch: {epoch_name}")
    print(f"[Info] Split: {SPLIT} | GPU: {args.gpu}")
    print(f"========================================")

    # 1. Prepare data list
    if not os.path.exists(splits_json):
        raise FileNotFoundError(f"Missing {splits_json}")

    with open(splits_json, 'r', encoding='utf-8') as f:
        splits = json.load(f)
        if SPLIT not in splits:
            # Fallback: if 'test' not found in json, try 'val'
            print(f"[Warn] '{SPLIT}' not found in json, trying 'val'...")
            SPLIT = 'val'

        # Get filename list
        uids = splits[SPLIT]
        patient_list = [_resolve_filename(uid, label_dir) for uid in uids]

    print(f'[Info] Cases to process: {len(patient_list)}')

    # 2. Load model
    ckpt_path = _ckpt_path(ckpt_dir, epoch_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f'[Info] Loading Checkpoint: {ckpt_path}')

    # Initialize network (automatically uses SnakeDenseUnet2d from import)
    # in_chl=3 because we feed a 2.5D stack (z-1, z, z+1)
    model = SnakeDenseUnet2d(in_chl=3, out_chl=1, model_chl=MODEL_CHL).cuda()

    checkpoint = torch.load(ckpt_path, map_location=gpu_device)
    # Support both full checkpoint dict and pure state_dict
    state = checkpoint.get('model', checkpoint)
    model.load_state_dict(state)
    model.eval()

    # 3. Per-case inference
    for i, name in enumerate(patient_list, 1):
        uid = _uid_from_name(name)
        try:
            # Load label just to get affine matrix and header
            label_path = os.path.join(label_dir, name)
            label_nii = nib.load(label_path)
            # Stage 2 doesn't need label data, only shape and header info
            # Use label shape mainly for safe cropping/alignment
            label_shape = label_nii.shape
            # Nibabel loads as (X, Y, Z); we prefer working in (Z, Y, X)

            # Load Stage 1 coarse result
            npy_path = os.path.join(stage1_dir, uid + '.npy')
            if not os.path.exists(npy_path):
                print(f'[SKIP] Missing Stage1 input for: {uid}')
                continue

            inp_xyz = np.load(npy_path).astype(np.float32)
            # Transpose to (Z, Y, X) for easy slicing
            inp_zyx = np.transpose(inp_xyz, (2, 1, 0))

            # Simple shape alignment (prevent small mismatches between Stage1 and GT)
            Z = min(inp_zyx.shape[0], label_shape[2])
            H = min(inp_zyx.shape[1], label_shape[1])
            W = min(inp_zyx.shape[2], label_shape[0])
            inp_zyx = inp_zyx[:Z, :H, :W]

            # Ensure input is in [0,1] range (Snake network prefers normalized input)
            inp_zyx_norm = np.clip(inp_zyx, 0.0, 1.0)

            # Prepare output container
            out_zyx_norm = np.zeros((Z, H, W), dtype=np.float32)

            tic = time.time()
            with torch.no_grad():
                # Slice-by-slice inference (batch size = 1)
                # If GPU memory allows, this loop can be batched for speedup, but per-slice is safest
                for z in range(Z):
                    # (★) Core logic: build 2.5D stack [z-1, z, z+1]
                    idx_prev = max(0, z - 1)
                    idx_curr = z
                    idx_next = min(Z - 1, z + 1)

                    # Extract three slices (H, W)
                    s_prev = inp_zyx_norm[idx_prev]
                    s_curr = inp_zyx_norm[idx_curr]
                    s_next = inp_zyx_norm[idx_next]

                    # Stack → (3, H, W)
                    stack = np.stack([s_prev, s_curr, s_next], axis=0)

                    # To tensor → (1, 3, H, W)
                    tensor_in = torch.from_numpy(stack).unsqueeze(0).float().to(gpu_device)

                    # Inference
                    pred = model(tensor_in)  # → (1, 1, H, W)

                    # Extract result → (H, W)
                    sl_out = pred.squeeze().cpu().numpy()

                    if CLIP01:
                        sl_out = np.clip(sl_out, 0.0, 1.0)

                    out_zyx_norm[z] = sl_out

            # Save result
            # Transpose back to (X, Y, Z) to match NIfTI convention
            out_xyz = np.transpose(out_zyx_norm, (2, 1, 0)).astype(np.float32)

            # Preserve original label's affine and header for consistent coordinate system
            out_nii = nib.Nifti1Image(out_xyz, label_nii.affine, label_nii.header)

            save_path = os.path.join(save_dir, f'{uid}.nii.gz')  # Recommended: save as .nii.gz to save space
            nib.save(out_nii, save_path)

            print(f'[{i}/{len(patient_list)}] {uid} DONE | Time: {int(time.time() - tic)}s')

        except Exception as e:
            print(f'[FAIL] Error processing {uid}: {e}')
            import traceback
            traceback.print_exc()

    print("All Inference Finished.")