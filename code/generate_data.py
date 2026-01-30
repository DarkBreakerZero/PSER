import os, glob
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from copy import deepcopy

# ===== CTorch Dependencies =====
import CTorch.utils.geometry as geometry
from CTorch.projector.projector_interface import Projector
from CTorch.reconstructor.fbpreconstructor import FBPReconstructor as FBP

from scipy.ndimage import zoom as spzoom

# ---------------- User Configuration ----------------
SRC_DIR = './Head3DDA/'  # init Data source directory
SAVE_ROOT = './Data'  # Root directory for saving processed data
N_VIEWS_DENSE = 1000  # Number of views for the label (high dose/quality)
N_VIEWS_SPARSE = 50  # Number of views for the input (sparse view)
USE_SHORT_SCAN_200_DEG = True  # Enable 200° short scan + Parker weighting
SAVE_DEBUG_PNG = False  # Whether to save debug images

# Geometry parameters
nx, ny, nz = 512, 512, 378
dx, dy, dz = 0.3418, 0.3418, 0.5
nu, nv = 1024, 1024
du, dv = 0.4, 0.4
detType = 'flat'
SAD, SDD = [750.0], [1250.0]
xOfst, yOfst, zOfst = [0.0], [0.0], [0.0]
uOfst, vOfst = [0.0], [0.0]
phi, psai = [0.0], [0.0]
xSrc, zSrc = [0.0], [0.0]
RECON_WINDOW = "hamming"
RECON_CUTOFF = 0.95

# (New) Resampling Target Voxel Count (Set to None to disable)
# Example: (256, 256, 192). If None, resampling is skipped.
RESAMPLE_TO_SHAPE = (256, 256, 192)

# (New) Preserve FOV (Field of View)
# If True: dx, dy, dz are automatically adjusted to keep the physical size constant.
# If False: Uses the dx, dy, dz defined above regardless of shape change.
PRESERVE_FOV = True

RESAMPLE_ORDER = 1  # Linear interpolation
# ----------------------------------------------------

LABEL_DIR = os.path.join(SAVE_ROOT, 'label')
INPUT_DIR = os.path.join(SAVE_ROOT, 'input')
os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)


def make_angles(n_views: int):
    """Generate projection angles."""
    total = 200.0 / 360.0 * 2.0 * np.pi if USE_SHORT_SCAN_200_DEG else 2.0 * np.pi
    return np.arange(0, total, total / n_views, dtype=np.float32)


def build_geom(n_views: int):
    """Construct the CT geometry object."""
    viewAngles = make_angles(n_views)
    geoRecon = geometry.CircGeom3D(
        nx, ny, nz, dx, dy, dz,
        nu, nv,
        n_views, viewAngles, du, dv, detType,
        SAD, SDD,
        xOfst, yOfst, zOfst,
        uOfst, vOfst, xSrc, zSrc,
        fixed=True
    )
    return geoRecon


def load_nii(path):
    """Load NIfTI file and return data (float32) and affine."""
    img = nib.load(path)
    arr = img.get_fdata(dtype=np.float32)  # Raw intensity (not necessarily standard HU)
    return arr, img.affine


def save_nii(arr, affine, path):
    """Save array as NIfTI file."""
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), path)


def prepare_volume_like_teacher(vol_xyz):
    """
    Rearrange axes to match the specific internal geometry expectation
    (Likely matching a reference implementation or 'teacher' code).
    Transforms (X, Y, Z) -> (Z, Y, X) with flips.
    """
    v = np.transpose(vol_xyz, [1, 0, 2])  # (Y,X,Z)
    v = v[::-1, :, :]  # flip axis-0
    v = np.transpose(v, [2, 0, 1])  # (Z,Y,X)
    v = v[::-1, :, :]  # flip axis-0
    v = np.ascontiguousarray(v, dtype=np.float32)
    return v


def inverse_prepare_volume_like_teacher(vol_internal):
    """
    Inverse the transformation to get back to original (X, Y, Z).
    Ensures contiguous memory layout.
    """
    v = vol_internal[::-1, :, :]
    v = np.transpose(v, [1, 2, 0])
    v = v[::-1, :, :]
    v = np.transpose(v, [1, 0, 2])
    v = np.ascontiguousarray(v, dtype=np.float32)
    return v


def project_and_recon(vol_xyz, n_views: int):
    """Perform forward projection and FBP reconstruction."""
    vol_internal = prepare_volume_like_teacher(vol_xyz)
    # Ensure contiguous array for PyTorch
    vol_np = np.ascontiguousarray(vol_internal[None, None, ...], dtype=np.float32)
    vol_t = torch.from_numpy(vol_np).float().cuda()

    geom = build_geom(n_views)
    A = Projector(geom, 'proj', 'DD', 'forward')
    Recon = FBP(geom, 'DD', window=RECON_WINDOW, parker=USE_SHORT_SCAN_200_DEG, cutoff=RECON_CUTOFF)

    proj = A(vol_t)
    vol_rec = Recon(proj)
    rec_np = np.squeeze(vol_rec.detach().cpu().numpy()).astype(np.float32)
    return rec_np  # Returns internal ZYX format


def list_cases(src_dir):
    """List all .nii or .nii.gz files in the source directory."""
    paths = sorted(glob.glob(os.path.join(src_dir, '*.nii'))) + \
            sorted(glob.glob(os.path.join(src_dir, '*.nii.gz')))
    return paths


def make_uid(path):
    """Extract a unique identifier (filename without extension) from path."""
    base = os.path.basename(path)
    if base.endswith('.nii.gz'): return base[:-7]
    if base.endswith('.nii'):    return base[:-4]
    return os.path.splitext(base)[0]


def _voxel_sizes_from_affine(aff):
    """Estimate voxel sizes for 3 axes from the affine matrix."""
    vx = float(np.linalg.norm(aff[:3, 0]))
    vy = float(np.linalg.norm(aff[:3, 1]))
    vz = float(np.linalg.norm(aff[:3, 2]))
    return vx, vy, vz


def _resample_xyz(arr_xyz: np.ndarray, target_shape, order=1):
    """Resample volume. Input/Output are both (X, Y, Z)."""
    sx, sy, sz = arr_xyz.shape
    tx, ty, tz = target_shape
    zx, zy, zz = tx / sx, ty / sy, tz / sz
    out = spzoom(arr_xyz, (zx, zy, zz), order=order)
    return np.ascontiguousarray(out, dtype=np.float32)


def main():
    assert torch.cuda.is_available(), "GPU (CUDA) is required to run CTorch."
    cases = list_cases(SRC_DIR)
    if not cases:
        print(f'[WARN] No .nii/.nii.gz files found in {SRC_DIR}')
        return

    global nx, ny, nz, dx, dy, dz

    # Handle Global Resampling Configuration
    if RESAMPLE_TO_SHAPE is not None:
        tx, ty, tz = RESAMPLE_TO_SHAPE
        if PRESERVE_FOV:
            dx = dx * (nx / tx)
            dy = dy * (ny / ty)
            dz = dz * (nz / tz)
        # Update voxel counts
        nx, ny, nz = int(tx), int(ty), int(tz)
        print(f'[INFO] Resampling Enabled: target shape = {(nx, ny, nz)}, '
              f'voxel size = ({dx:.6f},{dy:.6f},{dz:.6f})')

    ok, failed = 0, 0
    for idx, p in enumerate(cases, 1):
        uid = make_uid(p)
        try:
            print(f'\n[{idx}/{len(cases)}] Processing: {uid}')
            vol_xyz, affine = load_nii(p)

            if SAVE_DEBUG_PNG:
                os.makedirs('./_debug', exist_ok=True)
                midz = vol_xyz.shape[2] // 2
                plt.imsave(f'./_debug/{uid}_src_mid.png', vol_xyz[:, :, midz], cmap='gray')

            # Perform Resampling if configured
            if RESAMPLE_TO_SHAPE is not None:
                ori_dx, ori_dy, ori_dz = _voxel_sizes_from_affine(affine)
                vol_xyz = _resample_xyz(vol_xyz, (nx, ny, nz), order=RESAMPLE_ORDER)

                # Update affine matrix to reflect new spacing
                new_affine = deepcopy(affine)
                if ori_dx > 0: new_affine[:3, 0] *= (dx / ori_dx)
                if ori_dy > 0: new_affine[:3, 1] *= (dy / ori_dy)
                if ori_dz > 0: new_affine[:3, 2] *= (dz / ori_dz)
                affine = new_affine

                # Generate Label (Dense Views) and Input (Sparse Views)
            label_internal = project_and_recon(vol_xyz, N_VIEWS_DENSE)
            input_internal = project_and_recon(vol_xyz, N_VIEWS_SPARSE)

            # Convert back to original orientation
            save_label = inverse_prepare_volume_like_teacher(label_internal)
            save_input = inverse_prepare_volume_like_teacher(input_internal)

            # Define output paths
            out_lab = os.path.join(LABEL_DIR, uid + '.nii.gz')
            out_inp = os.path.join(INPUT_DIR, uid + '.nii.gz')

            # Save results
            save_nii(save_label, affine, out_lab)
            save_nii(save_input, affine, out_inp)

            print(f'  [OK] label -> {out_lab}')
            print(f'  [OK] input -> {out_inp}')
            ok += 1

            torch.cuda.empty_cache()

        except Exception as e:
            print(f'  [FAIL] {uid}: {e}')
            failed += 1
            torch.cuda.empty_cache()

    print(f'\nFinished: Success {ok} / Total {len(cases)}, Failed {failed}')


if __name__ == '__main__':
    main()