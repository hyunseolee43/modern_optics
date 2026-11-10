"""Compare measured 3D point-spread functions with previous Fourier-optics simulations.

Install on your Windows computer (if needed):
    pip install numpy scipy matplotlib tifffile

Run:
    python compare_experimental_simulated_psf.py

If only the ZIP is available, this also works:
    python compare_experimental_simulated_psf.py --zip "cropped_center block aperture 2.zip"

The two figures and a CSV summary are saved to Lab 3/PSF_comparison/.
"""
from __future__ import annotations

import argparse
import csv
import io
import re
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates

# ------------------- User settings -------------------
ROOT = Path(r"C:\Users\zhang\PycharmProjects\PythonProject\modern_optics\Lab 3")
DATA_DIR = ROOT / "Data"
CAMERA_PIXEL_UM = 5.3
MAGNIFICATION = 5.0
STAGE_Z_STEP_UM = 1.0
EXPOSURES_PER_Z = 3

# All three 'cropped_N.tif' files are separate 3D stacks for the same aperture.
# Each stack contains approximately three exposures at each z stage location.
APERTURES = {
    1: ("No aperture", "cropped_no aperture"),
    3: ("Horizontal slit", "cropped_center horizontal aperture"),
    4: ("Central obstruction", "cropped_center block aperture 2"),
}

# If individual images were excluded from the middle of a cropped TIFF stack,
# their original frame indices must be provided to recover exact z positions.
# Frame indices are zero-based within the ORIGINAL un-cropped acquisition.
# Example: {(1, 2): [7, 8]} means frames 7 and 8 were deleted from
# aperture 1, cropped_2.tif. Do not add guessed missing indices.
MISSING_ORIGINAL_FRAMES: dict[tuple[int, int], list[int]] = {}
# If the first retained frame was not original frame 0, set its index here.
FIRST_ORIGINAL_FRAME: dict[tuple[int, int], int] = {}
# Alternatively, provide the original index for every retained TIFF page.
# Example: {(3, 1): [0, 1, 2, 3, 5, 6, ...]}.
EXPLICIT_ORIGINAL_FRAME_INDICES: dict[tuple[int, int], list[int]] = {}

# Model NPZ coordinates were generated for a 100 mm imaging lens in the
# CAMERA/image plane. Convert to object/stage coordinates for 5x magnification:
# x_obj = x_image / M; z_obj = z_image / M^2 (paraxial conjugate relation).
# This assumes the simulation's pupil approximates the actual imaging pupil.
MODEL_XY_SCALE = 1.0 / MAGNIFICATION
MODEL_Z_SCALE = 1.0 / MAGNIFICATION**2

# Optional orientation corrections if the microscope flips/mirrors the image.
FLIP_EXPERIMENT_X = False
FLIP_EXPERIMENT_Y = False
ROTATE_EXPERIMENT_QUARTER_TURNS = 0

XY_HALF_WIDTH_UM = 24.0
XZ_HALF_HEIGHT_UM = 5.0  # ±150 um image-side model defocus -> ±6 um object-side
PLOT_XY_STEP_UM = CAMERA_PIXEL_UM / MAGNIFICATION
PLOT_Z_STEP_UM = 0.25  # interpolated for prettier graphics; measured steps are 1 um
LOG_FLOOR = 1e-3
# -----------------------------------------------------


def read_stack(path: Path, archive: ZipFile | None, zip_lookup: dict | None):
    if path.exists():
        data = tifffile.imread(path)
    else:
        if archive is None:
            raise FileNotFoundError(f"Not found: {path}")
        key = (path.parent.name.lower(), path.name.lower())
        member = zip_lookup.get(key)
        if member is None:
            raise FileNotFoundError(f"Not in extracted data or ZIP: {path}")
        data = tifffile.imread(io.BytesIO(archive.read(member)))
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected a multi-page TIFF with axes (frame, y, x): {path}: {data.shape}")
    return data


def frame_to_stage_group_indices(n: int, aperture: int, replicate: int):
    key = (aperture, replicate)
    if key in EXPLICIT_ORIGINAL_FRAME_INDICES:
        original = np.asarray(EXPLICIT_ORIGINAL_FRAME_INDICES[key], dtype=int)
        if original.size != n:
            raise ValueError(f"{key}: explicit frame list length {original.size}, but TIFF has {n} pages")
    else:
        original_start = FIRST_ORIGINAL_FRAME.get(key, 0)
        missing = set(MISSING_ORIGINAL_FRAMES.get(key, []))
        original = []
        i = original_start
        while len(original) < n:
            if i not in missing:
                original.append(i)
            i += 1
        original = np.asarray(original, dtype=int)
    if np.any(np.diff(original) <= 0):
        raise ValueError(f"Frame numbers for {key} must be strictly increasing")
    return original // EXPOSURES_PER_Z


def border_background_subtract(stack):
    # Subtract a robust per-frame background to account for offsets/exposure drift.
    n, h, w = stack.shape
    edge = max(1, min(3, h // 8, w // 8))
    borders = np.concatenate(
        [stack[:, :edge].reshape(n, -1), stack[:, -edge:].reshape(n, -1),
         stack[:, :, :edge].reshape(n, -1), stack[:, :, -edge:].reshape(n, -1)],
        axis=1,
    )
    bg = np.median(borders, axis=1)
    return np.maximum(stack - bg[:, None, None], 0)


def group_by_z(stack, group_ids):
    unique = np.unique(group_ids)
    planes = np.stack([np.mean(stack[group_ids == g], axis=0) for g in unique])
    # Keep z indices, allowing gaps from missing frames.
    return unique.astype(float), planes


def locate_focus_and_center(z_indices, volume):
    # Use a smoothed peak to find focus; robust to a bright single-pixel noise spike.
    blurred = gaussian_filter(volume, sigma=(0, 0.8, 0.8))
    brightness = blurred.max(axis=(1, 2))
    smooth_brightness = gaussian_filter1d(brightness, sigma=0.65)
    highest = np.max(smooth_brightness)
    near_focus = smooth_brightness >= 0.90 * highest
    # Center of the near-peak plateau gives sub-stage-position precision.
    focus_idx = np.average(z_indices[near_focus], weights=smooth_brightness[near_focus] ** 2)
    near = np.abs(z_indices - focus_idx) <= 1.5
    focal_image = blurred[near].mean(axis=0) if np.any(near) else blurred[np.argmax(smooth_brightness)]
    weights = np.maximum(focal_image - 0.20 * focal_image.max(), 0) ** 2
    yy, xx = np.indices(focal_image.shape)
    if weights.sum() == 0:
        cy, cx = np.unravel_index(np.argmax(focal_image), focal_image.shape)
    else:
        cy = (yy * weights).sum() / weights.sum()
        cx = (xx * weights).sum() / weights.sum()
    return float(focus_idx), float(cy), float(cx)


def resample_stack(volume, z_indices, focus_z, cy, cx, xy_um, z_um):
    # Map from physical target coordinates to the measured pixel/z index grid.
    # Missing z positions are interpolated; outside original crop stays NaN.
    dz = float(STAGE_Z_STEP_UM)
    step = CAMERA_PIXEL_UM / MAGNIFICATION
    zz, yy, xx = np.meshgrid(z_um, xy_um, xy_um, indexing='ij')
    # z_indices usually begin at 0 but may have gaps; use np.interp for real index.
    wanted_stage = zz / dz + focus_z
    real_z_index = np.interp(wanted_stage, z_indices, np.arange(len(z_indices)), left=np.nan, right=np.nan)
    tgt_y = cy + yy / step
    tgt_x = cx + xx / step
    valid = (np.isfinite(real_z_index) & (tgt_y >= 0) & (tgt_y <= volume.shape[1]-1)
             & (tgt_x >= 0) & (tgt_x <= volume.shape[2]-1))
    samples = map_coordinates(
        volume, [np.nan_to_num(real_z_index), tgt_y, tgt_x], order=1, mode='constant', cval=0
    )
    samples[~valid] = np.nan
    return samples


def experimental_psf(tiff, aperture, replicate, archive, lookup, xy_um, z_um):
    raw = read_stack(tiff, archive, lookup)
    raw = border_background_subtract(raw)
    if FLIP_EXPERIMENT_X:
        raw = raw[:, :, ::-1]
    if FLIP_EXPERIMENT_Y:
        raw = raw[:, ::-1, :]
    if ROTATE_EXPERIMENT_QUARTER_TURNS:
        raw = np.rot90(raw, ROTATE_EXPERIMENT_QUARTER_TURNS, axes=(1, 2))
    groups = frame_to_stage_group_indices(raw.shape[0], aperture, replicate)
    stage_indices, vol = group_by_z(raw, groups)
    focus_z, cy, cx = locate_focus_and_center(stage_indices, vol)
    norm = max(np.nanmax(vol), 1e-12)
    vol = vol / norm
    sampled = resample_stack(vol, stage_indices, focus_z, cy, cx, xy_um, z_um)
    print(f"  cropped_{replicate}: {raw.shape[0]} frames => {len(stage_indices)} z positions; "
          f"focus at grouped z={focus_z:.2f}; center (y,x)=({cy:.1f},{cx:.1f}) px")
    return sampled


def simulation_psf(npz_file, xy_um, z_um):
    with np.load(npz_file) as data:
        xy0 = np.asarray(data['camera_pixel_integrated_psf'], dtype=float)
        xz0 = np.asarray(data['camera_pixel_integrated_psf_xz'], dtype=float)
        model_x = np.asarray(data['image_um']) * MODEL_XY_SCALE
        model_z = np.asarray(data['z_um']) * MODEL_Z_SCALE

    # Map the same object-side coordinates as the experimental stacks.
    zz_xy, yy_xy, xx_xy = np.meshgrid(np.array([0]), xy_um, xy_um, indexing='ij')
    x_ind = (xx_xy[0] - model_x[0]) / (model_x[1] - model_x[0])
    y_ind = (yy_xy[0] - model_x[0]) / (model_x[1] - model_x[0])
    sim_xy = map_coordinates(xy0, [y_ind, x_ind], order=1, mode='constant', cval=np.nan)

    zz, xx = np.meshgrid(z_um, xy_um, indexing='ij')
    z_ind = (zz - model_z[0]) / (model_z[1] - model_z[0])
    x_ind = (xx - model_x[0]) / (model_x[1] - model_x[0])
    sim_xz = map_coordinates(xz0, [z_ind, x_ind], order=1, mode='constant', cval=np.nan)
    denom = max(float(np.nanmax(sim_xy)), 1e-12)
    return sim_xy / denom, sim_xz / denom


def fwhm(axis, values):
    x, y = np.asarray(axis), np.asarray(values, dtype=float)
    valid = np.isfinite(y)
    if not np.any(valid):
        return np.nan
    y = np.where(valid, y, 0)
    y = gaussian_filter1d(y, 0.6)
    center = np.argmax(y)
    level = y[center] / 2
    i = center
    while i > 0 and y[i] >= level:
        i -= 1
    if i == 0 and y[i] >= level:
        return np.nan
    left = np.interp(level, [y[i], y[i+1]], [x[i], x[i+1]])
    i = center
    while i < len(y) - 1 and y[i] >= level:
        i += 1
    if i == len(y)-1 and y[i] >= level:
        return np.nan
    right = np.interp(level, [y[i], y[i-1]], [x[i], x[i-1]])
    return float(right - left)


def plot_comparisons(results, xy_um, z_um, dest):
    # Figure 1: directly matched PSF slices, simulated vs measured.
    fig, ax = plt.subplots(3, 4, figsize=(15.2, 11.2), layout='constrained')
    headings = ["Simulation: xy at z=0", "Measurement: xy at z=0",
                "Simulation: xz at y=0", "Measurement: xz at y=0"]
    for col, heading in enumerate(headings):
        ax[0, col].set_title(heading, fontsize=12, weight='bold')
    cmap = plt.get_cmap('inferno').copy()
    cmap.set_bad('#f1f2f5')
    for row, (k, data) in enumerate(results.items()):
        sxy, sxz = data['sim_xy'], data['sim_xz']
        exy, exz = data['exp_xy'], data['exp_xz']
        for col, image in enumerate([sxy, exy, sxz, exz]):
            axes = ax[row, col]
            is_xy = col < 2
            extent = [xy_um[0], xy_um[-1], xy_um[0], xy_um[-1]] if is_xy else [xy_um[0], xy_um[-1], z_um[0], z_um[-1]]
            axes.imshow(np.log10(np.maximum(image, LOG_FLOOR)), origin='lower',
                        extent=extent, aspect='equal' if is_xy else 'auto',
                        cmap=cmap, vmin=np.log10(LOG_FLOOR), vmax=0)
            axes.set_xlabel('x (µm, object plane)')
            axes.set_ylabel('y (µm)' if is_xy else 'z (µm, stage)')
            if col == 0:
                axes.text(-0.25, .5, f"Aperture {k}\n{data['name']}", transform=axes.transAxes,
                          rotation=90, va='center', ha='center', fontsize=11, weight='bold')
    fig.suptitle('Pixel-integrated PSF: numerical simulation vs measured 3D stacks\n'
                 'Each panel individually normalized to in-focus peak; displayed as log10 intensity', fontsize=13)
    fig.savefig(dest / 'PSF_xy_xz_simulation_vs_experiment.png', dpi=220, bbox_inches='tight')
    plt.close(fig)

    # Figure 2: 1D line profiles in x/y (xy focus) and z (center).
    fig, axes = plt.subplots(3, 3, figsize=(14.7, 10.5), layout='constrained')
    headers = ["At focus: horizontal x-profile", "At focus: vertical y-profile", "On-axis z-profile"]
    for j, h in enumerate(headers):
        axes[0, j].set_title(h, fontsize=12, weight='bold')
    c = len(xy_um) // 2
    zc = len(z_um) // 2
    csv_rows = []
    for row, (k, data) in enumerate(results.items()):
        sxy, sxz, exy, exz = data['sim_xy'], data['sim_xz'], data['exp_xy'], data['exp_xz']
        sim_profiles = [sxy[c, :], sxy[:, c], sxz[:, c]]
        exp_profiles = [exy[c, :], exy[:, c], exz[:, c]]
        coords = [xy_um, xy_um, z_um]
        for col, a in enumerate(axes[row]):
            x = coords[col]
            s = sim_profiles[col]
            e = exp_profiles[col]
            a.plot(x, s, '--', color='black', label='Simulation', linewidth=1.8)
            a.plot(x, e, '-', color='#126eb1', label='Experiment (mean of 3)', linewidth=2)
            for trial in data['trials']:
                trace = (trial[zc, c, :] if col == 0 else
                         trial[zc, :, c] if col == 1 else trial[:, c, c])
                a.plot(x, trace, color='#126eb1', alpha=0.18, linewidth=1)
            a.set_xlim(x[0], x[-1])
            a.set_ylim(-0.04, 1.12)
            a.grid(alpha=0.18)
            a.set_xlabel('x (µm)' if col == 0 else ('y (µm)' if col == 1 else 'z (µm)'))
            a.set_ylabel('Relative intensity')
            if row == 0 and col == 0:
                a.legend(loc='upper right', fontsize=8)
            if col == 0:
                a.text(-.30, .5, f"Aperture {k}\n{data['name']}", transform=a.transAxes,
                       rotation=90, va='center', ha='center', fontsize=11, weight='bold')
            csv_rows.append([k, data['name'], ['x', 'y', 'z'][col],
                             fwhm(x, s), fwhm(x, e)])
    fig.suptitle('PSF line profiles: dashed = simulation; solid = average measured; faint = individual stacks', fontsize=13)
    fig.savefig(dest / 'PSF_line_profiles.png', dpi=220, bbox_inches='tight')
    plt.close(fig)

    with (dest / 'PSF_FWHM_summary.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['aperture', 'name', 'axis', 'sim_fwhm_um', 'measured_fwhm_um'])
        writer.writerows(csv_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='Lab 3 directory containing NPZ files and Data/')
    parser.add_argument('--zip', type=Path, default=None, help='Optional ZIP with the 9 cropped TIFF stacks')
    args = parser.parse_args()
    root = args.root
    out_dir = root / 'PSF_comparison'
    out_dir.mkdir(parents=True, exist_ok=True)

    xy_um = np.arange(-int(XY_HALF_WIDTH_UM / PLOT_XY_STEP_UM),
                       int(XY_HALF_WIDTH_UM / PLOT_XY_STEP_UM) + 1) * PLOT_XY_STEP_UM
    z_um = np.arange(-round(XZ_HALF_HEIGHT_UM / PLOT_Z_STEP_UM),
                      round(XZ_HALF_HEIGHT_UM / PLOT_Z_STEP_UM) + 1) * PLOT_Z_STEP_UM
    archive = ZipFile(args.zip) if args.zip is not None else None
    lookup = None
    if archive is not None:
        lookup = {}
        for name in archive.namelist():
            p = Path(name)
            if name.lower().endswith(('.tif', '.tiff')):
                lookup[(p.parent.name.lower(), p.name.lower())] = name
    results = {}
    try:
        for k, (name, folder) in APERTURES.items():
            npz = root / f'aperture_{k}_results.npz'
            if not npz.exists():
                raise FileNotFoundError(f'Simulated model not found: {npz}')
            sim_xy, sim_xz = simulation_psf(npz, xy_um, z_um)
            print(f'Aperture {k}: {name}')
            trials = []
            for trial in (1, 2, 3):
                path = root / 'Data' / folder / f'cropped_{trial}.tif'
                sample = experimental_psf(path, k, trial, archive, lookup, xy_um, z_um)
                trials.append(sample)
            count = np.sum(np.isfinite(np.stack(trials)), axis=0)
            accum = np.nansum(np.stack(trials), axis=0)
            exp = np.divide(accum, count, out=np.full_like(accum, np.nan), where=count > 0)
            focus_idx = len(z_um) // 2
            # The same factor normalizes in-focus xy and the xz slice.
            scale = max(float(np.nanmax(exp[focus_idx])), 1e-12)
            exp /= scale
            trials = [trial / scale for trial in trials]
            results[k] = dict(name=name, sim_xy=sim_xy, sim_xz=sim_xz,
                              exp_xy=exp[focus_idx], exp_xz=exp[:, len(xy_um)//2, :],
                              trials=trials)
        plot_comparisons(results, xy_um, z_um, out_dir)
        print(f'\nSaved figures and FWHM CSV to: {out_dir}')
        print('IMPORTANT: When cropped TIFF stacks omitted frames internally, edit the')
        print('MISSING_ORIGINAL_FRAMES / FIRST_ORIGINAL_FRAME settings for exact z mapping.')
    finally:
        if archive is not None:
            archive.close()


if __name__ == '__main__':
    main()
