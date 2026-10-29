"""Coherent Fourier-plane filtering of the Thorlabs R1L1S1N 50-um grid.

Target: 20 x 20 grid, 50 um pitch, 5 um clear-line width.
Optics: first Fourier lens f1 = 30 mm; 4 nominal pupil configurations.

Requirements: numpy, matplotlib, Pillow; tifffile optional for scientific TIFFs.
Run: python 4f_simulation_50um_grid.py

The Fourier-plane camera/observation screen intensity is |FT(E0) P|^2.
The final image intensity is |IFT(FT(E0) P)|^2. Coordinates in
final image are object-equivalent (unity magnification); if the second
lens focal length is f2, physical camera coordinates scale by |f2/f1|.
"""
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

# ----------------------------- USER SETTINGS -----------------------------
WAVELENGTH_NM = 633.0           # Assumed monochromatic illumination; EDIT if known
FOURIER_LENS_F_MM = 30.0
GRID_PITCH_UM = 50.0
GRID_LINE_WIDTH_UM = 5.0
GRID_COUNT = 20
NEGATIVE_TARGET = True           # R1L1S1N: clear grid lines in opaque chrome

N = 2048                         # 2-D simulation array, even
FOV_MM = 1.20                    # Object-equivalent full field of view
CLEAR_APERTURE_D_MM = 25.0
SLIT_LONG_MM = 20.0
SLIT_SHORT_MM = 0.5
CENTRAL_STOP_D_MM = 2.0

# The diagonal support arm in the sketch may be part of the physical stop.
# Its width is unspecified; omit it by default, or enable + measure width.
INCLUDE_STOP_SUPPORT = False
SUPPORT_WIDTH_MM = 0.3

# Optionally substitute a photo WITHOUT a Fourier filter as the reference.
# The photo gives intensity but not complex phase; sqrt(I) and zero phase
# are used as an approximation. Calibrate FOV_MM to the photo if used.
UNFILTERED_PHOTO_PATH = None      # e.g. r'C:\\data\\grid_no_filter.png'

# Display settings (per-panel normalization is ONLY for visibility).
NORMALIZE_EACH_FINAL_IMAGE = True
# The four recorded TIFFs, in the same order as the simulated apertures.
# Point this directory to the folder containing your recorded images.
MEASURED_IMAGE_DIR = Path(__file__).resolve().parent / 'data'
MEASURED_FILES = [
    'aperture1_10us.tif',
    'aperture2_30us.tif',
    'aperture3_30us.tif',
    'aperture4_330us.tif',
]
# Each camera image is independently contrast-stretched for visualization.
MEASURED_DISPLAY_PERCENTILES = (0.5, 99.5)
OUT_DIR = Path(__file__).resolve().parent / '4f_simulation_results'
# -------------------------------------------------------------------------


def generate_grid(x_mm):
    """Real transmission amplitude of the 1x1 mm, 20x20 grid."""
    pitch = GRID_PITCH_UM * 1e-3
    width = GRID_LINE_WIDTH_UM * 1e-3
    side = GRID_COUNT * pitch
    X, Y = np.meshgrid(x_mm, x_mm, indexing='xy')
    in_square = (np.abs(X) < side / 2) & (np.abs(Y) < side / 2)
    # 20 bars at x,y = +/-0.475,...,+/-0.025 mm for a 50-um pitch.
    dist_to_x_line = np.abs(np.mod(X, pitch) - pitch / 2)
    dist_to_y_line = np.abs(np.mod(Y, pitch) - pitch / 2)
    lines = in_square & ((dist_to_x_line <= width / 2) |
                         (dist_to_y_line <= width / 2))
    if NEGATIVE_TARGET:
        return lines.astype(np.float32)
    # Positive target: opaque grid lines, otherwise transparent glass.
    return (~lines).astype(np.float32)


def read_unfiltered_photo(path, size):
    """Approximate input complex field using measured intensity only."""
    from PIL import Image, ImageOps
    im = Image.open(path).convert('L')
    im = ImageOps.fit(im, (size, size), method=Image.Resampling.LANCZOS)
    intensity = np.asarray(im, dtype=np.float32) / 255.0
    return np.sqrt(np.maximum(intensity, 0.0))


def build_masks(xf_mm):
    """Make masks sampled in physical Fourier-plane millimeters."""
    xf = xf_mm[None, :]  # columns: physical x_F
    yf = xf_mm[:, None]  # rows: physical y_F
    r2 = xf**2 + yf**2
    pupil = r2 <= (CLEAR_APERTURE_D_MM / 2)**2

    # White transparent slit on opaque background, within 25-mm pupil.
    vert = (np.abs(xf) <= SLIT_SHORT_MM / 2) & (
        np.abs(yf) <= SLIT_LONG_MM / 2) & pupil
    horiz = (np.abs(yf) <= SLIT_SHORT_MM / 2) & (
        np.abs(xf) <= SLIT_LONG_MM / 2) & pupil

    stop = pupil & (r2 >= (CENTRAL_STOP_D_MM / 2)**2)
    if INCLUDE_STOP_SUPPORT:
        # Example holder runs along +45 degrees in the drawing.
        support = np.abs(yf - xf) / np.sqrt(2) <= SUPPORT_WIDTH_MM / 2
        stop = stop & ~support

    return {
        '1. 25 mm circular aperture (reference)': pupil,
        '2. Vertical slit: 20 x 0.5 mm': vert,
        '3. Horizontal slit: 20 x 0.5 mm': horiz,
        '4. 2 mm opaque central stop': stop,
    }


def rel_fft(u):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u), norm='ortho'))


def rel_ifft(u):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(u), norm='ortho'))


def load_measured_tiff(path):
    """Load 8-/16-bit grayscale or RGB TIFF preserving native pixel values."""
    # Scientific camera TIFFs are most reliably handled by tifffile.
    # Pillow is kept as a fallback so tifffile is not required.
    try:
        import tifffile
        raw = tifffile.imread(str(path))
    except (ImportError, OSError, ValueError, RuntimeError, KeyError):
        from PIL import Image
        with Image.open(path) as src:
            raw = np.asarray(src)
    raw = np.squeeze(raw)
    if raw.ndim == 3 and raw.shape[-1] in (3, 4):
        # Display an RGB/RGBA camera image as grayscale if necessary.
        raw = np.mean(raw[..., :3], axis=-1)
    if raw.ndim != 2:
        raise ValueError(f'Expected a 2-D camera image in {path}; got {raw.shape}')
    raw = raw.astype(np.float32)
    good = np.isfinite(raw)
    if not np.any(good):
        raise ValueError(f'No finite pixel values in {path}')
    low, high = np.percentile(raw[good], MEASURED_DISPLAY_PERCENTILES)
    if high <= low:
        high = float(np.max(raw[good]))
        low = float(np.min(raw[good]))
    if high <= low:
        return np.zeros_like(raw)
    return np.clip((raw - low) / (high - low), 0.0, 1.0)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Script directory: {Path(__file__).resolve().parent}')
    print(f'Looking for measured TIFFs in: {MEASURED_IMAGE_DIR.resolve()}')
    dx_mm = FOV_MM / N
    x_mm = (np.arange(N) - N // 2) * dx_mm
    fx_cyc_per_mm = np.fft.fftshift(np.fft.fftfreq(N, d=dx_mm))
    lam_mm = WAVELENGTH_NM * 1e-6
    # Paraxial Fourier-plane coordinates of first lens
    xf_mm = lam_mm * FOURIER_LENS_F_MM * fx_cyc_per_mm

    if UNFILTERED_PHOTO_PATH:
        E0 = read_unfiltered_photo(UNFILTERED_PHOTO_PATH, N)
        source_desc = 'Measured unfiltered camera image (zero phase assumed)'
    else:
        E0 = generate_grid(x_mm)
        source_desc = 'Synthetic Thorlabs 50 um grid, 5 um clear lines'

    F0 = rel_fft(E0)
    source_power = float(np.sum(np.abs(F0)**2))
    filters = build_masks(xf_mm)
    intensity_results = {}
    throughput_results = {}

    for label, mask in filters.items():
        F = F0 * mask
        I = np.abs(rel_ifft(F))**2
        intensity_results[label] = I
        throughput_results[label] = float(np.sum(np.abs(F)**2) / source_power)
        print(f'{label}: transmitted power = {throughput_results[label]*100:.3f}% of input')

    reference_max = float(np.max(next(iter(intensity_results.values()))))
    half_field = FOV_MM / 2
    fft_extent = [xf_mm[0], xf_mm[-1], xf_mm[0], xf_mm[-1]]
    image_extent_um = [-half_field * 1000, half_field * 1000,
                       -half_field * 1000, half_field * 1000]

    # Three columns only: aperture, simulated camera, measured camera.
    fig, axs = plt.subplots(4, 3, figsize=(15.5, 17.2),
                            gridspec_kw={'width_ratios': [1, 1, 1]})
    fig.suptitle('50 um grid: simulated vs. recorded Fourier filtering  |  '
                 f'f1 = {FOURIER_LENS_F_MM:g} mm, '
                 f'lambda = {WAVELENGTH_NM:g} nm', fontsize=17, y=0.995)

    for i, (label, mask) in enumerate(filters.items()):
        ax_mask, ax_sim, ax_real = axs[i]
        I = intensity_results[label]
        im_file = MEASURED_IMAGE_DIR / MEASURED_FILES[i]

        ax_mask.imshow(mask, origin='lower', extent=fft_extent,
                       cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax_mask.set_title(f'{label}\nAperture', fontsize=11)
        ax_mask.set_xlim(-13.5, 13.5)
        ax_mask.set_ylim(-13.5, 13.5)
        ax_mask.set_xlabel('Fourier-plane x (mm)')
        ax_mask.set_ylabel('Fourier-plane y (mm)')

        norm = max(float(I.max()), 1e-30) if NORMALIZE_EACH_FINAL_IMAGE else reference_max
        ax_sim.imshow(I / norm, extent=image_extent_um, origin='lower',
                      cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax_sim.set_title('Simulated camera image\n'
                         f'Transmission: {throughput_results[label]*100:.2f}%', fontsize=11)
        ax_sim.set_xlabel('Object x (um)')
        ax_sim.set_ylabel('Object y (um)')

        if im_file.is_file():
            try:
                recorded = load_measured_tiff(im_file)
                ax_real.imshow(recorded, origin='upper', cmap='gray',
                               vmin=0, vmax=1, interpolation='nearest')
                print(f'Loaded: {im_file}, shape={recorded.shape}')
            except Exception as exc:
                ax_real.text(0.5, 0.5, f'Cannot read TIFF\n{exc}',
                             transform=ax_real.transAxes,
                             ha='center', va='center', fontsize=10, wrap=True)
                print(f'WARNING: Cannot load {im_file}: {exc}')
        else:
            ax_real.text(0.5, 0.5, f'Recorded TIFF not found\n{im_file}',
                         transform=ax_real.transAxes,
                         ha='center', va='center', fontsize=11, wrap=True)
            print(f'WARNING: TIFF not found: {im_file}')

        ax_real.set_title(f'Recorded camera image\n{im_file.name}', fontsize=11)
        ax_real.set_xlabel('Camera x (pixels)')
        ax_real.set_ylabel('Camera y (pixels)')

        for ax in (ax_mask, ax_sim, ax_real):
            ax.set_aspect('equal')
            ax.tick_params(labelsize=9)

    fig.text(0.5, 0.009,
             'Simulations and recordings are normalized separately for display; '
             'recorded-image exposure times differ.',
             ha='center', fontsize=10)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.956,
                        bottom=0.047, hspace=0.34, wspace=0.32)
    fig_path = OUT_DIR / 'fourier_filter_comparison.png'
    fig.savefig(fig_path, dpi=160)
    fig.savefig(OUT_DIR / 'fourier_filter_comparison.svg', format='svg')
    plt.close(fig)

    # Save the source template and numeric transmission values.
    fig2, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(np.abs(E0)**2, extent=image_extent_um, origin='lower',
              cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ax.set_xlabel('Object x (um)')
    ax.set_ylabel('Object y (um)')
    ax.set_title(source_desc)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / 'unfiltered_template.png', dpi=160)
    plt.close(fig2)

    with open(OUT_DIR / 'power_throughput.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filter', 'power_fraction_of_input', 'percent'])
        for label, fraction in throughput_results.items():
            writer.writerow([label, f'{fraction:.9f}', f'{100*fraction:.6f}'])

    sep = lam_mm * FOURIER_LENS_F_MM / (GRID_PITCH_UM * 1e-3)
    print(f'Grid first-order spacing in Fourier plane: {sep:.4f} mm')
    print(f'Comparison figure saved to: {fig_path}')
    print(f'Source: {source_desc}')


if __name__ == '__main__':
    main()
