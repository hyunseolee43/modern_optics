"""Scalar diffraction synthesis of ATF, OTF, and pixel-integrated PSFs for pupil stops 1, 3, 4.
Columns in the summary figure:
1) ATF magnitude (pupil function)
2) OTF magnitude
3) Pixel-integrated PSF in the image plane (x,y) at z=0
4) Pixel-integrated PSF in the meridional plane (x,z) at y=0

Assumptions:
- wavelength = 532 nm
- Fourier / tube lens focal length = 100 mm
- stop located in the pupil plane of lens 4
- DCC3240C camera pixel pitch = 5.3 um
- aperture 4 geometry is estimated from the sketch
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from pathlib import Path

OUT = Path(__file__).resolve().parent
WAVELENGTH_MM = 532e-6
FOCAL_LENGTH_MM = 100.0
PIXEL_MM = 5.3e-3
N = 1024
PUPIL_SPAN_MM = 50.0
APERTURE_DIAMETER_MM = 25.0
SLIT_LENGTH_MM = 20.0
SLIT_WIDTH_MM = 0.5
INNER_OPEN_DIAMETER_4_MM = 22.5  # estimated clear aperture
DIAGONAL_STRUT_WIDTH_MM = 0.5    # estimated obstruction width
CENTER_STOP_DIAMETER_MM = 2.0
Z_MAX_UM = 150.0
NZ = 121
XY_VIEW_UM = 45.0
OTF_VIEW_CY_PER_MM = 500.0

step_pupil = PUPIL_SPAN_MM / N
p = (np.arange(N) - N // 2) * step_pupil
X, Y = np.meshgrid(p, p)
r = np.hypot(X, Y)

P = {}
P[1] = (r <= APERTURE_DIAMETER_MM / 2).astype(float)
P[3] = ((np.abs(X) <= SLIT_LENGTH_MM / 2) & (np.abs(Y) <= SLIT_WIDTH_MM / 2)).astype(float)
P[4] = ((r <= INNER_OPEN_DIAMETER_4_MM / 2) &
        (np.abs(X + Y) / np.sqrt(2) > DIAGONAL_STRUT_WIDTH_MM / 2) &
        (r > CENTER_STOP_DIAMETER_MM / 2)).astype(float)

# Imaging-plane coordinate sampling from the Fourier transform relation.
step_image_mm = WAVELENGTH_MM * FOCAL_LENGTH_MM / PUPIL_SPAN_MM
image_um = (np.arange(N) - N // 2) * step_image_mm * 1000
nu = np.fft.fftshift(np.fft.fftfreq(N, d=step_image_mm))
cam_window = max(1, round(PIXEL_MM / step_image_mm))
z_um = np.linspace(-Z_MAX_UM, Z_MAX_UM, NZ)
z_mm = z_um * 1e-3

# Defocus phase in the pupil for axial displacement from paraxial image plane.
defocus_prefactor = np.pi / (WAVELENGTH_MM * FOCAL_LENGTH_MM ** 2)
r2 = X**2 + Y**2

fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
metrics = []

for row, k in enumerate([1, 3, 4]):
    pupil = P[k]

    # In-focus coherent field / intensity PSF.
    field0 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))
    psf0 = np.abs(field0) ** 2
    psf0 /= psf0.sum()
    pixel_psf0 = uniform_filter(psf0, size=cam_window, mode='constant')

    # Incoherent OTF from the ideal intensity PSF.
    otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf0)))
    otf /= otf[N // 2, N // 2]
    otf_mag = np.abs(otf)

    # Axial synthesis for x-z slice at y=0 using the pixel-integrated PSF.
    xz = np.empty((NZ, N), dtype=np.float32)
    for iz, z in enumerate(z_mm):
        phase = np.exp(-1j * defocus_prefactor * z * r2)
        field_z = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil * phase)))
        psf_z = np.abs(field_z) ** 2
        psf_z /= psf_z.sum()
        pixel_psf_z = uniform_filter(psf_z, size=cam_window, mode='constant')
        xz[iz, :] = pixel_psf_z[N // 2, :]

    np.savez_compressed(
        OUT / f'aperture_{k}_results.npz',
        pupil=pupil.astype(np.uint8),
        atf=pupil.astype(np.float32),
        psf=psf0.astype(np.float32),
        otf=otf.astype(np.complex64),
        camera_pixel_integrated_psf=pixel_psf0.astype(np.float32),
        camera_pixel_integrated_psf_xz=xz,
        pupil_mm=p,
        image_um=image_um,
        z_um=z_um,
        frequency_cycles_per_mm=nu,
    )

    # Column 1: ATF
    ax = axes[row, 0]
    ax.imshow(
        pupil,
        origin='lower',
        cmap='gray',
        extent=[p[0], p[-1], p[0], p[-1]],
        vmin=0,
        vmax=1,
    )
    ax.set_xlim(-14, 14)
    ax.set_ylim(-14, 14)
    ax.set_title(f'Aperture {k}: |ATF|')
    ax.set_xlabel('Pupil x (mm)')
    ax.set_ylabel('Pupil y (mm)')

    # Column 2: OTF magnitude
    ax = axes[row, 1]
    ax.imshow(
        otf_mag,
        origin='lower',
        cmap='viridis',
        vmin=0,
        vmax=1,
        extent=[nu[0], nu[-1], nu[0], nu[-1]],
    )
    ax.set_xlim(-OTF_VIEW_CY_PER_MM, OTF_VIEW_CY_PER_MM)
    ax.set_ylim(-OTF_VIEW_CY_PER_MM, OTF_VIEW_CY_PER_MM)
    ax.set_title('|OTF|')
    ax.set_xlabel('Spatial freq x (cycles/mm)')
    ax.set_ylabel('Spatial freq y (cycles/mm)')

    # Column 3: pixel-integrated PSF at z=0 (xy plane)
    ax = axes[row, 2]
    arr_xy = pixel_psf0 / pixel_psf0.max()
    ax.imshow(
        np.log10(np.maximum(arr_xy, 1e-5)),
        origin='lower',
        extent=[image_um[0], image_um[-1], image_um[0], image_um[-1]],
        cmap='inferno',
        vmin=-5,
        vmax=0,
    )
    ax.set_xlim(-XY_VIEW_UM, XY_VIEW_UM)
    ax.set_ylim(-XY_VIEW_UM, XY_VIEW_UM)
    ax.set_title('Pixel-integrated PSF, z = 0')
    ax.set_xlabel('Image x (µm)')
    ax.set_ylabel('Image y (µm)')

    # Column 4: pixel-integrated PSF in xz plane at y=0
    ax = axes[row, 3]
    arr_xz = xz / xz.max()
    ax.imshow(
        np.log10(np.maximum(arr_xz, 1e-5)),
        origin='lower',
        aspect='auto',
        extent=[image_um[0], image_um[-1], z_um[0], z_um[-1]],
        cmap='inferno',
        vmin=-5,
        vmax=0,
    )
    ax.set_xlim(-XY_VIEW_UM, XY_VIEW_UM)
    ax.set_ylim(-Z_MAX_UM, Z_MAX_UM)
    ax.set_title('Pixel-integrated PSF, y = 0')
    ax.set_xlabel('Image x (µm)')
    ax.set_ylabel('Defocus z (µm)')

    metrics.append((k, pupil.sum() * step_pupil ** 2, pixel_psf0.max(), xz.max()))

fig.suptitle('Apertures 1, 3, 4 | λ = 532 nm | f = 100 mm | camera pixel = 5.3 µm', fontsize=16)
fig.savefig(OUT / 'ATF_PSF_OTF_comparison.png', dpi=180)
plt.close(fig)

print('Image-plane grid step (um):', step_image_mm * 1000)
print('Pixel integration box width (samples):', cam_window)
print('Camera Nyquist (cycles/mm):', 1 / (2 * PIXEL_MM))
print('Defocus range (um):', (-Z_MAX_UM, Z_MAX_UM), 'with', NZ, 'samples')
print('Aperture, effective open pupil area (mm^2), peak pixel-integrated xy PSF, peak xz slice value')
for item in metrics:
    print(item)
