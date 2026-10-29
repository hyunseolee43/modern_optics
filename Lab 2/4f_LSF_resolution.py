"""Analyze the 50 µm grid in the unfiltered 532 nm camera image.

Run next to the Lab 2/data/ folder:
    python 5_lsf_532nm.py
or specify a TIFF path:
    python 5_lsf_532nm.py --image "C:\\...\\data\\aperture1_10us.tif"

Dependencies: numpy, scipy, matplotlib, pillow
Outputs: 4f_simulation_results/lsf_532nm_summary.png
         4f_simulation_results/lsf_532nm_summary.csv

Method: Average many short, non-intersection grid-line segments. Fit each
5 µm-wide line as a rectangular intensity profile convolved with a Gaussian;
fit aligned averaged profiles for a final effective Gaussian LSF estimate.
The estimate includes camera sampling and is NOT a diffraction-limited PSF
measurement; coherent-image formation may not follow an intensity convolution.
"""
from pathlib import Path
import argparse
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.special import ndtr

PITCH_UM = 50.0
LINE_WIDTH_UM = 5.0
LAMBDA_NM = 532.0
APERTURE_DIAMETER_MM = 25.0
FIRST_LENS_F_MM = 30.0
FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))


def gaussian(x, b, a, x0, sigma):
    return b + a * np.exp(-0.5 * ((x - x0) / sigma) ** 2)


def blurred_line(x, baseline, amplitude, x0, sigma, width_px):
    """Gaussian-blurred rect; amplitude is unblurred line height."""
    sigma = np.maximum(sigma, 1e-5)
    z1 = (x - x0 + width_px / 2) / sigma
    z2 = (x - x0 - width_px / 2) / sigma
    return baseline + amplitude * (ndtr(z1) - ndtr(z2))


def find_periodic_peaks(image):
    ny, nx = image.shape
    # Exclude the illuminated logo, grid edges, and top/bottom borders.
    x_projection = image[int(.17*ny):int(.85*ny), int(.12*nx):int(.78*nx)].mean(axis=0)
    y_projection = image[int(.14*ny):int(.88*ny), int(.17*nx):int(.79*nx)].mean(axis=1)
    ans = []
    for proj, offset in [(x_projection, int(.12*nx)), (y_projection, int(.14*ny))]:
        prominence = max(6., 0.12 * (np.percentile(proj, 95)-np.percentile(proj, 5)))
        peaks, props = find_peaks(proj, distance=30, prominence=prominence)
        peaks += offset
        # A few detected false/missed peaks can be rejected through median spacing.
        if len(peaks) < 9:
            raise RuntimeError('Could not detect enough grid lines. Check crop and focus.')
        ans.append(peaks)
    return ans


def patches(image, peaks_parallel, peaks_perp, direction, um_per_px):
    ny, nx = image.shape
    coords = np.arange(-14, 15, dtype=float)
    width_px = LINE_WIDTH_UM / um_per_px
    accepted = []
    for index in range(3, len(peaks_parallel)-3):
        line = int(peaks_parallel[index])
        for j in range(3, len(peaks_perp)-4):
            between = int(round((peaks_perp[j]+peaks_perp[j+1])/2))
            if direction == 'x':
                crop = image[between-8:between+9, line-14:line+15]
                if crop.shape != (17,29):
                    continue
                profile = crop.mean(axis=0)
            else:
                crop = image[line-14:line+15, between-8:between+9]
                if crop.shape != (29,17):
                    continue
                profile = crop.mean(axis=1)
            if (crop == 255).mean() > .03:  # Reject saturated line segments.
                continue
            b0 = max(0., float(np.median(np.r_[profile[:5], profile[-5:]])))
            a0 = float(profile.max()-b0)
            if a0 < 18:
                continue
            try:
                params, _ = curve_fit(
                    lambda xx,b,a,x0,s: blurred_line(xx,b,a,x0,s,width_px),
                    coords, profile, p0=[b0,max(a0,5),0.,1.5],
                    bounds=([0.,0.,-7.,.25],[200.,600.,7.,9.]), maxfev=4000)
            except (RuntimeError, ValueError):
                continue
            fitted = blurred_line(coords, *params, width_px)
            quality = 1 - np.sum((profile-fitted)**2) / max(1e-9,np.sum((profile-profile.mean())**2))
            if quality < 0.78 or abs(params[2])>6.7 or params[3]>6.5:
                continue
            if params[1] < 15:
                continue
            # Normalize amplitude relative to the real maximum (rather than
            # extrapolated peak) and align to the fitted line center.
            peak_height = max(fitted.max()-params[0],1)
            norm_profile = (profile-params[0])/peak_height
            accepted.append((coords-params[2],norm_profile,params[3],quality))
    return accepted


def stacked_fit(samples, um_per_px):
    if len(samples) < 10:
        raise RuntimeError(f'Only {len(samples)} clean line profiles; need at least 10.')
    u = np.linspace(-10.,10.,201)
    stack = np.array([np.interp(u, xp, yp, left=np.nan,right=np.nan)
                      for xp,yp,_,_ in samples])
    averaged = np.nanmean(stack,axis=0)
    spread = np.nanstd(stack,axis=0)/np.sqrt(stack.shape[0])
    width_px = LINE_WIDTH_UM/um_per_px
    # Fit the principal line only: neighboring diffraction/ringing features
    # outside ~5.5 µm are not part of the central line profile.
    core = np.abs(u) <= 5.5
    pars, _ = curve_fit(
        lambda xx,b,a,s:blurred_line(xx,b,a,0.,s,width_px),
        u[core],averaged[core],p0=[0.,1.1,1.5],
        bounds=([-.2,.1,.15],[.3,4.,12.]),maxfev=10000)
    fitted = blurred_line(u,pars[0],pars[1],0.,pars[2],width_px)
    # FWHM of observed target-line profile, distinct from effective blur LSF.
    mid = pars[0] + 0.5*(fitted.max()-pars[0])
    above = u[fitted>=mid]
    measured_fwhm_um = (above[-1]-above[0])*um_per_px
    lsf_um = FWHM_FACTOR*pars[2]*um_per_px
    return dict(u=u,mean=averaged,sem=spread,fit=fitted,params=pars,
                blur_fwhm_um=lsf_um,profile_fwhm_um=measured_fwhm_um,
                n=len(samples),um_per_px=um_per_px)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',type=Path,
       default=Path(__file__).resolve().parent/'data'/'aperture1_10us.tif')
    parser.add_argument('--output-dir',type=Path,
       default=Path(__file__).resolve().parent/'4f_simulation_results')
    args=parser.parse_args()
    path=args.image
    if not path.is_file():
        alternate=Path(__file__).resolve().parent/path.name
        if alternate.is_file(): path=alternate
        else: raise FileNotFoundError(f'Cannot find TIFF: {path}')
    img=np.array(Image.open(path).convert('L'),dtype=np.float64)
    px,py=find_periodic_peaks(img)
    # Median spacing is resistant to isolated detection errors.
    dx=np.median(np.diff(px));dy=np.median(np.diff(py))
    umx=PITCH_UM/dx;umy=PITCH_UM/dy
    x_samples=patches(img,px,py,'x',umx)
    y_samples=patches(img,py,px,'y',umy)
    fits={'x':stacked_fit(x_samples,umx),'y':stacked_fit(y_samples,umy)}
    # Geometric NA assuming a fully filled 25 mm pupil and air on object side.
    theta=np.arctan((APERTURE_DIAMETER_MM/2)/FIRST_LENS_F_MM)
    na=np.sin(theta)
    abbe_um=(LAMBDA_NM*1e-3)/(2*na)
    print(f'TIFF: {path}\nGrid pitch: {dx:.2f} px (x), {dy:.2f} px (y)')
    print(f'Object sampling: {umx:.3f} µm/px (x), {umy:.3f} µm/px (y)')
    for key in ('x','y'):
        d=fits[key]
        print(f'{key}: {d["n"]} clean segments; target-line FWHM = {d["profile_fwhm_um"]:.2f} µm; '
              f'Gaussian effective LSF FWHM = {d["blur_fwhm_um"]:.2f} µm')
    print(f'Assumed NA = {na:.3f}; Abbe d = {abbe_um:.3f} µm')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(15.5,4.5),constrained_layout=True)
    fig.suptitle('50 µm grid · 532 nm · reference 25 mm aperture',fontsize=16,weight='semibold')
    ax=axes[0]
    ax.imshow(img,cmap='gray',vmin=np.percentile(img,2),vmax=np.percentile(img,99.6))
    ax.set_xlim(240,900);ax.set_ylim(820,170)
    ax.set_title('(a) Unfiltered image · 10 µs')
    ax.set_xlabel('Camera x (pixels)');ax.set_ylabel('Camera y (pixels)')
    color={'x':'#1f77b4','y':'#d65f40'}
    ax=axes[1]
    for key,tag in [('x','Vertical grid lines (x)'),('y','Horizontal grid lines (y)')]:
        d=fits[key]
        um=d['u']*d['um_per_px']
        cc=color[key]
        ax.plot(um,d['mean'],color=cc,lw=1.4,alpha=.65,label=f'{tag}: measured')
        ax.plot(um,d['fit'],ls='--',color=cc,lw=2,label=f'Fit: LSF {d["blur_fwhm_um"]:.2f} µm')
    ax.set_xlim(-5.5,5.5)
    ax.set_title('(b) Grid-line profiles and 5 µm-line fits')
    ax.set_xlabel('Object-space displacement (µm)');ax.set_ylabel('Normalized intensity')
    ax.grid(alpha=.22);ax.legend(frameon=False,fontsize=8)
    ax=axes[2]
    names=['LSF: x','LSF: y','Abbe limit']
    values=[fits['x']['blur_fwhm_um'],fits['y']['blur_fwhm_um'],abbe_um]
    bars=ax.bar(names,values,color=[color['x'],color['y'],'#777777'],width=.63)
    for rect,val in zip(bars,values):
        ax.text(rect.get_x()+rect.get_width()/2,val+.12,f'{val:.2f}',ha='center',fontsize=11)
    ax.set_ylim(0,max(values)*1.25+0.25)
    ax.set_ylabel('Object-space length (µm)')
    ax.set_title('(c) Effective LSF FWHM vs. Abbe benchmark')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.text(.5,-.02,'LSF is a fitted Gaussian blur after accounting for the 5 µm target lines; it includes sampling and is not a direct NA measurement.',
             ha='center',fontsize=9,color='#555555')
    imageout=args.output_dir/'lsf_532nm_summary.png'
    fig.savefig(imageout,dpi=180,bbox_inches='tight')
    plt.close(fig)
    csvout=args.output_dir/'lsf_532nm_summary.csv'
    with csvout.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['metric','value','unit'])
        for name,val,unit in [
            ('wavelength',LAMBDA_NM,'nm'),('aperture_diameter',APERTURE_DIAMETER_MM,'mm'),
            ('focal_length',FIRST_LENS_F_MM,'mm'),('grid_pitch',PITCH_UM,'µm'),
            ('target_line_width',LINE_WIDTH_UM,'µm'),('NA_assumed',na,''),
            ('Abbe_limit',abbe_um,'µm'),('x_calibration',umx,'µm/pixel'),
            ('y_calibration',umy,'µm/pixel'),
            ('x_line_profile_FWHM',fits['x']['profile_fwhm_um'],'µm'),
            ('y_line_profile_FWHM',fits['y']['profile_fwhm_um'],'µm'),
            ('x_effective_LSF_FWHM',fits['x']['blur_fwhm_um'],'µm'),
            ('y_effective_LSF_FWHM',fits['y']['blur_fwhm_um'],'µm'),
            ('number_x_patches',fits['x']['n'],'profiles'),
            ('number_y_patches',fits['y']['n'],'profiles')]:w.writerow([name,round(float(val),5),unit])
    print('Saved figure:',imageout,'\nSaved results:',csvout)

if __name__=='__main__':main()
