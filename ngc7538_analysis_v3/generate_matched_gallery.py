import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

OUT = Path("pol_out")
FIGURES_DIR = Path("../pipeline/figures")
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

APER_O_R = 12.0
APER_E_A, APER_E_B = 50.0, 12.0

def getdata(path):
    return fits.getdata(path).astype(np.float32)

def main():
    print("Loading calibration masters...")
    dark40 = getdata("pol_out/stack_dark40.fits")
    flat = getdata("pol_out/flat_single.fits")
    master_reduced = getdata("pol_out/master_reduced.fits")
    
    calib = json.load(open("pol_out/calib.json"))
    BEAM_DX = calib["beam_dx"]
    BEAM_DY = calib["beam_dy"]
    print(f"Loaded Beam Offset: dx={BEAM_DX:+.2f}, dy={BEAM_DY:+.2f}")
    
    sci_f = "../NGC7538_d0-0001_0.fit"
    # To see stars without nebula
    data_calib = (getdata(sci_f) - dark40) / flat - master_reduced
    # To see the context clearly
    data_full = (getdata(sci_f) - dark40) / flat
    
    df_k = pd.read_csv("../compare/ngc7538_avg_R 2.csv")
    df_v3 = pd.read_csv("formatted_v3_polarimetry_catalog.csv").drop_duplicates(subset=['ra_wcs', 'dec_wcs']).copy()
    
    ck = SkyCoord(ra=df_k['ra_wcs'], dec=df_k['dec_wcs'], unit='deg')
    cv3 = SkyCoord(ra=df_v3['ra_wcs'], dec=df_v3['dec_wcs'], unit='deg')
    
    idx, sep, _ = cv3.match_to_catalog_sky(ck)
    mask = sep.arcsec < 2.0
    
    matched_v3 = df_v3[mask].reset_index(drop=True)
    matched_k = df_k.iloc[idx[mask]].reset_index(drop=True)
    
    num_matches = len(matched_v3)
    print(f"Found {num_matches} matches! Generating individual plots...")
    
    for i in range(num_matches):
        row_v3 = matched_v3.iloc[i]
        row_k = matched_k.iloc[i]
        
        cx, cy = row_v3['X_IMAGE'], row_v3['Y_IMAGE']
        rx, ry = cx + BEAM_DX, cy + BEAM_DY
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Context Panel
        vmin, vmax = np.percentile(data_full, [5, 99.5])
        axes[0].imshow(data_full, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
        
        # Zoom context enough to see both bubbles but not the whole 4000px image
        pad = 200
        min_x = min(cx, rx) - pad
        max_x = max(cx, rx) + pad
        min_y = min(cy, ry) - pad
        max_y = max(cy, ry) + pad
        
        axes[0].set_xlim(min_x, max_x)
        axes[0].set_ylim(min_y, max_y)
        
        # Highlight boxes
        rect_o = Rectangle((cx-50, cy-50), 100, 100, fill=False, color='cyan', lw=2)
        rect_e = Rectangle((rx-60, ry-30), 120, 60, fill=False, color='red', lw=2)
        axes[0].add_patch(rect_o)
        axes[0].add_patch(rect_e)
        axes[0].text(cx, cy-60, "O-Beam", color='cyan', ha='center', va='top', fontweight='bold')
        axes[0].text(rx, ry-40, "E-Beam", color='red', ha='center', va='top', fontweight='bold')
        axes[0].set_title("Dual-Beam Context (Master Flat Corrected)")
        axes[0].set_xlabel("X (pixels)")
        axes[0].set_ylabel("Y (pixels)")
        
        # 2. Zoom O
        size = 60
        o_crop = data_calib[int(cy)-size:int(cy)+size, int(cx)-size:int(cx)+size]
        vmin, vmax = np.percentile(o_crop, [1, 99.5])
        axes[1].imshow(o_crop, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax, extent=[cx-size, cx+size, cy-size, cy+size])
        circ = Circle((cx, cy), radius=APER_O_R, fill=False, color='cyan', lw=2, linestyle='--')
        axes[1].add_patch(circ)
        axes[1].set_title("Ordinary Beam (O) Zoom")
        axes[1].set_xlabel("X (pixels)")
        
        # 3. Zoom E
        e_crop = data_calib[int(ry)-size:int(ry)+size, int(rx)-size:int(rx)+size]
        vmin, vmax = np.percentile(e_crop, [1, 99.5])
        axes[2].imshow(e_crop, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax, extent=[rx-size, rx+size, ry-size, ry+size])
        # Major axis width is 2*A, height is 2*B
        ell = Ellipse((rx, ry), width=2*APER_E_A, height=2*APER_E_B, fill=False, color='red', lw=2, linestyle='--')
        axes[2].add_patch(ell)
        axes[2].set_title("Extraordinary Beam (E) Streak Zoom")
        axes[2].set_xlabel("X (pixels)")
        
        c_str = SkyCoord(row_v3['ra_wcs'], row_v3['dec_wcs'], unit='deg').to_string('hmsdms', precision=2)
        fig.suptitle(f"Matched Star #{i+1} | RA/DEC: {c_str} | DESIGNATION: {row_v3['DESIGNATION']}\n"
                     f"Pipeline V3 (Dual Beam):  P = {row_v3['P_avg']:.2f}%  |  PA = {row_v3['PA_avg']:.1f} deg\n"
                     f"Kanata (Benchmark):       P = {row_k['P_avg']:.2f}%  |  PA = {row_k['PA_avg']:.1f} deg", 
                     fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"matched_star_{i+1:02d}.png", dpi=100, bbox_inches='tight')
        plt.close()
        
    print("Saved all matched star galleries to pipeline/figures/")

if __name__ == "__main__":
    main()
