import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from pathlib import Path
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings
warnings.filterwarnings('ignore')

RAW_DIR = ".."
OUT = Path("pol_out")
FIGURES_DIR = Path("../pipeline/figures")
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

APER_O_R = 12.0
APER_E_A, APER_E_B = 50.0, 12.0

def getdata(path):
    return fits.getdata(path).astype(np.float32)

def main():
    print("Loading masters and catalogs...")
    bias = getdata("pol_out/stack_bias.fits")
    dark40 = getdata("pol_out/stack_dark40.fits")
    flat = getdata("pol_out/flat_single.fits")
    master_reduced = getdata("pol_out/master_reduced.fits")
    
    # Load calibrated parameters
    calib = json.load(open("pol_out/calib.json"))
    BEAM_DX = calib["beam_dx"]
    BEAM_DY = calib["beam_dy"]
    print(f"Using Beam Offset: dx={BEAM_DX:+.2f}, dy={BEAM_DY:+.2f} px")
    
    # Load the science catalog
    df = pd.read_csv("pol_out/ngc7538_polarimetry.csv")
    
    # Let's find the top 6 most reliable, bright stars to display in the gallery
    # We will include Star I, II, III based on their matches, and 3 other bright field stars
    target_coords = {
        "Star I": SkyCoord("23h13m36.813s +61d30m39.61s", frame='icrs'),
        "Star II": SkyCoord("23h13m34.397s +61d30m14.63s", frame='icrs'),
        "Star III": SkyCoord("23h13m30.243s +61d30m10.33s", frame='icrs')
    }
    
    # Add coordinates of the stars in catalog
    cat_coords = SkyCoord(ra=df['ra_deg'], dec=df['dec_deg'], unit=(u.deg, u.deg))
    
    df['label'] = "Field Star"
    for name, sc in target_coords.items():
        idx, sep, _ = sc.match_to_catalog_sky(cat_coords)
        if sep[0].arcsec < 5.0:
            df.at[int(idx), 'label'] = name
            
    # Sort: put labeled targets first, then sort remaining by brightness (flux_o0)
    df_targets = df[df['label'] != 'Field Star']
    df_field = df[df['label'] == 'Field Star'].sort_values('flux_o0', ascending=False)
    
    # Take top 3 field stars + the 3 targets = 6 stars total for the gallery
    gallery_stars = pd.concat([df_targets, df_field.head(3)]).reset_index(drop=True)
    
    # Load one calibrated science frame to extract the stellar crops
    # We'll use the d0-0001_0 frame
    sci_f = f"{RAW_DIR}/NGC7538_d0-0001_0.fit"
    # (RAW - DARK) / FLAT - MASTER to isolate stars from nebulosity
    data_calib = (getdata(sci_f) - dark40) / flat - master_reduced
    
    # We also need WCS to display coordinates
    hdr = fits.getheader(sci_f)
    wcs = WCS(hdr).celestial
    # Adjust WCS using our solved translation & scale from pipeline if needed,
    # but the catalog already has the absolute RA/Dec coordinates which is what we want!
    
    print(f"Creating gallery of {len(gallery_stars)} stars...")
    
    fig, axes = plt.subplots(6, 2, figsize=(12, 24))
    
    for i, row in gallery_stars.iterrows():
        cx, cy = row['x'], row['y']
        rx, ry = cx + BEAM_DX, cy + BEAM_DY
        
        # Crop Ordinary
        size = 60
        o_crop = data_calib[int(cy)-size:int(cy)+size, int(cx)-size:int(cx)+size]
        
        # Crop Extraordinary
        e_crop = data_calib[int(ry)-size:int(ry)+size, int(rx)-size:int(rx)+size]
        
        # Plot Ordinary
        vmin, vmax = np.percentile(o_crop, [1, 99.5])
        axes[i, 0].imshow(o_crop, cmap='viridis', origin='lower', extent=[cx-size, cx+size, cy-size, cy+size])
        axes[i, 0].set_title(f"{row['label']} - Ordinary Beam (O)")
        axes[i, 0].set_xlabel("X (pixels)")
        axes[i, 0].set_ylabel("Y (pixels)")
        
        # Draw circular aperture
        circ = Circle((cx, cy), radius=APER_O_R, fill=False, color='red', lw=2, linestyle='--', label=f'Circular Aperture (R={APER_O_R})')
        axes[i, 0].add_patch(circ)
        
        # Plot Extraordinary
        vmin, vmax = np.percentile(e_crop, [1, 99.5])
        axes[i, 1].imshow(e_crop, cmap='viridis', origin='lower', extent=[rx-size, rx+size, ry-size, ry+size])
        axes[i, 1].set_title(f"{row['label']} - Extraordinary Beam (E) Streak")
        axes[i, 1].set_xlabel("X (pixels)")
        axes[i, 1].set_ylabel("Y (pixels)")
        
        # Draw elliptical aperture
        ell = Ellipse((rx, ry), width=2*APER_E_A, height=2*APER_E_B, fill=False, color='red', lw=2, linestyle='--', label=f'Elliptical Aperture (a={APER_E_A}, b={APER_E_B})')
        axes[i, 1].add_patch(ell)
        
        # Add metadata text
        ra_str = f"{row['ra_deg']:.6f}"
        dec_str = f"{row['dec_deg']:.6f}"
        
        # Convert degrees to sexagesimal format for professional look
        c_star = SkyCoord(row['ra_deg'], row['dec_deg'], unit='deg')
        coord_str = c_star.to_string('hmsdms', precision=2)
        
        meta_text = (
            f"RA: {coord_str.split()[0]} | DEC: {coord_str.split()[1]}\n"
            f"P: {row['P_debiased']*100:.2f}% +/- {row['sP']*100:.2f}%\n"
            f"PA: {row['theta_sky']:.1f} +/- {row['stheta']:.1f} deg\n"
            f"Cycles: {int(row['n_cycles'])} / 9"
        )
        
        # Add text in the middle or top of the row
        fig.text(0.5, 0.985 - i*0.163, meta_text, ha='center', va='top', fontsize=12, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(FIGURES_DIR / "step6_stars_gallery.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Gallery generated and saved to pipeline/figures/step6_stars_gallery.png!")

if __name__ == "__main__":
    import json
    main()
