import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
import warnings
warnings.filterwarnings('ignore')

RAW_DIR = ".."
OUT = Path("pol_out")
OUT.mkdir(exist_ok=True)
FIGURES_DIR = Path("../pipeline/figures")
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

ANGLES = ["0", "22_5", "45", "67_5"]

def getdata(path):
    return fits.getdata(path).astype(np.float32)

def main():
    print("Generating Pipeline Step Visualizations...")
    
    # -------------------------------------------------------------
    # STEP 1: BIAS & DARK CALIBRATION
    # -------------------------------------------------------------
    print("  Step 1: Bias & Dark...")
    raw_bias_f = sorted(glob.glob(f"{RAW_DIR}/BIAS-000?_b1.fit"))[0]
    raw_bias = getdata(raw_bias_f)
    master_bias = getdata("pol_out/stack_bias.fits")
    
    raw_dark_f = sorted(glob.glob(f"{RAW_DIR}/DARK-000?_D40.fit"))[0]
    raw_dark = getdata(raw_dark_f)
    master_dark = getdata("pol_out/stack_dark40.fits")
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].imshow(raw_bias, cmap='gray', origin='lower')
    axes[0, 0].set_title(f"Raw Bias Frame ({Path(raw_bias_f).name})")
    
    axes[0, 1].imshow(master_bias, cmap='gray', origin='lower')
    axes[0, 1].set_title("Master Bias (Median Stacked)")
    
    axes[1, 0].imshow(raw_dark, cmap='gray', origin='lower')
    axes[1, 0].set_title(f"Raw 40s Dark ({Path(raw_dark_f).name})")
    
    axes[1, 1].imshow(master_dark, cmap='gray', origin='lower')
    axes[1, 1].set_title("Master Dark40 (Median Stacked)")
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "step1_bias_dark.png", dpi=150)
    plt.close()
    
    # -------------------------------------------------------------
    # STEP 2: MASTER FLAT
    # -------------------------------------------------------------
    print("  Step 2: Master Flat...")
    master_flat = getdata("pol_out/flat_single.fits")
    
    plt.figure(figsize=(8, 6))
    plt.imshow(master_flat, cmap='viridis', origin='lower')
    plt.colorbar(label="Normalized Transmission")
    plt.title("Master Flat-Field (Normalized, All-Angles Combined)")
    plt.savefig(FIGURES_DIR / "step2_master_flat.png", dpi=150)
    plt.close()
    
    # -------------------------------------------------------------
    # STEP 3: SCIENCE REDUCTION
    # -------------------------------------------------------------
    print("  Step 3: Science Frame Calibration...")
    raw_sci_f = f"{RAW_DIR}/NGC7538_d0-0001_0.fit"
    raw_sci = getdata(raw_sci_f)
    
    # Reduction formula: (RAW - DARK) / FLAT
    calib_sci = (raw_sci - master_dark) / master_flat
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    vmin, vmax = np.percentile(raw_sci, [5, 99.5])
    axes[0].imshow(raw_sci, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Raw Science Frame\n(Contains Dark Current & QE Defects)")
    
    vmin, vmax = np.percentile(calib_sci, [5, 99.5])
    axes[1].imshow(calib_sci, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Calibrated Science Frame\n((RAW - Master Dark) / Master Flat)")
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "step3_reduction.png", dpi=150)
    plt.close()
    
    # -------------------------------------------------------------
    # STEP 4: O->E STREAK PROFILING
    # -------------------------------------------------------------
    print("  Step 4: O->E Offset Profiling...")
    # Let's plot standard star VI Cyg 12
    data = calib_sci # just use science frame for illustration of O and E positions
    cx, cy = 1797.74, 1630.47 # Star I
    BEAM_DX, BEAM_DY = 1362.3, 34.0
    exc, eyc = int(cx + BEAM_DX), int(cy + BEAM_DY)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # O beam box
    o_box = data[int(cy)-30:int(cy)+30, int(cx)-30:int(cx)+30]
    axes[0, 0].imshow(o_box, cmap='viridis', origin='lower')
    axes[0, 0].set_title("Ordinary (O) Beam Point Source")
    
    # E beam box with search limits
    e_box = data[eyc-15:eyc+16, exc-60:exc+60]
    axes[0, 1].imshow(e_box, cmap='viridis', origin='lower', extent=[exc-60, exc+60, eyc-15, eyc+15])
    # Draw profiling boxes
    rect = plt.Rectangle((exc-50, eyc-10), 100, 20, fill=False, color='red', lw=2, linestyle='--')
    axes[0, 1].add_patch(rect)
    axes[0, 1].set_title("Extraordinary (E) Beam Streak\n(Elliptical Photometry Box in Red)")
    
    # Vertical profile dy
    ybandy = data[eyc-15:eyc+16, exc-60:exc+60].sum(axis=1)
    ybandy_sub = ybandy - np.median(ybandy)
    axes[1, 0].plot(np.arange(len(ybandy_sub)) - 15 + eyc, ybandy_sub, color='red', marker='o')
    axes[1, 0].axvline(eyc - 15 + float((np.arange(31)*np.clip(ybandy_sub,0,None)).sum() / np.clip(ybandy_sub,0,None).sum()), color='black', linestyle='--')
    axes[1, 0].set_title("Vertical Profile (Refining dy)")
    axes[1, 0].set_xlabel("Y (pixels)")
    
    # Horizontal profile dx
    dy_ref = 34.0
    band = data[int(cy+dy_ref)-8:int(cy+dy_ref)+9, exc-60:exc+60]
    prof = band.sum(axis=0)
    prof_sub = prof - np.median(prof)
    axes[1, 1].plot(np.arange(len(prof_sub)) - 60 + exc, prof_sub, color='blue')
    axes[1, 1].set_title("Horizontal Profile (Refining dx)")
    axes[1, 1].set_xlabel("X (pixels)")
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "step4_profiling.png", dpi=150)
    plt.close()
    
    # -------------------------------------------------------------
    # STEP 5: MASTER SUBTRACTION
    # -------------------------------------------------------------
    print("  Step 5: Master Subtraction...")
    master_reduced = getdata("pol_out/master_reduced.fits")
    sub_sci = calib_sci - master_reduced
    
    # Crop to the O bubble
    O_BUB_cx, O_BUB_cy, O_BUB_r = 1648, 1717, 452
    x0 = int(O_BUB_cx - O_BUB_r) ; x1 = int(O_BUB_cx + O_BUB_r)
    y0 = int(O_BUB_cy - O_BUB_r) ; y1 = int(O_BUB_cy + O_BUB_r)
    
    crop_calib = calib_sci[y0:y1, x0:x1]
    crop_sub = sub_sci[y0:y1, x0:x1]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    vmin, vmax = np.percentile(crop_calib, [5, 99])
    axes[0].imshow(crop_calib, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title("Calibrated Frame\n(Contains Dense Nebulosity of NGC 7538)")
    
    vmin, vmax = np.percentile(crop_sub, [5, 99])
    axes[1].imshow(crop_sub, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title("Master-Subtracted Frame\n(Nebula Removed, Point Sources Stand Out)")
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "step5_subtraction.png", dpi=150)
    plt.close()
    
    print("All intermediate step figures saved in pipeline/figures/!")

if __name__ == "__main__":
    main()
