#!/usr/bin/env python3
"""
Customizable script to overlay polarization vectors from one or more CSV catalogs
onto a background FITS map.

Features added over the original:
- Dynamic FITS color scaling using astropy.visualization (handles arbitrary maps better).
- Command-line interface (CLI) to easily change parameters without editing code.
- `p_multiplier` parameter to handle catalogs where P is a fraction (0-1) instead of % (0-100).
- Automatic squeezing of FITS data to handle 3D/4D datacubes with degenerate axes.
- Clean separation of configuration.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.visualization import ImageNormalize, PercentileInterval, AsinhStretch

warnings.filterwarnings('ignore')

@dataclass
class Dataset:
    """One CSV catalog of polarization measurements to overlay."""
    path: str
    label: str = None
    color: str = "cyan"
    ra_col: str = "ra_wcs"      # Right Ascension (deg) column
    dec_col: str = "dec_wcs"    # Declination (deg) column
    p_col: str = "P_avg"        # Polarization degree column
    pa_col: str = "PA_avg"      # Position angle (deg, East of North) column
    p_multiplier: float = 1.0   # Set to 100.0 if P is a fraction instead of percentage
    scale: float = None         # per-dataset length scale override
    linewidth: float = None     # per-dataset line width override
    min_p: float = None         # Minimum polarization (%) to display
    max_p: float = None         # Maximum polarization (%) to display

    def __post_init__(self):
        if self.label is None:
            self.label = Path(self.path).stem


# =============================================================================
# DEFAULT CONFIGURATION (Fallback if run without CLI arguments)
# =============================================================================
DEFAULT_FITS_FILE = "ngc7538_jcmt.fits"
DEFAULT_OUTPUT_FILE = "ngc7538_pol_vectors_custom.png"

# Vector geometry defaults
DEFAULT_VECTOR_SCALE = 3.0        # Arcsec of vector length per 1% polarization
DEFAULT_LINE_WIDTH = 1.5
DEFAULT_MIN_P = None              # Min P in % (after applying multiplier)
DEFAULT_MAX_P = None              # Max P in % (after applying multiplier)
DRAW_LEGEND_SCALEBAR = True
LEGEND_P_PERCENT = 5.0            # Reference polarization drawn in the scale bar

# Add or edit entries here to compare multiple CSVs by default
DEFAULT_DATASETS = [
    Dataset("ngc7538_avg_R 2.csv", label="Kanata Results (Benchmark)",
            color="gray", p_col="P_avg", pa_col="PA_avg", ra_col="ra_wcs", dec_col="dec_wcs", p_multiplier=1.0, linewidth=2.0),
    # Dataset("ngc7538_polarimetry.csv", label="First try",
    #         color="red", p_col="P_etcal", pa_col="PA_etcal", ra_col="ra_deg", dec_col="dec_deg", p_multiplier=100.0, linewidth=2.0),
    Dataset("formatted_v3_polarimetry_catalog.csv", label="Pipeline V3 (Dual Beam)",
            color="cyan", p_col="P_avg", pa_col="PA_avg", ra_col="ra_wcs", dec_col="dec_wcs", p_multiplier=1.0, linewidth=1.5),
]


# =============================================================================
# CORE LOGIC
# =============================================================================

def load_background(fits_file):
    """Return (2D image_data, WCS) from the FITS primary HDU."""
    with fits.open(fits_file) as hdul:
        # Sometimes FITS data is in HDU 1 if 0 is empty
        ext = 0 if hdul[0].data is not None else 1
        data = hdul[ext].data.astype(float)
        data = np.squeeze(data) # Handle 1x1xNyxNx arrays

        if data.ndim != 2:
            raise ValueError(f"FITS file must contain 2D image data. Found {data.ndim}D after squeezing.")

        # Extract WCS only for the celestial axes
        wcs = WCS(hdul[ext].header).celestial
    return data, wcs

def compute_vector_endpoints(ra, dec, p, pa_deg, scale_arcsec_per_pct):
    """
    Given RA/Dec (deg), polarization P (%), and PA (deg, East of North),
    return endpoints in world coordinates.
    """
    half_len_deg = (p * scale_arcsec_per_pct / 2.0) / 3600.0
    pa = np.radians(pa_deg)

    ddec = half_len_deg * np.cos(pa)
    dra = half_len_deg * np.sin(pa) / np.cos(np.radians(dec))

    ra1, dec1 = ra - dra, dec - ddec
    ra2, dec2 = ra + dra, dec + ddec
    return (ra1, dec1), (ra2, dec2)

def overlay_dataset(ax, wcs, ds, global_min_p, global_max_p, global_scale, global_lw):
    """Read one CSV and draw its polarization vectors onto ax."""
    try:
        df = pd.read_csv(ds.path)
    except Exception as e:
        print(f"  [error] Failed to read {ds.path}: {e}")
        return 0

    # Ensure columns exist
    missing = [c for c in [ds.ra_col, ds.dec_col, ds.p_col, ds.pa_col] if c not in df.columns]
    if missing:
        print(f"  [error] Missing columns {missing} in {ds.path}")
        return 0

    # Drop NaNs
    cols = [ds.ra_col, ds.dec_col, ds.p_col, ds.pa_col]
    df = df.dropna(subset=cols).copy()

    # Convert P to percentage representation if needed
    df['_p_calc'] = df[ds.p_col] * ds.p_multiplier

    # Filtering
    lo = ds.min_p if ds.min_p is not None else global_min_p
    hi = ds.max_p if ds.max_p is not None else global_max_p
    if lo is not None:
        df = df[df['_p_calc'] >= lo]
    if hi is not None:
        df = df[df['_p_calc'] <= hi]

    ra = df[ds.ra_col].to_numpy()
    dec = df[ds.dec_col].to_numpy()
    p = df['_p_calc'].to_numpy()
    pa = df[ds.pa_col].to_numpy()

    scale = ds.scale if ds.scale is not None else global_scale
    lw = ds.linewidth if ds.linewidth is not None else global_lw

    (ra1, dec1), (ra2, dec2) = compute_vector_endpoints(ra, dec, p, pa, scale)

    x1, y1 = wcs.world_to_pixel_values(ra1, dec1)
    x2, y2 = wcs.world_to_pixel_values(ra2, dec2)

    # Use LineCollection for much faster rendering
    from matplotlib.collections import LineCollection
    segments = [[(xi1, yi1), (xi2, yi2)] for xi1, yi1, xi2, yi2 in zip(x1, y1, x2, y2)]
    lc = LineCollection(segments, color=ds.color, lw=lw, capstyle="round", zorder=5)
    ax.add_collection(lc)

    # Proxy line for the legend
    ax.plot([], [], color=ds.color, lw=lw + 0.6, label=f"{ds.label} (n={len(df)})")
    return len(df)

def add_scalebar(ax, wcs, image_shape, scale_arcsec_per_pct):
    """Draw a reference polarization vector so vector lengths are interpretable."""
    ny, nx = image_shape
    x0, y0 = 0.12 * nx, 0.08 * ny
    ra0, dec0 = wcs.pixel_to_world_values(x0, y0)

    half_len_deg = (LEGEND_P_PERCENT * scale_arcsec_per_pct / 2.0) / 3600.0

    # We draw horizontal line for scale
    ra1 = ra0 - half_len_deg / np.cos(np.radians(dec0))
    ra2 = ra0 + half_len_deg / np.cos(np.radians(dec0))

    xa, ya = wcs.world_to_pixel_values(ra1, dec0)
    xb, yb = wcs.world_to_pixel_values(ra2, dec0)

    ax.plot([xa, xb], [ya, yb], color="white", lw=2.5, solid_capstyle="round", zorder=6)
    ax.text((xa + xb) / 2, ya + 0.02 * ny, f"{LEGEND_P_PERCENT:.0f}% pol.",
            color="white", ha="center", va="bottom", fontsize=10, fontweight='bold', zorder=6)

def main():
    parser = argparse.ArgumentParser(description="Overlay polarization vectors from CSV onto a FITS map.")
    parser.add_argument("--fits", type=str, default=DEFAULT_FITS_FILE, help="Path to background FITS image")
    parser.add_argument("--out", type=str, default=DEFAULT_OUTPUT_FILE, help="Output PNG filename")
    parser.add_argument("--scale", type=float, default=DEFAULT_VECTOR_SCALE, help="Arcsec per 1 percent polarization")
    parser.add_argument("--min_p", type=float, default=DEFAULT_MIN_P, help="Minimum P(percent) to plot")
    parser.add_argument("--max_p", type=float, default=DEFAULT_MAX_P, help="Maximum P(percent) to plot")

    args = parser.parse_args()

    print(f"Loading background: {args.fits}")
    try:
        data, wcs = load_background(args.fits)
    except FileNotFoundError:
        print(f"ERROR: Background FITS file '{args.fits}' not found.")
        return

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection=wcs)

    # Dynamic normalization to avoid hardcoded vmin/vmax issues on different maps
    # norm = ImageNormalize(data, interval=PercentileInterval(99.5), stretch=AsinhStretch(a=0.1))
    im = ax.imshow(data, origin="lower", cmap="RdYlBu_r", vmin=-1, vmax=1.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Intensity")

    total = 0
    for ds in DEFAULT_DATASETS:
        if not Path(ds.path).exists():
            print(f"  [skip] {ds.path} not found")
            continue
        n = overlay_dataset(ax, wcs, ds, args.min_p, args.max_p, args.scale, DEFAULT_LINE_WIDTH)
        print(f"  [ok]   {ds.label}: {n} vectors plotted")
        total += n

    if DRAW_LEGEND_SCALEBAR and total > 0:
        add_scalebar(ax, wcs, data.shape, args.scale)

    ax.set_xlabel("Right Ascension (J2000)")
    ax.set_ylabel("Declination (J2000)")
    ax.set_title(f"Polarization Vectors overlaid on {Path(args.fits).stem}")
    ax.legend(loc="upper right", framealpha=0.7, facecolor="black", labelcolor="white", fontsize=9)
    ax.grid(color="white", ls=":", alpha=0.3)

    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSuccessfully saved {args.out} ({total} vectors total)")

if __name__ == "__main__":
    main()
