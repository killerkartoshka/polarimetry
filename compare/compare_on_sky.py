#!/usr/bin/env python3
"""
Overlay polarization vectors from one or more CSV catalogs onto a FITS map.

Built for NGC 7538 (JCMT map + stellar polarimetry), but written generically so
you can drop in additional CSV files and compare their vectors on the same image.

------------------------------------------------------------------------------
HOW TO ADD ANOTHER CSV (the whole point of this script)
------------------------------------------------------------------------------
Scroll down to the `DATASETS` list near the bottom and append another
`Dataset(...)` entry. Each dataset can point at a different CSV, use different
column names, and gets its own color/label in the legend. Example:

    DATASETS = [
        Dataset("ngc7538_avg_R 2.csv", label="R band", color="cyan"),
        Dataset("ngc7538_avg_V.csv",   label="V band", color="yellow"),
    ]

------------------------------------------------------------------------------
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize, PercentileInterval, AsinhStretch


# =============================================================================
# CONFIGURATION
# =============================================================================

FITS_FILE = "ngc7538_jcmt.fits"     # background map
OUTPUT_FILE = "ngc7538_pol_vectors.png"

# Vector geometry / appearance defaults (can be overridden per-dataset).
VECTOR_SCALE = 3.0        # arcsec of vector length per 1% polarization
LINE_WIDTH = 1.4
MIN_P = None              # e.g. 0.5 to hide low-polarization noise; None = keep all
MAX_P = None              # e.g. 10  to hide unreliable high-P outliers; None = keep all
DRAW_LEGEND_SCALEBAR = True
LEGEND_P_PERCENT = 5.0    # reference polarization drawn in the scale bar


@dataclass
class Dataset:
    """One CSV catalog of polarization measurements to overlay."""
    path: str
    label: str = None
    color: str = "cyan"
    # Column names — change these if a new CSV uses different headers.
    ra_col: str = "ra_wcs"      # Right Ascension (deg)
    dec_col: str = "dec_wcs"    # Declination (deg)
    p_col: str = "P"            # polarization degree (percent)
    pa_col: str = "PA"          # position angle (deg, East of North)
    scale: float = None         # per-dataset length scale; None -> global VECTOR_SCALE
    linewidth: float = None
    min_p: float = None         # per-dataset override; None -> global MIN_P
    max_p: float = None         # per-dataset override; None -> global MAX_P

    def __post_init__(self):
        if self.label is None:
            self.label = Path(self.path).stem


# =============================================================================
# CORE PLOTTING LOGIC  (you normally don't need to touch below here)
# =============================================================================

def load_background(fits_file):
    """Return (image_data, WCS) from the FITS primary HDU."""
    with fits.open(fits_file) as hdul:
        data = hdul[0].data.astype(float)
        wcs = WCS(hdul[0].header)
    return data, wcs


def compute_vector_endpoints(ra, dec, p, pa_deg, scale_arcsec_per_pct):
    """
    Given arrays of RA/Dec (deg), polarization P (%), and position angle PA (deg,
    measured East of North), return the two endpoints of each vector in world
    coordinates. Vectors are centered on the star and drawn as half-length bars
    on each side (standard polarization pseudo-vector convention).

    Working in world coordinates and converting BOTH endpoints through the WCS
    means any map rotation / projection is handled correctly.
    """
    half_len_deg = (p * scale_arcsec_per_pct / 2.0) / 3600.0
    pa = np.radians(pa_deg)

    # PA is East-of-North: North = +Dec, East = +RA (which increases to the "left").
    # dDec = cos(PA)*len ; dRA (on sky) = sin(PA)*len, converted to coordinate RA
    # by dividing by cos(Dec).
    ddec = half_len_deg * np.cos(pa)
    dra = half_len_deg * np.sin(pa) / np.cos(np.radians(dec))

    ra1, dec1 = ra - dra, dec - ddec
    ra2, dec2 = ra + dra, dec + ddec
    return (ra1, dec1), (ra2, dec2)


def filter_rows(df, ds):
    """Apply min/max polarization cuts, drop NaNs in the columns we use."""
    cols = [ds.ra_col, ds.dec_col, ds.p_col, ds.pa_col]
    df = df.dropna(subset=cols).copy()

    lo = ds.min_p if ds.min_p is not None else MIN_P
    hi = ds.max_p if ds.max_p is not None else MAX_P
    if lo is not None:
        df = df[df[ds.p_col] >= lo]
    if hi is not None:
        df = df[df[ds.p_col] <= hi]
    return df


def overlay_dataset(ax, wcs, ds):
    """Read one CSV and draw its polarization vectors onto ax."""
    df = pd.read_csv(ds.path)
    df = filter_rows(df, ds)

    ra = df[ds.ra_col].to_numpy()
    dec = df[ds.dec_col].to_numpy()
    p = df[ds.p_col].to_numpy()
    pa = df[ds.pa_col].to_numpy()

    scale = ds.scale if ds.scale is not None else VECTOR_SCALE
    lw = ds.linewidth if ds.linewidth is not None else LINE_WIDTH

    (ra1, dec1), (ra2, dec2) = compute_vector_endpoints(ra, dec, p, pa, scale)

    # World -> pixel for both endpoints.
    x1, y1 = wcs.world_to_pixel_values(ra1, dec1)
    x2, y2 = wcs.world_to_pixel_values(ra2, dec2)

    # Draw each vector as a line segment.
    for i in range(len(x1)):
        ax.plot([x1[i], x2[i]], [y1[i], y2[i]],
                color=ds.color, lw=lw, solid_capstyle="round",
                zorder=5)

    # A single proxy line for the legend.
    ax.plot([], [], color=ds.color, lw=lw + 0.6, label=f"{ds.label} (n={len(df)})")
    return len(df)


def add_scalebar(ax, wcs, image_shape):
    """Draw a reference polarization vector so vector lengths are interpretable."""
    ny, nx = image_shape
    # Place near lower-left in pixel space; convert a fixed sky length there.
    x0, y0 = 0.12 * nx, 0.08 * ny
    ra0, dec0 = wcs.pixel_to_world_values(x0, y0)
    half_len_deg = (LEGEND_P_PERCENT * VECTOR_SCALE / 2.0) / 3600.0
    ra1, dec1 = ra0 - half_len_deg / np.cos(np.radians(dec0)), dec0
    ra2, dec2 = ra0 + half_len_deg / np.cos(np.radians(dec0)), dec0
    xa, ya = wcs.world_to_pixel_values(ra1, dec1)
    xb, yb = wcs.world_to_pixel_values(ra2, dec2)
    ax.plot([xa, xb], [ya, yb], color="white", lw=2.5, solid_capstyle="round", zorder=6)
    ax.text((xa + xb) / 2, ya + 0.02 * ny, f"{LEGEND_P_PERCENT:.0f}% pol.",
            color="white", ha="center", va="bottom", fontsize=9, zorder=6)


def main():
    data, wcs = load_background(FITS_FILE)

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection=wcs)

    # norm = ImageNormalize(data, interval=PercentileInterval(99.0),
    #                       stretch=AsinhStretch())
    im = ax.imshow(data, origin="lower", cmap="RdYlBu_r", vmin=-1, vmax=1.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("JCMT intensity")

    total = 0
    for ds in DATASETS:
        if not Path(ds.path).exists():
            print(f"  [skip] {ds.path} not found")
            continue
        n = overlay_dataset(ax, wcs, ds)
        print(f"  [ok]   {ds.label}: {n} vectors")
        total += n

    if DRAW_LEGEND_SCALEBAR:
        add_scalebar(ax, wcs, data.shape)

    ax.set_xlabel("Right Ascension (J2000)")
    ax.set_ylabel("Declination (J2000)")
    ax.set_title("NGC 7538 — polarization vectors on JCMT map")
    ax.legend(loc="upper right", framealpha=0.7, facecolor="black",
              labelcolor="white", fontsize=9)
    ax.grid(color="white", ls=":", alpha=0.3)

    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    print(f"\nSaved {OUTPUT_FILE}  ({total} vectors total)")


# =============================================================================
# DATASETS — ADD / EDIT ENTRIES HERE TO COMPARE MULTIPLE CSVs
# =============================================================================

DATASETS = [
    # "im = ax.imshow(data, origin=\"lower\", cmap=\"RdYlBu_r\",vmin=-1, vmax=1.5)\n",
    # "               # norm=AsinhNorm(linear_width=0.05, vmin=-1, vmax=1.5))\n",

    Dataset("ngc7538_avg_R 2.csv", label="R band", color="gray", p_col="P_avg", pa_col="PA_avg"),

    # To compare another catalog, uncomment and edit:
    # Dataset("another_catalog.csv", label="V band", color="yellow",
    #         p_col="P_avg", pa_col="PA_avg"),
]


if __name__ == "__main__":
    main()
