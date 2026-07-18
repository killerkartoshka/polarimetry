import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from matplotlib.collections import LineCollection
import warnings
warnings.filterwarnings('ignore')

FITS_FILE = "ngc7538_jcmt.fits"
OUTPUT_FILE = "ngc7538_matched_comparison.png"

def compute_vector_endpoints(ra, dec, p, pa_deg, scale_arcsec_per_pct=3.0):
    half_len_deg = (p * scale_arcsec_per_pct / 2.0) / 3600.0
    pa = np.radians(pa_deg)
    ddec = half_len_deg * np.cos(pa)
    dra = half_len_deg * np.sin(pa) / np.cos(np.radians(dec))
    ra1, dec1 = ra - dra, dec - ddec
    ra2, dec2 = ra + dra, dec + ddec
    return (ra1, dec1), (ra2, dec2)

def main():
    # 1. Load FITS background
    with fits.open(FITS_FILE) as hdul:
        ext = 0 if hdul[0].data is not None else 1
        data = np.squeeze(hdul[ext].data.astype(float))
        wcs = WCS(hdul[ext].header).celestial

    # 2. Read catalogs
    df_kanata = pd.read_csv("ngc7538_avg_R 2.csv")
    # df_v3_all = pd.read_csv("formatted_v3_polarimetry_catalog.csv")
    df_v3_all = pd.read_csv("formatted_v3_polarimetry_catalog.csv")
    df_v3 = df_v3_all.drop_duplicates(subset=['ra_wcs', 'dec_wcs']).copy()

    # 3. Match coordinates within 2 arcseconds
    c_k = SkyCoord(ra=df_kanata['ra_wcs'], dec=df_kanata['dec_wcs'], unit=(u.deg, u.deg))
    c_m = SkyCoord(ra=df_v3['ra_wcs'], dec=df_v3['dec_wcs'], unit=(u.deg, u.deg))

    idx, sep, _ = c_m.match_to_catalog_sky(c_k)
    mask = sep.arcsec < 2.0

    matched_v3 = df_v3[mask].copy()
    matched_kanata = df_kanata.iloc[idx[mask]].copy()

    print(f"Found {len(matched_v3)} matched stars!")

    # Extract values (converting V3 values to percentage like Kanata)
    ra = matched_v3['ra_wcs'].to_numpy()
    dec = matched_v3['dec_wcs'].to_numpy()

    # Kanata values
    p_k = matched_kanata['P_avg'].to_numpy()
    pa_k = matched_kanata['PA_avg'].to_numpy()

    # Pipeline V3 values
    # In formatted_v3_polarimetry_catalog, P_avg is already percentage
    p_v3 = matched_v3['P_avg'].to_numpy()
    pa_v3 = matched_v3['PA_avg'].to_numpy()

    # Save the comparison data to a CSV so the user has the numbers
    comp_df = pd.DataFrame({
        'ra': ra,
        'dec': dec,
        'P_Kanata(%)': p_k,
        'P_V3(%)': p_v3,
        'PA_Kanata(deg)': pa_k,
        'PA_V3(deg)': pa_v3,
        'P_diff(%)': p_v3 - p_k,
        'PA_diff(deg)': (pa_v3 - pa_k + 90) % 180 - 90
    }).dropna()  # drop rows with NaNs to get clean stats

    # Extract clean arrays
    clean_p_k = comp_df['P_Kanata(%)'].to_numpy()
    clean_p_v3 = comp_df['P_V3(%)'].to_numpy()
    clean_pa_k = comp_df['PA_Kanata(deg)'].to_numpy()
    clean_pa_v3 = comp_df['PA_V3(deg)'].to_numpy()

    # 4. Multi-panel plotting
    fig = plt.figure(figsize=(16, 8))

    # Subplot 1: Vector Overlay Map (Left)
    ax_map = fig.add_subplot(121, projection=wcs)
    # vmin, vmax = np.nanpercentile(data, [5, 99.5])
    im = ax_map.imshow(data, origin="lower", cmap="RdYlBu_r", vmin=-1, vmax=1.5)
    cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
    cbar.set_label("JCMT Intensity")

    # Plot Kanata vectors for matched stars
    (ra1_k, dec1_k), (ra2_k, dec2_k) = compute_vector_endpoints(comp_df['ra'].to_numpy(), comp_df['dec'].to_numpy(), clean_p_k, clean_pa_k, scale_arcsec_per_pct=5.0)
    x1_k, y1_k = wcs.world_to_pixel_values(ra1_k, dec1_k)
    x2_k, y2_k = wcs.world_to_pixel_values(ra2_k, dec2_k)

    segments_k = [[(x1, y1), (x2, y2)] for x1, y1, x2, y2 in zip(x1_k, y1_k, x2_k, y2_k)]
    lc_k = LineCollection(segments_k, color='blue', lw=1, alpha=0.8, label="Kanata (Benchmark)", zorder=5)
    ax_map.add_collection(lc_k)

    # Plot Pipeline V3 vectors for matched stars
    (ra1_v3, dec1_v3), (ra2_v3, dec2_v3) = compute_vector_endpoints(comp_df['ra'].to_numpy(), comp_df['dec'].to_numpy(), clean_p_v3, clean_pa_v3, scale_arcsec_per_pct=5.0)
    x1_v3, y1_v3 = wcs.world_to_pixel_values(ra1_v3, dec1_v3)
    x2_v3, y2_v3 = wcs.world_to_pixel_values(ra2_v3, dec2_v3)

    segments_v3 = [[(x1, y1), (x2, y2)] for x1, y1, x2, y2 in zip(x1_v3, y1_v3, x2_v3, y2_v3)]
    lc_v3 = LineCollection(segments_v3, color='red', lw=1, label="Pipeline V3 (Dual Beam)", zorder=6)
    # lc_v3 = LineCollection(segments_v3, color='red', lw=1, label="Pipeline V1", zorder=6)
    ax_map.add_collection(lc_v3)

    # Draw reference scale vector (5%)
    x0, y0 = 0.12 * data.shape[1], 0.08 * data.shape[0]
    ra0, dec0 = wcs.pixel_to_world_values(x0, y0)
    half_len_deg = (5.0 * 5.0 / 2.0) / 3600.0
    ra1_ref, dec1_ref = ra0 - half_len_deg / np.cos(np.radians(dec0)), dec0
    ra2_ref, dec2_ref = ra0 + half_len_deg / np.cos(np.radians(dec0)), dec0
    xa, ya = wcs.world_to_pixel_values(ra1_ref, dec1_ref)
    xb, yb = wcs.world_to_pixel_values(ra2_ref, dec2_ref)
    ax_map.plot([xa, xb], [ya, yb], color="white", lw=2.5, solid_capstyle="round", zorder=7)
    ax_map.text((xa + xb) / 2, ya + 0.02 * data.shape[0], "5% pol.",
                color="white", ha="center", va="bottom", fontsize=10, fontweight='bold', zorder=7)

    # Center map on the targets
    avg_x, avg_y = wcs.world_to_pixel_values(comp_df['ra'].mean(), comp_df['dec'].mean())
    # ax_map.set_xlim(avg_x - 350, avg_x + 350)
    # ax_map.set_ylim(avg_y - 350, avg_y + 350)

    ax_map.set_xlabel("Right Ascension (J2000)")
    ax_map.set_ylabel("Declination (J2000)")
    ax_map.set_title("Direct Vector Overlay (Matched Stars Only)")
    ax_map.legend(loc="upper right", facecolor="black", labelcolor="white")
    ax_map.grid(color="white", ls=":", alpha=0.3)

    # Subplot 2: Numerical Correlation (Right)
    ax_corr = fig.add_subplot(122)

    # 2a. P scatter
    ax_corr.scatter(clean_p_k, clean_p_v3, color='blue', edgecolor='black', s=80, label='Polarization Degree P (%)', zorder=5)
    # 2b. PA scatter
    ax_corr.scatter(clean_pa_k, clean_pa_v3, color='red', edgecolor='black', marker='s', s=80, label='Position Angle PA (deg)', zorder=5)

    # Draw 1:1 line for reference
    lims = [0, 180]
    ax_corr.plot(lims, lims, color='gray', linestyle='--', alpha=0.7, label='1:1 Identity', zorder=1)

    ax_corr.set_xlim(-5, 185)
    ax_corr.set_ylim(-5, 185)
    ax_corr.set_xlabel("Kanata Benchmark (Value)")
    ax_corr.set_ylabel("Pipeline V3 (Value)")
    # ax_corr.set_ylabel("Pipeline V1 (Value)")
    ax_corr.set_title("Numerical Correlation (P & PA)")
    ax_corr.legend(loc="upper left")
    ax_corr.grid(True, linestyle=':', alpha=0.5)

    # Add stats as text
    p_corr = np.corrcoef(clean_p_k, clean_p_v3)[0, 1] if len(clean_p_k) > 1 else 0
    pa_corr = np.corrcoef(clean_pa_k, clean_pa_v3)[0, 1] if len(clean_pa_k) > 1 else 0
    stats_text = (
        f"Matched Stars: {len(comp_df)}\n"
        f"P correlation: {p_corr:.3f}\n"
        f"PA correlation: {pa_corr:.3f}\n"
        f"Mean P diff: {np.mean(comp_df['P_diff(%)']):.2f}%\n"
        f"Mean PA diff: {np.mean(comp_df['PA_diff(deg)']):.2f} deg"
    )
    ax_corr.text(0.95, 0.05, stats_text, transform=ax_corr.transAxes,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
                 ha='right', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
