import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

def find_v3_matches():
    df_kanata = pd.read_csv("../compare/ngc7538_avg_R 2.csv")
    df_v3 = pd.read_csv("formatted_v3_polarimetry_catalog.csv")
    
    # In formatted_v3_polarimetry_catalog.csv, we have one row per star per dither.
    # We should unique them by SOURCE_ID or coordinates to get the unique stars.
    df_v3_unique = df_v3.drop_duplicates(subset=['ra_wcs', 'dec_wcs']).copy()
    
    c_k = SkyCoord(ra=df_kanata['ra_wcs'], dec=df_kanata['dec_wcs'], unit=(u.deg, u.deg))
    c_m = SkyCoord(ra=df_v3_unique['ra_wcs'], dec=df_v3_unique['dec_wcs'], unit=(u.deg, u.deg))
    
    idx, sep, _ = c_m.match_to_catalog_sky(c_k)
    matches = sep.arcsec < 2.0
    print(f"Matched {matches.sum()} stars within 2.0 arcsec.")

if __name__ == "__main__":
    find_v3_matches()
