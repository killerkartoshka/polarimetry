import os
import glob
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings
warnings.filterwarnings('ignore')

RAW_DIR = ".."
OUT = Path("pol_out")
OUT.mkdir(exist_ok=True)
(OUT / "strips").mkdir(exist_ok=True)
(OUT / "cat").mkdir(exist_ok=True)

ANGLES = ["0", "22_5", "45", "67_5"]
TARGET_PREFIX = "NGC7538"
STD_NOMINAL = {"m1": (1804, 1428), "m2": (1804, 1428)}
FALLBACK_Q0U0 = dict(q0=0.0043, u0=0.0288)

APER_O_R = 12.0
APER_E_A, APER_E_B = 50.0, 12.0
SKY_R1, SKY_R2 = 25, 40
DET_NSIGMA = 5.0
DET_MINPIX = 6
SMOOTH_SIGMA = 2.0
EDGE_MARGIN = 15
MATCH_TOL = 5.0

# Precise lists of columns required in the final output
req_columns = [
    "X_IMAGE","Y_IMAGE","FLAGS",
    "X_IMAGE_O0","Y_IMAGE_O0","FLUX_O0","FLUXERR_O0","FLUXMAX_O0","D2D_O0","KRON_RADIUS_O0","APERTURE_O0",
    "X_IMAGE_E0","Y_IMAGE_E0","FLUX_E0","FLUXERR_E0","FLUXMAX_E0","D2D_E0","KRON_RADIUS_E0","APERTURE_E0",
    "X_IMAGE_O4","Y_IMAGE_O4","FLUX_O4","FLUXERR_O4","FLUXMAX_O4","D2D_O4","KRON_RADIUS_O4","APERTURE_O4",
    "X_IMAGE_E4","Y_IMAGE_E4","FLUX_E4","FLUXERR_E4","FLUXMAX_E4","D2D_E4","KRON_RADIUS_E4","APERTURE_E4",
    "X_IMAGE_O2","Y_IMAGE_O2","FLUX_O2","FLUXERR_O2","FLUXMAX_O2","D2D_O2","KRON_RADIUS_O2","APERTURE_O2",
    "X_IMAGE_E2","Y_IMAGE_E2","FLUX_E2","FLUXERR_E2","FLUXMAX_E2","D2D_E2","KRON_RADIUS_E2","APERTURE_E2",
    "X_IMAGE_O6","Y_IMAGE_O6","FLUX_O6","FLUXERR_O6","FLUXMAX_O6","D2D_O6","KRON_RADIUS_O6","APERTURE_O6",
    "X_IMAGE_E6","Y_IMAGE_E6","FLUX_E6","FLUXERR_E6","FLUXMAX_E6","D2D_E6","KRON_RADIUS_E6","APERTURE_E6",
    "FLUX_ALL","q","qe","u","ue","p","ep","pa","epa","P","eP","PA","ePA","band","dither",
    "ra_wcs","dec_wcs","data_index","distortion_ra","distortion_dec","ra_offset","dec_offset","ra_mod","dec_mod","ang_dist",
    "solution_id","DESIGNATION","random_index","ref_epoch","ra","ra_error","dec","dec_error",
    "parallax","parallax_error","parallax_over_error","pm","pmra","pmra_error","pmdec","pmdec_error",
    "ra_dec_corr","ra_parallax_corr","ra_pmra_corr","ra_pmdec_corr","dec_parallax_corr","dec_pmra_corr",
    "dec_pmdec_corr","parallax_pmra_corr","parallax_pmdec_corr","pmra_pmdec_corr","astrometric_n_obs_al",
    "astrometric_n_obs_ac","astrometric_n_good_obs_al","astrometric_n_bad_obs_al","astrometric_gof_al",
    "astrometric_chi2_al","astrometric_excess_noise","astrometric_excess_noise_sig","astrometric_params_solved",
    "astrometric_primary_flag","nu_eff_used_in_astrometry","pseudocolour","pseudocolour_error",
    "ra_pseudocolour_corr","dec_pseudocolour_corr","parallax_pseudocolour_corr","pmra_pseudocolour_corr",
    "pmdec_pseudocolour_corr","astrometric_matched_transits","visibility_periods_used","astrometric_sigma5d_max",
    "matched_transits","new_matched_transits","matched_transits_removed","ipd_gof_harmonic_amplitude",
    "ipd_gof_harmonic_phase","ipd_frac_multi_peak","ipd_frac_odd_win","ruwe","scan_direction_strength_k1",
    "scan_direction_strength_k2","scan_direction_strength_k3","scan_direction_strength_k4","scan_direction_mean_k1",
    "scan_direction_mean_k2","scan_direction_mean_k3","scan_direction_mean_k4","duplicated_source","phot_g_n_obs",
    "phot_g_mean_flux","phot_g_mean_flux_error","phot_g_mean_flux_over_error","phot_g_mean_mag","phot_bp_n_obs",
    "phot_bp_mean_flux","phot_bp_mean_flux_error","phot_bp_mean_flux_over_error","phot_bp_mean_mag","phot_rp_n_obs",
    "phot_rp_mean_flux","phot_rp_mean_flux_error","phot_rp_mean_flux_over_error","phot_rp_mean_mag",
    "phot_bp_rp_excess_factor","phot_bp_n_contaminated_transits","phot_bp_n_blended_transits",
    "phot_rp_n_contaminated_transits","phot_rp_n_blended_transits","phot_proc_mode","bp_rp","bp_g","g_rp",
    "radial_velocity","radial_velocity_error","rv_method_used","rv_nb_transits","rv_nb_deblended_transits",
    "rv_visibility_periods_used","rv_expected_sig_to_noise","rv_renormalised_gof","rv_chisq_pvalue","rv_time_duration",
    "rv_amplitude_robust","rv_template_teff","rv_template_logg","rv_template_fe_h","rv_atm_param_origin","vbroad",
    "vbroad_error","vbroad_nb_transits","grvs_mag","grvs_mag_error","grvs_mag_nb_transits","rvs_spec_sig_to_noise",
    "phot_variable_flag","l","b","ecl_lon","ecl_lat","in_qso_candidates","in_galaxy_candidates","non_single_star",
    "has_xp_continuous","has_xp_sampled","has_rvs","has_epoch_photometry","has_epoch_rv","has_mcmc_gspphot",
    "has_mcmc_msc","in_andromeda_survey","classprob_dsc_combmod_quasar","classprob_dsc_combmod_galaxy",
    "classprob_dsc_combmod_star","teff_gspphot","teff_gspphot_lower","teff_gspphot_upper","logg_gspphot",
    "logg_gspphot_lower","logg_gspphot_upper","mh_gspphot","mh_gspphot_lower","mh_gspphot_upper","distance_gspphot",
    "distance_gspphot_lower","distance_gspphot_upper","azero_gspphot","azero_gspphot_lower","azero_gspphot_upper",
    "ag_gspphot","ag_gspphot_lower","ag_gspphot_upper","ebpminrp_gspphot","ebpminrp_gspphot_lower",
    "ebpminrp_gspphot_upper","libname_gspphot","r_med_geo","r_lo_geo","r_hi_geo","r_med_photogeo","r_lo_photogeo",
    "r_hi_photogeo","flag","Bmag","Rmag","Vmag","Icmag","Ksmag","Hmag","Jmag","flux_var","SOURCE_ID",
    "q_starnum","q_starnum_init","q_avg","qe_avg","u_starnum","u_starnum_init","u_avg","ue_avg","p_avg","ep_avg",
    "pa_avg","epa_avg","P_avg","eP_avg","PA_avg","ePA_avg"
]

def getdata(path):
    return fits.getdata(path).astype(np.float32)

def raw_strip(path, y0, y1, width):
    with open(path, "rb") as f:
        off = 0
        while True:
            block = f.read(2880); off += 2880
            if any(block[i:i+8] == b"END     " for i in range(0, 2880, 80)):
                break
    raw = np.memmap(path, dtype=">i2", mode="r", offset=off)
    return raw[y0*width:y1*width].reshape(y1-y0, width).astype(np.float32) + 32768.0

def median_stack_raw(files, tag, n_strips=6):
    out = OUT / f"stack_{tag}.fits"
    if out.exists(): return getdata(out)
    hdr = fits.getheader(files[0]); H, W = hdr["NAXIS2"], hdr["NAXIS1"]
    edges = np.linspace(0, H, n_strips+1).astype(int)
    strips = []
    for si in range(n_strips):
        cache = OUT / "strips" / f"{tag}_{si}.npy"
        if cache.exists(): strips.append(np.load(cache)); continue
        st = np.median(np.stack([raw_strip(f, edges[si], edges[si+1], W) for f in files]), axis=0)
        np.save(cache, st); strips.append(st)
    m = np.vstack(strips).astype(np.float32)
    fits.writeto(out, m, overwrite=True)
    return m

def gauss_smooth(img, sigma=SMOOTH_SIGMA):
    H, W = img.shape
    fy = np.fft.fftfreq(H)[:, None]; fx = np.fft.rfftfreq(W)[None, :]
    g = np.exp(-2*(np.pi*sigma)**2 * (fy**2 + fx**2))
    return np.fft.irfft2(np.fft.rfft2(img) * g, s=img.shape).astype(np.float32)

def label_regions(mask):
    todo = set(zip(*np.where(mask)))
    comps = []
    while todo:
        seed = todo.pop(); stack = [seed]; comp = [seed]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    p = (y+dy, x+dx)
                    if p in todo:
                        todo.discard(p); stack.append(p); comp.append(p)
        comps.append((np.array([c[0] for c in comp]), np.array([c[1] for c in comp])))
    return comps

def centroid(data, cx, cy, half=15, n_iter=3):
    for _ in range(n_iter):
        x1, y1 = int(round(cx))-half, int(round(cy))-half
        reg = data[max(y1,0):y1+2*half, max(x1,0):x1+2*half]
        reg = np.clip(reg - np.median(reg), 0, None)
        tot = reg.sum()
        if tot <= 0: return cx, cy
        yy, xx = np.mgrid[0:reg.shape[0], 0:reg.shape[1]]
        cx, cy = max(x1,0) + (xx*reg).sum()/tot, max(y1,0) + (yy*reg).sum()/tot
    return cx, cy

def _cut(data, cx, cy, half):
    x1, y1 = int(round(cx))-half, int(round(cy))-half
    return data[y1:y1+2*half+1, x1:x1+2*half+1], cx-x1, cy-y1

def phot_circle(data, cx, cy, r=APER_O_R, r1=SKY_R1, r2=SKY_R2):
    reg, px, py = _cut(data, cx, cy, int(r2)+2)
    yy, xx = np.mgrid[0:reg.shape[0], 0:reg.shape[1]]
    d = np.hypot(xx-px, yy-py)
    ap, sky = d <= r, (d >= r1) & (d <= r2)
    sky_med = np.median(reg[sky])
    sky_std = 1.4826*np.median(np.abs(reg[sky] - sky_med))
    n = ap.sum()
    return float(reg[ap].sum() - sky_med*n), float(sky_std*np.sqrt(n*(1+n/sky.sum())))

def phot_ellipse(data, cx, cy, a=APER_E_A, b=APER_E_B):
    reg, px, py = _cut(data, cx, cy, int(2*a))
    yy, xx = np.mgrid[0:reg.shape[0], 0:reg.shape[1]]
    r_ell = np.hypot((xx-px)/a, (yy-py)/b)
    ap, sky = r_ell <= 1.0, (r_ell >= 1.3) & (r_ell <= 1.8)
    sky_med = np.median(reg[sky])
    sky_std = 1.4826*np.median(np.abs(reg[sky] - sky_med))
    n = ap.sum()
    return float(reg[ap].sum() - sky_med*n), float(sky_std*np.sqrt(n*(1+n/sky.sum())))

def hough_shift(src_xy, dst_xy, bin_px=4.0, win=4.0, n_iter=3):
    if len(src_xy) == 0 or len(dst_xy) == 0:
        return 0.0, 0.0, 0
    dx = (dst_xy[:, 0][:, None] - src_xy[:, 0][None, :]).ravel()
    dy = (dst_xy[:, 1][:, None] - src_xy[:, 1][None, :]).ravel()
    key = np.round(dx/bin_px).astype(np.int64)*1_000_003 + np.round(dy/bin_px).astype(np.int64)
    vals, counts = np.unique(key, return_counts=True)
    m = key == vals[np.argmax(counts)]
    sx, sy = dx[m].mean(), dy[m].mean()
    for _ in range(n_iter):
        m = (np.abs(dx-sx) < win) & (np.abs(dy-sy) < win)
        if m.sum() < 3: break
        sx, sy = dx[m].mean(), dy[m].mean()
    return sx, sy, int(counts.max())

def nn_match(src_xy, dst_xy, tol=MATCH_TOL):
    if len(dst_xy) == 0 or len(src_xy) == 0:
        n = len(src_xy); return np.zeros(n, bool), np.zeros(n, int)
    d2 = ((src_xy[:, None, :] - dst_xy[None, :, :])**2).sum(-1)
    idx = d2.argmin(1)
    return np.sqrt(d2[np.arange(len(src_xy)), idx]) < tol, idx

def stokes_double_ratio(o0, e0, o45, e45, o22, e22, o67, e67,
                        so0, se0, so45, se45, so22, se22, so67, se67):
    Rq = np.sqrt((o0/e0)/(o45/e45)); Ru = np.sqrt((o22/e22)/(o67/e67))
    q = (Rq-1)/(Rq+1); u = (Ru-1)/(Ru+1)
    rq = 0.5*np.sqrt((so0/o0)**2+(se0/e0)**2+(so45/o45)**2+(se45/e45)**2)
    ru = 0.5*np.sqrt((so22/o22)**2+(se22/e22)**2+(so67/o67)**2+(se67/e67)**2)
    return q, u, 2*Rq*rq/(Rq+1)**2, 2*Ru*ru/(Ru+1)**2

def pol_from_qu(q, u, sq, su):
    P = np.hypot(q, u)
    sP = np.sqrt((q*sq)**2 + (u*su)**2)/P if P > 0 else np.nan
    Pd = np.sqrt(max(P**2 - sP**2, 0.0)) if P > sP else 0.0
    th = np.degrees(0.5*np.arctan2(u, q)) % 180.0
    sth = np.degrees(0.5*sP/P) if P > 0 else np.nan
    return P, sP, Pd, th, sth

def main():
    print("CELL 1: Calibration (Bias, Dark, Flats)...")
    bias = median_stack_raw(sorted(glob.glob(f"{RAW_DIR}/BIAS-000?_b1.fit")), "bias")
    dark40 = median_stack_raw(sorted(glob.glob(f"{RAW_DIR}/DARK-000?_D40.fit")), "dark40")
    
    flat_f = OUT / "flat_single.fits"
    if flat_f.exists():
        FLAT = getdata(flat_f)
    else:
        frames = [getdata(f) - bias for f in sorted(glob.glob(f"{RAW_DIR}/FLAT_d?-0001_*.fit"))]
        normed = []
        for fr in frames:
            illum = fr > 0.3*np.percentile(fr, 99)
            normed.append(fr / np.median(fr[illum]))
        FLAT = np.median(np.stack(normed), axis=0)
        illum = FLAT > 0.3
        FLAT /= np.median(FLAT[illum])
        FLAT = np.clip(FLAT, 0.05, None).astype(np.float32)
        fits.writeto(flat_f, FLAT, overwrite=True)
        
    def load_reduced(fname):
        return (getdata(fname) - dark40) / FLAT
        
    print("CELL 2: Beam Offset O->E from VI Cyg 12 standard...")
    offs = []
    for ang in ["0", "22_5", "45", "67_5"]:
        data = load_reduced(f"{RAW_DIR}/SIG6_m1-0001_{ang}.fit")
        cx, cy = centroid(data, *STD_NOMINAL["m1"], half=8)
        exc, eyc = int(cx + 1377), int(cy + 32)
        ybandy = data[eyc-15:eyc+16, exc-60:exc+60].sum(axis=1)
        ybandy -= np.median(ybandy)
        dy = 32 - 15 + float((np.arange(31)*np.clip(ybandy,0,None)).sum() / np.clip(ybandy,0,None).sum())
        
        band = data[int(cy+dy)-8:int(cy+dy)+9, exc-60:exc+60]
        prof = band.sum(axis=0)
        prof = prof - np.median(prof)
        k = 15
        run = np.convolve(prof, np.ones(k)/k, mode="same")
        pk = int(np.argmax(run))
        lo, hi = max(pk-60, 0), min(pk+60, len(prof))
        w = np.clip(prof[lo:hi], 0, None)
        dx = 1377 - 60 + lo + float((np.arange(hi-lo)*w).sum()/w.sum())
        offs.append((dx, dy))
        
    BEAM_DX = float(np.median([o[0] for o in offs]))
    BEAM_DY = float(np.median([o[1] for o in offs]))
    print(f"  Fitted Beam Offset: dx={BEAM_DX:+.2f}, dy={BEAM_DY:+.2f} px")
    
    # Assembly calibration Zero points
    c927 = dict(FALLBACK_Q0U0, eff=0.945, chi0=41.17)
    EFF, CHI0 = c927["eff"], c927["chi0"]
    
    def list_sci():
        out = []
        for f in sorted(glob.glob(f"{RAW_DIR}/{TARGET_PREFIX}_d*-000*_*.fit")):
            if "_d.fit" in f: continue
            m = re.match(rf"{TARGET_PREFIX}_d(\d)-(\d{{4}})_(0|22_5|45|67_5)\.fit", os.path.basename(f))
            if m: out.append(dict(file=f, dither=int(m.group(1)), seq=int(m.group(2)), ang=m.group(3)))
        return out
        
    SCI = list_sci()
    
    # Master stacked frame
    master_f = OUT / "master_reduced.fits"
    if not master_f.exists():
        hdr = fits.getheader(SCI[0]["file"]); H, W = hdr["NAXIS2"], hdr["NAXIS1"]
        edges = np.linspace(0, H, 9).astype(int)
        strips = []
        for si in range(8):
            cache = OUT / "strips" / f"masterred_{si}.npy"
            if cache.exists(): strips.append(np.load(cache)); continue
            st = np.stack([(raw_strip(s["file"], edges[si], edges[si+1], W)
                            - dark40[edges[si]:edges[si+1]]) / FLAT[edges[si]:edges[si+1]]
                           for s in SCI])
            st = np.median(st, axis=0); np.save(cache, st); strips.append(st)
        fits.writeto(master_f, np.vstack(strips).astype(np.float32), overwrite=True)
        
    MASTER = getdata(master_f)
    
    # Find bubble locations
    b = 8; H, W = MASTER.shape
    fl = FLAT
    sm = np.median(fl[:H//b*b, :W//b*b].reshape(H//b, b, W//b, b), axis=(1, 3))
    ys, xs = np.where(sm > 0.5)
    order = np.argsort(xs); xs, ys = xs[order], ys[order]
    split = np.split(np.arange(len(xs)), np.where(np.diff(xs) > 5)[0] + 1)
    blobs = sorted([dict(cx=float(xs[s].mean()*b), cy=float(ys[s].mean()*b),
                         r=float((xs[s].max()-xs[s].min())*b/2))
                    for s in split if len(s) > 200], key=lambda d: d["cx"])
    O_BUB, E_BUB = blobs
    
    # Frame catalogs cached load/build
    def detect_and_phot(rec):
        out = OUT / "cat" / (Path(rec["file"]).stem + ".csv")
        if out.exists(): return pd.read_csv(out)
        data = load_reduced(rec["file"]) - MASTER
        x0 = int(O_BUB["cx"] - O_BUB["r"]) ; x1 = int(O_BUB["cx"] + O_BUB["r"])
        y0 = int(O_BUB["cy"] - O_BUB["r"]) ; y1 = int(O_BUB["cy"] + O_BUB["r"])
        crop = data[y0:y1, x0:x1]
        det = crop * FLAT[y0:y1, x0:x1]
        sig_raw = 1.4826*np.median(np.abs(det - np.median(det)))
        smc = gauss_smooth(det)
        sig = 1.4826*np.median(np.abs(smc - np.median(smc)))
        mask = smc > DET_NSIGMA*sig
        rows = []
        for ys_, xs_ in label_regions(mask):
            if len(ys_) < DET_MINPIX: continue
            w = np.clip(smc[ys_, xs_], 0, None)
            cx = (xs_*w).sum()/w.sum() + x0
            cy = (ys_*w).sum()/w.sum() + y0
            if np.hypot(cx-O_BUB["cx"], cy-O_BUB["cy"]) > O_BUB["r"] - EDGE_MARGIN: continue
            if (det[ys_, xs_] > 3.5*sig_raw).sum() < 4: continue
            fo, so = phot_circle(data, cx, cy)
            fe, se = phot_ellipse(data, cx + BEAM_DX, cy + BEAM_DY)
            rows.append(dict(x=cx, y=cy, o=fo, oe=so, e=fe, ee=se,
                             peak=float(det[ys_, xs_].max()), npix=len(ys_)))
        df = pd.DataFrame(rows)
        df.to_csv(out, index=False)
        return df

    print("CELL 3: Processing frame-by-frame science catalogs...")
    for rec in SCI:
        detect_and_phot(rec)
        
    print("CELL 4: Assembling cycles and computing Stokes parameters...")
    cycles = []
    for (dith, seq) in sorted({(s["dither"], s["seq"]) for s in SCI}):
        dfs = {}
        for s in SCI:
            if s["dither"] == dith and s["seq"] == seq:
                dfs[s["ang"]] = detect_and_phot(s)
        if len(dfs) < 4: continue
        ref = dfs["0"]; ref_xy = ref[["x", "y"]].values
        merged = ref.rename(columns={c: f"{c}_0" for c in ["o", "oe", "e", "ee"]})[
            ["x", "y", "o_0", "oe_0", "e_0", "ee_0"]].copy()
        for ang in ["22_5", "45", "67_5"]:
            oth = dfs[ang]; xy = oth[["x", "y"]].values
            if len(xy) >= 3:
                sx, sy, _ = hough_shift(xy, ref_xy)
            else: sx = sy = 0.0
            ok, idx = nn_match(ref_xy, xy + [sx, sy])
            for c in ["o", "oe", "e", "ee"]:
                merged[f"{c}_{ang}"] = np.where(ok, oth[c].values[idx], np.nan)
        merged = merged.dropna().reset_index(drop=True)
        q, u, sq, su = stokes_double_ratio(
            merged.o_0, merged.e_0, merged.o_45, merged.e_45,
            merged.o_22_5, merged.e_22_5, merged.o_67_5, merged.e_67_5,
            merged.oe_0, merged.ee_0, merged.oe_45, merged.ee_45,
            merged.oe_22_5, merged.ee_22_5, merged.oe_67_5, merged.ee_67_5)
        merged["q"], merged["u"], merged["sq"], merged["su"] = q, u, sq, su
        merged["dither"], merged["seq"] = dith, seq
        cycles.append(merged)
        
    print("CELL 5: Combining across cycles...")
    prev = cycles[0].copy(); prev["src"] = np.arange(len(prev))
    prev_xy = prev[["x", "y"]].values; next_id = len(prev)
    allrows = [prev]
    for df in cycles[1:]:
        xy = df[["x", "y"]].values
        sx, sy, votes = hough_shift(xy, prev_xy)
        ok, idx = nn_match(xy + [sx, sy], prev_xy)
        df = df.copy()
        df["src"] = np.where(ok, prev["src"].values[np.where(ok, idx, 0)], -1)
        nnew = int((~ok).sum())
        df.loc[~ok, "src"] = next_id + np.arange(nnew); next_id += nnew
        allrows.append(df)
        prev, prev_xy = df, xy + [sx, sy]
        
    long = pd.concat(allrows, ignore_index=True)
    
    res = []
    for sid, g in long.groupby("src"):
        wq, wu = 1/g.sq**2, 1/g.su**2
        qm, um = (g.q*wq).sum()/wq.sum(), (g.u*wu).sum()/wu.sum()
        n = len(g)
        sqm = max(1/np.sqrt(wq.sum()), g.q.std(ddof=1)/np.sqrt(n) if n > 1 else np.inf)
        sum_ = max(1/np.sqrt(wu.sum()), g.u.std(ddof=1)/np.sqrt(n) if n > 1 else np.inf)
        qc, uc = (qm - c927["q0"])/(EFF), (um - c927["u0"])/(EFF)
        sqc, suc = sqm/EFF, sum_/EFF
        P, sP, Pd, th, sth = pol_from_qu(qc, uc, sqc, suc)
        res.append(dict(src=sid, n_cycles=n, x=g.x.iloc[0], y=g.y.iloc[0],
                        flux_o0=g.o_0.mean(), q_avg=qc, u_avg=uc, qe_avg=sqc, ue_avg=suc,
                        P_avg=P*100.0, eP_avg=sP*100.0, P_debiased_avg=Pd*100.0,
                        PA_avg=(th + CHI0) % 180.0, ePA_avg=sth))
                        
    final = pd.DataFrame(res).sort_values("n_cycles", ascending=False).reset_index(drop=True)
    
    print("CELL 7: Querying Gaia DR3 for Astrometry...")
    hdr0 = fits.getheader(SCI[0]["file"])
    h, m, s = [float(v) for v in hdr0["OBJCTRA"].split()]
    RA0 = 15*(h + m/60 + s/3600)
    dd, dm, ds = [float(v) for v in hdr0["OBJCTDEC"].replace("+", " ").split()]
    DEC0 = np.sign(dd if dd != 0 else 1)*(abs(dd) + dm/60 + ds/3600)
    PLATE = 0.3532
    RA0 += 49.0/3600/np.cos(np.radians(DEC0))
    
    # Load or pull Gaia
    gaia_f = OUT / "gaia_dr3.csv"
    if not gaia_f.exists():
        import urllib.request, urllib.parse
        adql = ("SELECT ra,dec,phot_g_mean_mag,source_id,designation,random_index,ref_epoch,ra_error,dec_error,parallax,parallax_error,parallax_over_error,pm,pmra,pmra_error,pmdec,pmdec_error FROM gaiadr3.gaia_source WHERE 1=CONTAINS("
                f"POINT('ICRS',ra,dec),CIRCLE('ICRS',{RA0:.5f},{DEC0:.5f},0.08)) "
                "AND phot_g_mean_mag<19.5")
        url = ("https://gea.esac.esa.int/tap-server/tap/sync?REQUEST=doQuery&LANG=ADQL"
               "&FORMAT=csv&QUERY=" + urllib.parse.quote(adql))
        urllib.request.urlretrieve(url, gaia_f)
        
    gaia = pd.read_csv(gaia_f)
    
    # Solve WCS manually using tangent plane projection
    a, d = np.radians(gaia.ra.values), np.radians(gaia.dec.values)
    a0, d0 = np.radians(RA0), np.radians(DEC0)
    D = np.sin(d0)*np.sin(d) + np.cos(d0)*np.cos(d)*np.cos(a - a0)
    gxi = 206265*np.cos(d)*np.sin(a - a0)/D
    geta = 206265*(np.cos(d0)*np.sin(d) - np.sin(d0)*np.cos(d)*np.cos(a - a0))/D
    gsky = np.column_stack([gxi, geta])
    
    xy = final[["x", "y"]].values
    ctr = xy.mean(axis=0)
    
    best = None
    for par in (1, -1):
        Pm = np.diag([par, 1.0])
        for deg in range(360):
            th = np.radians(deg)
            Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            t = (PLATE*(Rm @ Pm @ (xy - ctr).T)).T
            ddx = gsky[:, 0][:, None] - t[:, 0][None, :]
            ddy = gsky[:, 1][:, None] - t[:, 1][None, :]
            key = (np.round(ddx.ravel()/3.0).astype(np.int64)*1_000_003
                   + np.round(ddy.ravel()/3.0).astype(np.int64))
            vals, cnt = np.unique(key, return_counts=True)
            if best is None or cnt.max() > best[0]:
                mm = key == vals[np.argmax(cnt)]
                best = (int(cnt.max()), par, deg, ddx.ravel()[mm].mean(), ddy.ravel()[mm].mean())
                
    votes, par, deg, ox, oy = best
    th = np.radians(deg)
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    t = (PLATE*(Rm @ np.diag([par, 1.0]) @ (xy - ctr).T)).T + [ox, oy]
    
    for _ in range(2):
        d2 = ((t[:, None, :] - gsky[None, :, :])**2).sum(-1)
        idx = d2.argmin(1)
        ok = np.sqrt(d2[np.arange(len(t)), idx]) < 2.5
        A = np.column_stack([xy, np.ones(len(xy))])
        coef, *_ = np.linalg.lstsq(A[ok], gsky[idx[ok]], rcond=None)
        t = A @ coef
        
    xi, eta = np.radians(t[:, 0]/3600), np.radians(t[:, 1]/3600)
    den = np.cos(d0) - eta*np.sin(d0)
    ra_coords = (np.degrees(a0 + np.arctan2(xi, den))) % 360
    dec_coords = np.degrees(np.arctan((np.sin(d0) + eta*np.cos(d0)) / np.hypot(xi, den)))
    
    final["ra_deg"] = ra_coords
    final["dec_deg"] = dec_coords
    final["gaia_matched"] = ok
    
    # -------------------------------------------------------------
    # BUILD FINAL RICH FORMATTED CSV (One row per star per dither)
    # -------------------------------------------------------------
    print("Formatting final output rows matching the requested format...")
    out_rows = []
    
    for i, row in long.iterrows():
        sid = row['src']
        dither_cycle = f"d{row['dither']}-000{row['seq']}"
        
        # Get matching averages
        avg = final[final['src'] == sid].iloc[0]
        
        out_row = {c: np.nan for c in req_columns}
        
        # Coordinates & dither
        out_row['X_IMAGE'] = row['x']
        out_row['Y_IMAGE'] = row['y']
        out_row['dither'] = dither_cycle
        out_row['band'] = 'R' # standard band
        out_row['FLAGS'] = 0
        
        # Positions & fluxes for individual angles (O and E)
        out_row['X_IMAGE_O0'] = row['x']
        out_row['Y_IMAGE_O0'] = row['y']
        out_row['FLUX_O0'] = row['o_0']
        out_row['FLUXERR_O0'] = row['oe_0']
        out_row['APERTURE_O0'] = APER_O_R
        
        out_row['X_IMAGE_E0'] = row['x'] + BEAM_DX
        out_row['Y_IMAGE_E0'] = row['y'] + BEAM_DY
        out_row['FLUX_E0'] = row['e_0']
        out_row['FLUXERR_E0'] = row['ee_0']
        out_row['APERTURE_E0'] = APER_E_A  # major axis
        
        out_row['X_IMAGE_O2'] = row['x'] # approximate dither coordinates
        out_row['Y_IMAGE_O2'] = row['y']
        out_row['FLUX_O2'] = row['o_22_5']
        out_row['FLUXERR_O2'] = row['oe_22_5']
        out_row['APERTURE_O2'] = APER_O_R
        
        out_row['X_IMAGE_E2'] = row['x'] + BEAM_DX
        out_row['Y_IMAGE_E2'] = row['y'] + BEAM_DY
        out_row['FLUX_E2'] = row['e_22_5']
        out_row['FLUXERR_E2'] = row['ee_22_5']
        out_row['APERTURE_E2'] = APER_E_A
        
        out_row['X_IMAGE_O4'] = row['x']
        out_row['Y_IMAGE_O4'] = row['y']
        out_row['FLUX_O4'] = row['o_45']
        out_row['FLUXERR_O4'] = row['oe_45']
        out_row['APERTURE_O4'] = APER_O_R
        
        out_row['X_IMAGE_E4'] = row['x'] + BEAM_DX
        out_row['Y_IMAGE_E4'] = row['y'] + BEAM_DY
        out_row['FLUX_E4'] = row['e_45']
        out_row['FLUXERR_E4'] = row['ee_45']
        out_row['APERTURE_E4'] = APER_E_A
        
        out_row['X_IMAGE_O6'] = row['x']
        out_row['Y_IMAGE_O6'] = row['y']
        out_row['FLUX_O6'] = row['o_67_5']
        out_row['FLUXERR_O6'] = row['oe_67_5']
        out_row['APERTURE_O6'] = APER_O_R
        
        out_row['X_IMAGE_E6'] = row['x'] + BEAM_DX
        out_row['Y_IMAGE_E6'] = row['y'] + BEAM_DY
        out_row['FLUX_E6'] = row['e_67_5']
        out_row['FLUXERR_E6'] = row['ee_67_5']
        out_row['APERTURE_E6'] = APER_E_A
        
        out_row['FLUX_ALL'] = (row['o_0'] + row['o_22_5'] + row['o_45'] + row['o_67_5'] +
                               row['e_0'] + row['e_22_5'] + row['e_45'] + row['e_67_5'])
                               
        # Single cycle Stokes (Fractions)
        qc, uc = (row['q'] - c927["q0"])/(EFF), (row['u'] - c927["u0"])/(EFF)
        sqc, suc = row['sq']/EFF, row['su']/EFF
        Pi, sPi, Pdi, thi, sthi = pol_from_qu(qc, uc, sqc, suc)
        
        out_row['q'] = qc
        out_row['qe'] = sqc
        out_row['u'] = uc
        out_row['ue'] = suc
        
        # Single cycle polarizations (Percentages)
        out_row['p'] = Pi * 100.0
        out_row['ep'] = sPi * 100.0
        out_row['P'] = Pdi * 100.0
        out_row['eP'] = sPi * 100.0
        out_row['pa'] = (thi + CHI0) % 180.0
        out_row['epa'] = sthi
        out_row['PA'] = out_row['pa']
        out_row['ePA'] = sthi
        
        # Averages (P in Percentages, q/u in Fractions)
        out_row['q_avg'] = avg['q_avg']
        out_row['qe_avg'] = avg['qe_avg']
        out_row['u_avg'] = avg['u_avg']
        out_row['ue_avg'] = avg['ue_avg']
        out_row['p_avg'] = avg['P_avg']
        out_row['ep_avg'] = avg['eP_avg']
        out_row['P_avg'] = avg['P_debiased_avg']
        out_row['eP_avg'] = avg['eP_avg']
        out_row['pa_avg'] = avg['PA_avg']
        out_row['epa_avg'] = avg['ePA_avg']
        out_row['PA_avg'] = avg['PA_avg']
        out_row['ePA_avg'] = avg['ePA_avg']
        
        out_row['q_starnum'] = int(avg['n_cycles'])
        out_row['u_starnum'] = int(avg['n_cycles'])
        
        # Coordinates
        out_row['ra_wcs'] = avg['ra_deg']
        out_row['dec_wcs'] = avg['dec_deg']
        
        # Match to Gaia to fill Gaia metadata
        sc = SkyCoord(avg['ra_deg'], avg['dec_deg'], unit='deg')
        gc = SkyCoord(gaia.ra, gaia.dec, unit='deg')
        idx, sep, _ = sc.match_to_catalog_sky(gc)
        if sep.arcsec < 2.0:
            grow = gaia.iloc[idx]
            for col in gaia.columns:
                if col in req_columns:
                    out_row[col] = grow[col]
            out_row['DESIGNATION'] = grow['designation']
            out_row['SOURCE_ID'] = grow['source_id']
            
        out_rows.append(out_row)
        
    out_df = pd.DataFrame(out_rows, columns=req_columns)
    out_df.to_csv("formatted_v3_polarimetry_catalog.csv", index=False)
    print(f"Catalog formatting complete! Saved {len(out_df)} rows to formatted_v3_polarimetry_catalog.csv")

if __name__ == "__main__":
    main()
