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

# We will run this script in polarimetry/ngc7538_analysis_v3/
# The raw data is in the parent directory '../'
RAW_DIR = ".."

OUT = Path("pol_out"); OUT.mkdir(exist_ok=True)
(OUT / "strips").mkdir(exist_ok=True)
(OUT / "cat").mkdir(exist_ok=True)

ANGLES = ["0", "22_5", "45", "67_5"]
ANG_VAL = {"0": 0.0, "22_5": 22.5, "45": 45.0, "67_5": 67.5}

TARGET_PREFIX = "NGC7538"                 # NGC7538_d{dither}-{seq}_{angle}.fit
DITHERS = ["d0", "d1", "d2"]
STD_POSITIONS = ["m1", "m2"]              # SIG6 = VI Cyg 12; m0/m3/m4 saturated
STD_NOMINAL = {"m1": (1804, 1428), "m2": (1804, 1428)}   # o-star approx position
STD_CATALOG = dict(P=7.893, PA=116.23)    # VI Cyg 12, R band (NOT/TURPOL; V: 8.947, 115.03)
CALIB_0927 = "../FITS/fixed_out/calib.json"   # q0,u0 from the 09-27 unpolarized standards
FALLBACK_Q0U0 = dict(q0=0.0043, u0=0.0288)    # values from that run, used if file missing

APER_O_R = 12.0                           # px, o-beam circular aperture
APER_E_A, APER_E_B = 50.0, 12.0           # px, e-beam ellipse (streak ~100 px long)
SKY_R1, SKY_R2 = 25, 40                   # sky annulus radii (both beams, scaled for e)
DET_NSIGMA = 5.0                          # detection threshold on smoothed image
DET_MINPIX = 6
SMOOTH_SIGMA = 2.0                        # px, gaussian matched filter
EDGE_MARGIN = 15
MATCH_TOL = 5.0                           # px, NN tolerance everywhere

# ---------- IO ----------
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
    """Strip-wise median of raw frames (bias/dark/flat masters)."""
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

# ---------- image ops ----------
def gauss_smooth(img, sigma=SMOOTH_SIGMA):
    """FFT gaussian smoothing (pure numpy)."""
    H, W = img.shape
    fy = np.fft.fftfreq(H)[:, None]; fx = np.fft.rfftfreq(W)[None, :]
    g = np.exp(-2*(np.pi*sigma)**2 * (fy**2 + fx**2))
    return np.fft.irfft2(np.fft.rfft2(img) * g, s=img.shape).astype(np.float32)

def label_regions(mask):
    """Connected components (8-conn) via BFS. Returns list of (ys, xs) arrays."""
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
    """Circular aperture + annulus median sky. Returns flux, sigma."""
    reg, px, py = _cut(data, cx, cy, int(r2)+2)
    yy, xx = np.mgrid[0:reg.shape[0], 0:reg.shape[1]]
    d = np.hypot(xx-px, yy-py)
    ap, sky = d <= r, (d >= r1) & (d <= r2)
    sky_med = np.median(reg[sky])
    sky_std = 1.4826*np.median(np.abs(reg[sky] - sky_med))
    n = ap.sum()
    return float(reg[ap].sum() - sky_med*n), float(sky_std*np.sqrt(n*(1+n/sky.sum())))

def phot_ellipse(data, cx, cy, a=APER_E_A, b=APER_E_B):
    """Horizontal ellipse aperture + elliptical annulus median sky (for the e streak)."""
    reg, px, py = _cut(data, cx, cy, int(2*a))
    yy, xx = np.mgrid[0:reg.shape[0], 0:reg.shape[1]]
    r_ell = np.hypot((xx-px)/a, (yy-py)/b)
    ap, sky = r_ell <= 1.0, (r_ell >= 1.3) & (r_ell <= 1.8)
    sky_med = np.median(reg[sky])
    sky_std = 1.4826*np.median(np.abs(reg[sky] - sky_med))
    n = ap.sum()
    return float(reg[ap].sum() - sky_med*n), float(sky_std*np.sqrt(n*(1+n/sky.sum())))

# ---------- matching ----------
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

# ---------- Stokes ----------
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
    print("Building master calibration frames...")
    bias = median_stack_raw(sorted(glob.glob(f"{RAW_DIR}/BIAS-000?_b1.fit")), "bias")
    dark40 = median_stack_raw(sorted(glob.glob(f"{RAW_DIR}/DARK-000?_D40.fit")), "dark40")
    print(f"bias: median={np.median(bias):.1f} | dark40: median={np.median(dark40):.1f}")
    
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
        print(f"Master flat created.")
        
    def load_reduced(fname):
        return (getdata(fname) - dark40) / FLAT
        
    print("Measuring O->E beam offset from standard star...")
    DX_PRIOR, DY_PRIOR, DX_WIN = 1377, 32, 60
    offs = []
    for ang in ANGLES:
        data = load_reduced(f"{RAW_DIR}/SIG6_m1-0001_{ang}.fit")
        cx, cy = centroid(data, *STD_NOMINAL["m1"], half=8)
        exc, eyc = int(cx + DX_PRIOR), int(cy + DY_PRIOR)
        ybandy = data[eyc-15:eyc+16, exc-DX_WIN:exc+DX_WIN].sum(axis=1)
        ybandy -= np.median(ybandy)
        dy = DY_PRIOR - 15 + float((np.arange(31)*np.clip(ybandy,0,None)).sum() / np.clip(ybandy,0,None).sum())
        
        band = data[int(cy+dy)-8:int(cy+dy)+9, exc-DX_WIN:exc+DX_WIN]
        prof = band.sum(axis=0)
        prof = prof - np.median(prof)
        k = 15
        run = np.convolve(prof, np.ones(k)/k, mode="same")
        pk = int(np.argmax(run))
        lo, hi = max(pk-60, 0), min(pk+60, len(prof))
        w = np.clip(prof[lo:hi], 0, None)
        dx = DX_PRIOR - DX_WIN + lo + float((np.arange(hi-lo)*w).sum()/w.sum())
        offs.append((dx, dy))
        
    BEAM_DX = float(np.median([o[0] for o in offs]))
    BEAM_DY = float(np.median([o[1] for o in offs]))
    print(f"Beam offset O->E = ({BEAM_DX:+.1f}, {BEAM_DY:+.1f}) px")
    
    # 09-27 calibration fallbacks
    c927 = dict(FALLBACK_Q0U0, eff=0.945, chi0=41.17)
    EFF, CHI0 = c927["eff"], c927["chi0"]
    json.dump(dict(q0=c927["q0"], u0=c927["u0"], eff=EFF, chi0=CHI0,
                   beam_dx=BEAM_DX, beam_dy=BEAM_DY), open(OUT/"calib.json", "w"), indent=2)

    def list_sci():
        out = []
        for f in sorted(glob.glob(f"{RAW_DIR}/{TARGET_PREFIX}_d*-000*_*.fit")):
            if "_d.fit" in f: continue  # exclude the pre-subtracted ones, we use raw files
            m = re.match(rf"{TARGET_PREFIX}_d(\d)-(\d{{4}})_(0|22_5|45|67_5)\.fit", os.path.basename(f))
            if m: out.append(dict(file=f, dither=int(m.group(1)), seq=int(m.group(2)), ang=m.group(3)))
        return out
        
    SCI = list_sci()
    print(f"Found {len(SCI)} science frames")
    
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
    
    # Bubble geometry
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
    print(f"o bubble ({O_BUB['cx']:.0f},{O_BUB['cy']:.0f}) r={O_BUB['r']:.0f} | "
          f"e bubble ({E_BUB['cx']:.0f},{E_BUB['cy']:.0f}) r={E_BUB['r']:.0f}")
          
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

    print("Running frame catalogs...")
    for rec in SCI:
        detect_and_phot(rec)
        
    print("Assembling cycles and computing Stokes parameters...")
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
        
    # Cross-cycle combine
    print("Cross-cycle combining...")
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
                        flux_o0=g.o_0.mean(), q=qc, u=uc, sq=sqc, su=suc,
                        P=P, sP=sP, P_debiased=Pd,
                        theta_sky=(th + CHI0) % 180.0, stheta=sth))
                        
    final = pd.DataFrame(res).sort_values("n_cycles", ascending=False).reset_index(drop=True)
    
    # 5. Gaia Astrometry
    print("Running Gaia DR3 pattern matching and affine refinement...")
    hdr0 = fits.getheader(SCI[0]["file"])
    h, m, s = [float(v) for v in hdr0["OBJCTRA"].split()]
    RA0 = 15*(h + m/60 + s/3600)
    dd, dm, ds = [float(v) for v in hdr0["OBJCTDEC"].replace("+", " ").split()]
    DEC0 = np.sign(dd if dd != 0 else 1)*(abs(dd) + dm/60 + ds/3600)
    PLATE = 0.3532
    RA0 += 49.0/3600/np.cos(np.radians(DEC0))
    
    # Check if local Gaia downloaded
    gaia_f = OUT / "gaia_dr3.csv"
    if not gaia_f.exists():
        import urllib.request, urllib.parse
        adql = ("SELECT ra,dec,phot_g_mean_mag,source_id,designation,random_index,ref_epoch,ra_error,dec_error,parallax,parallax_error,parallax_over_error,pm,pmra,pmra_error,pmdec,pmdec_error FROM gaiadr3.gaia_source WHERE 1=CONTAINS("
                f"POINT('ICRS',ra,dec),CIRCLE('ICRS',{RA0:.5f},{DEC0:.5f},0.08)) "
                "AND phot_g_mean_mag<19.5")
        url = ("https://gea.esac.esa.int/tap-server/tap/sync?REQUEST=doQuery&LANG=ADQL"
               "&FORMAT=csv&QUERY=" + urllib.parse.quote(adql))
        print("Downloading Gaia DR3...")
        urllib.request.urlretrieve(url, gaia_f)
        
    gaia = pd.read_csv(gaia_f)
    print(f"Gaia sources: {len(gaia)}")
    
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
    print(f"Pattern match: votes={votes} rotation={deg} deg offset=({ox:.1f}\",{oy:.1f}\")")
    
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
        
    rms = np.sqrt(((t[ok] - gsky[idx[ok]])**2).sum(1).mean())
    print(f"WCS RMS: {rms:.3f} arcsec, Matched stars: {int(ok.sum())}/{len(t)}")
    
    xi, eta = np.radians(t[:, 0]/3600), np.radians(t[:, 1]/3600)
    den = np.cos(d0) - eta*np.sin(d0)
    ra_coords = (np.degrees(a0 + np.arctan2(xi, den))) % 360
    dec_coords = np.degrees(np.arctan((np.sin(d0) + eta*np.cos(d0)) / np.hypot(xi, den)))
    
    final["ra_deg"], final["dec_deg"] = ra_coords, dec_coords
    final["gaia_matched"] = ok
    
    # 6. Reformat to exact custom format!
    # Let's map gaia columns
    final_with_gaia = pd.merge(final, gaia, left_on='src', right_index=True, how='left') # Approximate index map
    # A cleaner way is match by coordinates
    final_rows = []
    for j, row in final.iterrows():
        # Match to Gaia by coordinates
        sc = SkyCoord(row['ra_deg'], row['dec_deg'], unit='deg')
        gc = SkyCoord(gaia.ra, gaia.dec, unit='deg')
        idx, sep, _ = sc.match_to_catalog_sky(gc)
        
        out_row = dict(row)
        if sep.arcsec < 2.0:
            grow = gaia.iloc[idx]
            for col in gaia.columns:
                out_row[col] = grow[col]
            out_row['DESIGNATION'] = grow['designation']
            out_row['SOURCE_ID'] = grow['source_id']
            
        final_rows.append(out_row)
        
    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(OUT / "ngc7538_polarimetry.csv", index=False)
    print("Catalog saved to pol_out/ngc7538_polarimetry.csv")

if __name__ == "__main__":
    main()
