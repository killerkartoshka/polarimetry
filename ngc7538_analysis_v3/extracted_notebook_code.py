
# ==========================================
# CELL 0
# ==========================================
# === 1. CONFIG & helpers — run first ===
import glob, json, re, os
from pathlib import Path
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

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

# ---------- Stokes (identical conventions to qlook_kazakh_fixed) ----------
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

print("config OK")

# ==========================================
# CELL 1
# ==========================================
# === C1. Masters ===
bias = median_stack_raw(sorted(glob.glob("BIAS-000?_b1.fit")), "bias")
dark40 = median_stack_raw(sorted(glob.glob("DARK-000?_D40.fit")), "dark40")
print(f"bias: median={np.median(bias):.1f} | dark40: median={np.median(dark40):.1f} "
      f"(dark current ~{np.median(dark40)-np.median(bias):.1f} ADU/40s in the median pixel)")

flat_f = OUT / "flat_single.fits"
if flat_f.exists():
    FLAT = getdata(flat_f)
else:
    frames = [getdata(f) - bias for f in sorted(glob.glob("FLAT_d?-0001_*.fit"))]
    # each frame normalized by its own illuminated-median first (twilight level
    # changes fast), THEN median-combined -> pixel response only
    normed = []
    for fr in frames:
        illum = fr > 0.3*np.percentile(fr, 99)
        normed.append(fr / np.median(fr[illum]))
    FLAT = np.median(np.stack(normed), axis=0)
    illum = FLAT > 0.3
    FLAT /= np.median(FLAT[illum])
    FLAT = np.clip(FLAT, 0.05, None).astype(np.float32)   # avoid /0 outside bubbles
    fits.writeto(flat_f, FLAT, overwrite=True)
    print(f"single flat from {len(frames)} frames (all angles+pointings), "
          f"illuminated fraction={illum.mean():.2f}")

def load_reduced(fname, ang=None):
    """(raw - dark40) / FLAT -- the SAME flat for every HWP angle (see above)."""
    return (getdata(fname) - dark40) / FLAT

# ==========================================
# CELL 2
# ==========================================
# === C2. Standard star: beam offset, then eff & chi0 ===
import pandas as pd

# measure the o->e offset from the m1 position, all 4 angles of cycle 1.
# The e-image is a ~100 px streak and the frames contain hot pixels, so:
# collapse a +-8 px band in y around the expected dy into a 1D x-profile
# (a hot pixel hits one column; the streak spans ~100), smooth the profile,
# and take the flux-weighted centroid around its peak.
DX_PRIOR, DY_PRIOR, DX_WIN = 1377, 32, 60
offs = []
for ang in ANGLES:
    data = load_reduced(f"SIG6_m1-0001_{ang}.fit", ang)
    cx, cy = centroid(data, *STD_NOMINAL["m1"], half=8)   # small box: crowded field
    # refine dy: y-profile of the e region (columns +- 60 around prior)
    exc, eyc = int(cx + DX_PRIOR), int(cy + DY_PRIOR)
    ybandy = data[eyc-15:eyc+16, exc-DX_WIN:exc+DX_WIN].sum(axis=1)
    ybandy -= np.median(ybandy)
    dy = DY_PRIOR - 15 + float((np.arange(31)*np.clip(ybandy,0,None)).sum()
                               / np.clip(ybandy,0,None).sum())
    # refine dx: x-profile of a +-8 px band around that dy
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
scat = np.std([o[0] for o in offs])
print(f"beam offset (o->e) = ({BEAM_DX:+.1f}, {BEAM_DY:+.1f}) px  (dx scatter {scat:.1f})")
assert abs(BEAM_DX - DX_PRIOR) < 50, "offset far from prior -- check the e region visually"

# --- standard-star measurement, with the two failure modes detected ---
rows = []
for pos in ["m0", "m1", "m2", "m3", "m4"]:
    # is the standard present and clean at this position?
    d0 = fits.getdata(f"SIG6_{pos}-0001_0.fit").astype(np.float32)
    nsat = int((d0 > 64000).sum())
    if nsat > 0:
        print(f"{pos}: standard SATURATED ({nsat} px at 65535) -> unusable"); continue
    # no saturation anywhere: at 40 s VI Cyg 12 *must* saturate if present,
    # so the star missed the field-stop hole in this pointing
    print(f"{pos}: no saturated pixels in frame -> standard NOT in the hole, skipped")

for pos in []:            # <- becomes usable positions on a night with good data
    for cyc in [1, 2, 3]:
        ph = {}
        for ang in ANGLES:
            data = load_reduced(f"SIG6_{pos}-000{cyc}_{ang}.fit", ang)
            cx, cy = centroid(data, *STD_NOMINAL[pos], half=8)
            fo, so = phot_circle(data, cx, cy)
            fe, se = phot_ellipse(data, cx + BEAM_DX, cy + BEAM_DY)
            ph[ang] = (fo, so, fe, se)
        q, u, sq, su = stokes_double_ratio(
            ph["0"][0], ph["0"][2], ph["45"][0], ph["45"][2],
            ph["22_5"][0], ph["22_5"][2], ph["67_5"][0], ph["67_5"][2],
            ph["0"][1], ph["0"][3], ph["45"][1], ph["45"][3],
            ph["22_5"][1], ph["22_5"][3], ph["67_5"][1], ph["67_5"][3])
        rows.append(dict(pos=pos, cycle=cyc, q=q, u=u, sq=sq, su=su))

# --- assemble the night's calibration ---
try:
    c927 = json.load(open(CALIB_0927))
    print(f"\n09-27 calibration loaded: q0={c927['q0']:+.4f} u0={c927['u0']:+.4f} "
          f"eff={c927['eff']:.3f} chi0={c927['chi0']:.2f}")
except FileNotFoundError:
    c927 = dict(FALLBACK_Q0U0, eff=0.945, chi0=41.17)
    print(f"\nWARNING: {CALIB_0927} not found, using hard-coded 09-27 values")

if rows:                  # same-night standard available -> derive eff, chi0
    std = pd.DataFrame(rows)
    wq, wu = 1/std.sq**2, 1/std.su**2
    qm = (std.q*wq).sum()/wq.sum(); um = (std.u*wu).sum()/wu.sum()
    qc, uc = qm - c927["q0"], um - c927["u0"]
    Pm = np.hypot(qc, uc); thm = np.degrees(0.5*np.arctan2(uc, qc)) % 180
    EFF = 100*Pm / STD_CATALOG["P"]
    CHI0 = (STD_CATALOG["PA"] - thm) % 180.0
    print(f"VI Cyg 12: P={100*Pm:.2f}% -> eff={EFF:.3f}, chi0={CHI0:.2f}")
else:                     # this dataset: fall back entirely to 09-27
    EFF, CHI0 = c927["eff"], c927["chi0"]
    print("WARNING: no usable same-night standard -> eff and chi0 taken from 09-27.")
    print("         Valid only if the HWP zero-point / camera rotation were not")
    print("         changed between the nights (beam offset DID move 1417->1377 px,")
    print("         so something was adjusted -- treat absolute PA with caution).")

json.dump(dict(q0=c927["q0"], u0=c927["u0"], eff=EFF, chi0=CHI0,
               beam_dx=BEAM_DX, beam_dy=BEAM_DY), open(OUT/"calib.json", "w"), indent=2)

# ==========================================
# CELL 3
# ==========================================
# === C3. Master, bubble geometry, per-frame detection + photometry ===
def list_sci():
    out = []
    for f in sorted(glob.glob(f"{TARGET_PREFIX}_d?-00??_*.fit")):
        m = re.match(rf"{TARGET_PREFIX}_d(\d)-(\d{{4}})_(0|22_5|45|67_5)\.fit", os.path.basename(f))
        if m: out.append(dict(file=f, dither=int(m.group(1)), seq=int(m.group(2)), ang=m.group(3)))
    return out

SCI = list_sci()
print(f"{len(SCI)} science frames")

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
        print(f"master strip {si+1}/8")
    fits.writeto(master_f, np.vstack(strips).astype(np.float32), overwrite=True)
MASTER = getdata(master_f)

# bubble geometry from the FLAT (crisp illumination edge; the flat-corrected
# master is flattened inside the bubbles, so blob-finding on it is unreliable)
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
    """One frame -> star list with o and e photometry (cached CSV).
    Detection runs on (reduced - master) * flat: multiplying the flat back in
    makes the noise uniform again (flat division inflates noise wherever the
    flat is small, e.g. near the bubble edge, which caused false detections).
    Photometry stays on the flat-corrected image."""
    out = OUT / "cat" / (Path(rec["file"]).stem + ".csv")
    if out.exists(): return pd.read_csv(out)
    data = load_reduced(rec["file"], rec["ang"]) - MASTER
    x0 = int(O_BUB["cx"] - O_BUB["r"]) ; x1 = int(O_BUB["cx"] + O_BUB["r"])
    y0 = int(O_BUB["cy"] - O_BUB["r"]) ; y1 = int(O_BUB["cy"] + O_BUB["r"])
    crop = data[y0:y1, x0:x1]
    det = crop * FLAT[y0:y1, x0:x1]        # uniform-noise image
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
        # real star: several UNsmoothed pixels above the raw noise
        if (det[ys_, xs_] > 3.5*sig_raw).sum() < 4: continue
        fo, so = phot_circle(data, cx, cy)
        fe, se = phot_ellipse(data, cx + BEAM_DX, cy + BEAM_DY)
        rows.append(dict(x=cx, y=cy, o=fo, oe=so, e=fe, ee=se,
                         peak=float(det[ys_, xs_].max()), npix=len(ys_)))
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return df

for rec in SCI:
    n = len(detect_and_phot(rec))
    print(f"{os.path.basename(rec['file'])}: {n} stars", end="  \r")
print("\nper-frame catalogs done -> pol_out/cat/")

# ==========================================
# CELL 4
# ==========================================
# === C4. Cycle assembly + per-cycle Stokes ===
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
    print(f"d{dith} seq{seq}: {len(merged)} stars at all 4 angles")

# ==========================================
# CELL 5
# ==========================================
# === C5. Cross-cycle combine + calibration ===
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
    print(f"d{df.dither.iloc[0]} seq{df.seq.iloc[0]}: shift=({sx:+.1f},{sy:+.1f}) "
          f"votes={votes} matched={int(ok.sum())}/{len(df)}")
    allrows.append(df)
    prev, prev_xy = df, xy + [sx, sy]

long = pd.concat(allrows, ignore_index=True)
long.to_csv(OUT / "all_cycles_long.csv", index=False)

cal = json.load(open(OUT / "calib.json"))
res = []
for sid, g in long.groupby("src"):
    wq, wu = 1/g.sq**2, 1/g.su**2
    qm, um = (g.q*wq).sum()/wq.sum(), (g.u*wu).sum()/wu.sum()
    n = len(g)
    sqm = max(1/np.sqrt(wq.sum()), g.q.std(ddof=1)/np.sqrt(n) if n > 1 else np.inf)
    sum_ = max(1/np.sqrt(wu.sum()), g.u.std(ddof=1)/np.sqrt(n) if n > 1 else np.inf)
    qc, uc = (qm - cal["q0"])/(cal["eff"]), (um - cal["u0"])/(cal["eff"])
    sqc, suc = sqm/cal["eff"], sum_/cal["eff"]
    P, sP, Pd, th, sth = pol_from_qu(qc, uc, sqc, suc)
    res.append(dict(src=sid, n_cycles=n, x=g.x.iloc[0], y=g.y.iloc[0],
                    flux_o0=g.o_0.mean(), q=qc, u=uc, sq=sqc, su=suc,
                    P=P, sP=sP, P_debiased=Pd,
                    theta_sky=(th + cal["chi0"]) % 180.0, stheta=sth))
final = pd.DataFrame(res).sort_values("n_cycles", ascending=False).reset_index(drop=True)
final.to_csv(OUT / "ngc7538_polarimetry.csv", index=False)
good = final[final.n_cycles >= 5]
print(f"\n{len(final)} unique stars, {len(good)} in >=5 cycles")
print(good.head(20)[["src", "n_cycles", "x", "y", "P_debiased", "sP",
                     "theta_sky", "stheta"]].round(4).to_string(index=False))

# ==========================================
# CELL 6
# ==========================================
# === C6. Vector map over the o-bubble of the master frame ===
final = pd.read_csv(OUT / "ngc7538_polarimetry.csv")
sel = final[(final.P_debiased/final.sP >= 3) & (final.n_cycles >= 3)]

x0, x1 = int(O_BUB["cx"]-O_BUB["r"])-20, int(O_BUB["cx"]+O_BUB["r"])+20
y0, y1 = int(O_BUB["cy"]-O_BUB["r"])-20, int(O_BUB["cy"]+O_BUB["r"])+20
img = MASTER[y0:y1, x0:x1]
vmin, vmax = np.nanpercentile(img, [5, 99])

fig, ax = plt.subplots(figsize=(9, 9))
ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
          extent=[x0, x1, y0, y1])
ang = np.radians(sel.theta_sky.values)
L = sel.P_debiased.values * 800                     # 5% pol -> 40 px
ax.quiver(sel.x, sel.y, -L*np.sin(ang), L*np.cos(ang), color="red",
          scale=1, scale_units="xy", angles="xy", width=0.004,
          headlength=0, headwidth=0, headaxislength=0, pivot="mid")
ax.quiver(x0+60, y0+40, 0.05*800, 0, color="yellow", scale=1, scale_units="xy",
          angles="xy", width=0.005, headlength=0, headwidth=0, headaxislength=0, pivot="mid")
ax.text(x0+60, y0+55, "5%", color="yellow", fontsize=13)
ax.set_title(f"NGC 7538: {len(sel)} stars, calibrated PA (chi0={json.load(open(OUT/'calib.json'))['chi0']:.1f})")
ax.set_xlabel("X [px]"); ax.set_ylabel("Y [px]")
plt.tight_layout(); plt.show(); plt.close()
print(f"median P (debiased) of plotted stars: {100*np.median(sel.P_debiased):.2f}%")
print("catalog -> pol_out/ngc7538_polarimetry.csv")

# ==========================================
# CELL 7
# ==========================================
# === C7. Gaia DR3 astrometric solution for the star catalog ===
import urllib.request, urllib.parse

hdr0 = fits.getheader(SCI[0]["file"])
h, m, s = [float(v) for v in hdr0["OBJCTRA"].split()]
RA0 = 15*(h + m/60 + s/3600)
dd, dm, ds = [float(v) for v in hdr0["OBJCTDEC"].replace("+", " ").split()]
DEC0 = np.sign(dd if dd != 0 else 1)*(abs(dd) + dm/60 + ds/3600)
# !! PLATE SCALE: the header XPIXSZ/FOCALLEN gives 0.1406"/px, but that is WRONG --
# a focal reducer is in the train (measured eff. focal length 5255 mm, not 13200 mm).
# Measured from 3 reference stars (TYC 4279-1349-1, TYC 4279-1463-1,
# 2MASS J23133024+6130103) identified in the field: 0.3532"/px, isotropic to 0.03%,
# rotation +5.6 deg, parity -1; field stop center is ~49" E of the header pointing.
PLATE = 0.3532                                             # arcsec / px (MEASURED)
RA0 += 49.0/3600/np.cos(np.radians(DEC0))                  # correct pointing offset
print(f"field center (corrected): {RA0:.4f} {DEC0:+.4f}  plate scale {PLATE:.4f} \"/px")

gaia_f = OUT / "gaia_dr3.csv"
if not gaia_f.exists():
    adql = ("SELECT ra,dec,phot_g_mean_mag FROM gaiadr3.gaia_source WHERE 1=CONTAINS("
            f"POINT('ICRS',ra,dec),CIRCLE('ICRS',{RA0:.5f},{DEC0:.5f},0.08)) "
            "AND phot_g_mean_mag<19.5")
    url = ("https://gea.esac.esa.int/tap-server/tap/sync?REQUEST=doQuery&LANG=ADQL"
           "&FORMAT=csv&QUERY=" + urllib.parse.quote(adql))
    print("downloading Gaia DR3 ...\n" + url)
    urllib.request.urlretrieve(url, gaia_f)
gaia = pd.read_csv(gaia_f)
print(f"{len(gaia)} Gaia sources in the cone")

# gnomonic projection of Gaia onto the tangent plane at (RA0, DEC0), arcsec
a, d = np.radians(gaia.ra.values), np.radians(gaia.dec.values)
a0, d0 = np.radians(RA0), np.radians(DEC0)
D = np.sin(d0)*np.sin(d) + np.cos(d0)*np.cos(d)*np.cos(a - a0)
gxi = 206265*np.cos(d)*np.sin(a - a0)/D
geta = 206265*(np.cos(d0)*np.sin(d) - np.sin(d0)*np.cos(d)*np.cos(a - a0))/D
gsky = np.column_stack([gxi, geta])

final = pd.read_csv(OUT / "ngc7538_polarimetry.csv")
xy = final[["x", "y"]].values
ctr = xy.mean(axis=0)

best = None                        # rotation x parity grid, translation vote
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
print(f"pattern match: votes={votes} parity={par} rotation={deg} deg offset=({ox:.0f}\",{oy:.0f}\")")
assert votes >= 8, "pattern match too weak -- check RA0/DEC0 and the Gaia cone"

# affine refinement (2 passes)
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
print(f"matched {int(ok.sum())}/{len(t)} stars, WCS RMS = {rms:.2f} arcsec")

# inverse gnomonic: tangent-plane (arcsec) -> RA/Dec (deg)
xi, eta = np.radians(t[:, 0]/3600), np.radians(t[:, 1]/3600)
den = np.cos(d0) - eta*np.sin(d0)
ra = (np.degrees(a0 + np.arctan2(xi, den))) % 360
dec = np.degrees(np.arctan((np.sin(d0) + eta*np.cos(d0)) / np.hypot(xi, den)))
final["ra_deg"], final["dec_deg"] = ra, dec
final["gaia_matched"] = ok
final.to_csv(OUT / "ngc7538_polarimetry.csv", index=False)
print("ra_deg/dec_deg written to pol_out/ngc7538_polarimetry.csv")
final[["src", "n_cycles", "ra_deg", "dec_deg", "P_debiased", "theta_sky"]].head(10).round(5)
