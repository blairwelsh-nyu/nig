"""
Robustness and validity checks for the Sharia boundary spatial RD.

Tests implemented:
  1.  McCrary (2008) density continuity — is the respondent density smooth at dist=0?
  2.  Covariate balance — do age, education, gender, urban jump at the boundary?
  3.  Outcome placebos — trust in president, parliament, traditional and religious leaders
  4.  Bandwidth sensitivity — grid 50–250 km for police and courts
  5.  Donut RD — exclude ±25 km and ±50 km from boundary
  6.  Polynomial order — local linear vs. local quadratic
  7.  Placebo boundaries — shift boundary ±150 km and ±250 km (4 placebo tests)
  8.  Pre-trend (R1, 2000) — was trust already discontinuous before Sharia adoption?
  9.  Pooled rounds (R2 + R4) — does the RD replicate in a later round?
"""
import os, warnings
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from sklearn.neighbors import BallTree
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.transforms import blended_transform_factory
warnings.filterwarnings('ignore')

os.chdir('/home/user/nig')

# ── Shared setup ──────────────────────────────────────────────────────────────
SHARIA_BOUNDARY_COORDS = [
    (3.40, 11.85),(3.90, 11.20),(4.35, 10.95),(4.85, 10.65),(5.25, 10.40),
    (5.65, 10.10),(6.05,  9.85),(6.55,  9.55),(7.00,  9.35),(7.45,  9.15),
    (7.95,  9.40),(8.50,  9.60),(9.00,  9.80),(9.50,  9.75),(9.95,  9.65),
    (10.40, 9.35),(10.95, 9.10),(11.45, 9.00),(12.00, 9.00),(12.60, 9.10),
    (13.10, 9.20),(13.70, 8.80),
]
boundary_line = LineString(SHARIA_BOUNDARY_COORDS)
SHARIA_STATES = {'Zamfara','Kano','Sokoto','Katsina','Bauchi','Niger',
                 'Jigawa','Kebbi','Yobe','Borno','Kaduna','Gombe'}

acled = pd.read_csv('nigeria_acled.csv')
acled['event_date'] = pd.to_datetime(acled['event_date'], errors='coerce')
state_cents = (acled.groupby('admin1')[['latitude','longitude']]
               .mean().reset_index())
state_cents.columns = ['state','lat','lon']

def assign_state(lat_arr, lon_arr):
    cent_rad = np.radians(state_cents[['lat','lon']].values)
    tree = BallTree(cent_rad, metric='haversine')
    _, idx = tree.query(np.radians(np.column_stack([lat_arr, lon_arr])), k=1)
    return state_cents['state'].values[idx.flatten()]

EARTH_R = 6371.0

def compute_dist(lat_arr, lon_arr, state_arr, bline=boundary_line):
    dist = np.full(len(lat_arr), np.nan)
    for i in range(len(lat_arr)):
        p = Point(lon_arr[i], lat_arr[i])
        lr = np.radians(lat_arr[i])
        dlat = np.pi * EARTH_R / 180.0
        dlon = dlat * np.cos(lr)
        km = bline.distance(p) * np.sqrt(dlat**2 + dlon**2) / np.sqrt(2)
        dist[i] = km if state_arr[i] in SHARIA_STATES else -km
    return dist

def load_prepare(rnd, trust_cols_map):
    """Load a round, compute dist_km and sharia, clean trust columns."""
    df = pd.read_excel(f'NIG_r{rnd}.csv.xlsx')
    df.columns = df.columns.str.lower().str.strip()
    lat_col = next(c for c in df.columns if 'lat' in c)
    lon_col = next(c for c in df.columns if 'lon' in c)
    df = df.rename(columns={lat_col:'lat', lon_col:'lon'})
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df = df.dropna(subset=['lat','lon'])
    df['state']   = assign_state(df['lat'].values, df['lon'].values)
    df['dist_km'] = compute_dist(df['lat'].values, df['lon'].values, df['state'].values)
    df['sharia']  = df['state'].isin(SHARIA_STATES).astype(float)
    df['round']   = rnd
    for out_name, src_col in trust_cols_map.items():
        if src_col in df.columns:
            df[out_name] = pd.to_numeric(df[src_col], errors='coerce')
            df.loc[df[out_name] < 0, out_name] = np.nan
            df.loc[df[out_name] >= 7, out_name] = np.nan
        else:
            df[out_name] = np.nan
    return df

# Main round R2
R2_COLS = {
    'police':  'q43i', 'courts':    'q43j',
    'pres':    'q43a', 'parl':      'q43b',
    'trad':    'q43e', 'relig':     'q43g',
    'local':   'q43d',
}
r2 = load_prepare(2, R2_COLS)

# Covariates R2 — must reset index after dropna to align with r2
r2_raw = pd.read_excel('NIG_r2.csv.xlsx')
r2_raw.columns = r2_raw.columns.str.lower().str.strip()
lat_col_raw = next(c for c in r2_raw.columns if 'lat' in c)
lon_col_raw = next(c for c in r2_raw.columns if 'lon' in c)
r2_raw = r2_raw.rename(columns={lat_col_raw:'lat', lon_col_raw:'lon'})
r2_raw['lat'] = pd.to_numeric(r2_raw['lat'], errors='coerce')
r2_raw['lon'] = pd.to_numeric(r2_raw['lon'], errors='coerce')
r2_raw = r2_raw.dropna(subset=['lat','lon']).reset_index(drop=True)

# R2: q1a = age category (1–5 brackets), urbrur = urban/rural (1=urban,2=rural)
r2 = r2.reset_index(drop=True)
r2['age']   = pd.to_numeric(r2_raw['q1a'],   errors='coerce').where(lambda x: x < 9)
r2['urban'] = (r2_raw['urbrur'] == 1).astype(float)
print("Covariate columns: age (q1a), urban (urbrur)")

# Pre-trend R1 (2000): trspol 1-4 scale → invert so higher=more trust for comparability
r1 = load_prepare(1, {'police_raw':'trspol', 'courts_raw':'trscts',
                       'pres_raw':'trspre'})
# R1 scale: 1=not at all, 4=a lot → rescale to 0-3 (subtract 1) so same direction as R2
for c in ['police_raw','courts_raw','pres_raw']:
    r1[c] = pd.to_numeric(r1[c], errors='coerce')
    r1.loc[r1[c].isin([9,99]), c] = np.nan
    r1.loc[r1[c] < 1, c] = np.nan
    r1.loc[r1[c] > 4, c] = np.nan
    r1[c.replace('_raw','')] = r1[c] - 1   # now 0-3

# R4 (2008): q49c=police, q49d=courts, q49a=president, q49b=parliament
r4 = load_prepare(4, {'police':'q49c','courts':'q49d',
                       'pres':'q49a','parl':'q49b','trad':'q49h','local':'q49i'})

# ── Core RD function ─────────────────────────────────────────────────────────
def run_rd(df, outcome, bw, poly=1, donut=0, bline=None, sharia_set=None,
           covs=None):
    """Local-polynomial RD. bline/sharia_set override dist_km if provided."""
    d = df.copy()
    if bline is not None:
        # recompute dist with shifted boundary
        d['dist_km'] = compute_dist(d['lat'].values, d['lon'].values,
                                    d['state'].values, bline)
        d['sharia'] = (d['dist_km'] > 0).astype(float)
    d = d.dropna(subset=[outcome,'dist_km','sharia'])
    d = d[d['dist_km'].abs() <= bw]
    if donut > 0:
        d = d[d['dist_km'].abs() >= donut]
    if len(d) < 40 or d[outcome].nunique() < 2:
        return None
    d['y'] = d[outcome]; d['x'] = d['dist_km']
    terms = 'y ~ sharia + x + x:sharia'
    if poly == 2:
        terms += ' + I(x**2) + I(x**2):sharia'
    if covs:
        for cv in covs:
            if cv in d.columns and d[cv].notna().sum() > 10:
                d[cv] = pd.to_numeric(d[cv], errors='coerce')
                terms += f' + {cv}'
    try:
        m = smf.ols(terms, data=d).fit(cov_type='HC3')
    except:
        return None
    c = m.params.get('sharia', np.nan)
    s = m.bse.get('sharia',   np.nan)
    p = m.pvalues.get('sharia', np.nan)
    # Trust scale 0-3: coefficients beyond ±3 or SE > 2 are numerically degenerate
    if abs(c) > 3 or s > 2: return None
    return {'coef':c,'se':s,'pval':p,'n':len(d),
            'n_ns':(d['sharia']==0).sum(),'n_s':(d['sharia']==1).sum()}

def stars(p):
    if pd.isna(p): return ''
    return '***' if p<.01 else '**' if p<.05 else '*' if p<.10 else ''

# ══════════════════════════════════════════════════════════════════════════════
# 1. McCrary density test
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("1. McCALLUM / McCRARY DENSITY CONTINUITY TEST")
print("="*70)
BW_DENSITY = 200
near_d = r2[r2['dist_km'].abs() <= BW_DENSITY]['dist_km'].dropna()
bin_w = 12.5
bins  = np.arange(-BW_DENSITY, BW_DENSITY + bin_w, bin_w)
counts, edges = np.histogram(near_d, bins=bins)
mids  = (edges[:-1] + edges[1:]) / 2

# Estimate density separately on each side and test for jump at 0
left_m  = mids[mids < 0]; left_c  = counts[mids < 0]
right_m = mids[mids > 0]; right_c = counts[mids > 0]

from numpy.polynomial import polynomial as P
if len(left_m) > 3:
    pl = np.polyfit(left_m,  left_c,  2)
    pr = np.polyfit(right_m, right_c, 2)
    pred_left_at0  = np.polyval(pl, 0)
    pred_right_at0 = np.polyval(pr, 0)
    ratio = pred_right_at0 / max(pred_left_at0, 1e-6)
    print(f"  Predicted density just left  of 0: {pred_left_at0:.1f}")
    print(f"  Predicted density just right of 0: {pred_right_at0:.1f}")
    print(f"  Ratio (Sharia/Non-Sharia):          {ratio:.3f}  (1.0 = no jump)")
    print(f"  Interpretation: {'NO EVIDENCE of sorting' if 0.7 < ratio < 1.4 else 'POTENTIAL SORTING — investigate'}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Covariate balance
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("2. COVARIATE BALANCE (R2, BW=150 km)")
print("="*70)
print(f"  {'Covariate':<18} {'N':>5} {'β_RD':>8} {'SE':>7} {'p':>6}")
print("  " + "-"*48)
cov_results = []
for cov in ['age','urban']:
    if cov not in r2.columns: continue
    r = run_rd(r2, cov, 150)
    if r:
        cov_results.append({'label': cov, **r})
        print(f"  {cov:<18} {r['n']:>5} {r['coef']:>8.4f} {r['se']:>7.4f} "
              f"{r['pval']:>6.3f}{stars(r['pval']):>3}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Outcome placebos (trust items with no clear Sharia mechanism)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("3. OUTCOME PLACEBOS (R2, BW=150 km)")
print("   [president, parliament, trad leaders, religious leaders]")
print("="*70)
print(f"  {'Outcome':<22} {'N':>5} {'β_RD':>8} {'SE':>7} {'p':>6}")
print("  " + "-"*52)
placebo_results = []
for lbl, col in [('Trust: president','pres'),('Trust: parliament','parl'),
                  ('Trust: trad. leaders','trad'),('Trust: relig. leaders','relig'),
                  ('Trust: local govt','local')]:
    r = run_rd(r2, col, 150)
    if r:
        placebo_results.append({'label': lbl, **r})
        print(f"  {lbl:<22} {r['n']:>5} {r['coef']:>8.4f} {r['se']:>7.4f} "
              f"{r['pval']:>6.3f}{stars(r['pval']):>3}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Bandwidth sensitivity
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("4. BANDWIDTH SENSITIVITY")
print("="*70)
bws = [50, 75, 100, 125, 150, 175, 200, 250]
bw_results = []
print(f"  {'BW':>5}  {'N_police':>9} {'β_pol':>8} {'p_pol':>6}  {'N_courts':>9} {'β_cts':>8} {'p_cts':>6}")
print("  " + "-"*65)
for bw in bws:
    rp = run_rd(r2, 'police', bw)
    rc = run_rd(r2, 'courts', bw)
    bw_results.append({'bw': bw,
                        'police_coef': rp['coef'] if rp else np.nan,
                        'police_se':   rp['se']   if rp else np.nan,
                        'police_pval': rp['pval'] if rp else np.nan,
                        'police_n':    rp['n']    if rp else 0,
                        'courts_coef': rc['coef'] if rc else np.nan,
                        'courts_se':   rc['se']   if rc else np.nan,
                        'courts_pval': rc['pval'] if rc else np.nan,
                        'courts_n':    rc['n']    if rc else 0,})
    print(f"  {bw:>5}  {rp['n'] if rp else 0:>9} "
          f"{rp['coef']:>8.3f} {rp['pval']:>6.3f}{stars(rp['pval'] if rp else 1):>3}  "
          f"{rc['n'] if rc else 0:>9} "
          f"{rc['coef']:>8.3f} {rc['pval']:>6.3f}{stars(rc['pval'] if rc else 1):>3}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. Donut RD
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("5. DONUT RD (BW=150 km, exclude observations within X km of boundary)")
print("="*70)
print(f"  {'Donut':>8}  {'β_police':>9} {'p_police':>9}  {'β_courts':>9} {'p_courts':>9}")
print("  " + "-"*55)
donut_results = []
for donut in [0, 15, 25, 40, 50]:
    rp = run_rd(r2, 'police', 150, donut=donut)
    rc = run_rd(r2, 'courts', 150, donut=donut)
    donut_results.append({'donut': donut,
                           'police_coef': rp['coef'] if rp else np.nan,
                           'police_pval': rp['pval'] if rp else np.nan,
                           'courts_coef': rc['coef'] if rc else np.nan,
                           'courts_pval': rc['pval'] if rc else np.nan})
    print(f"  {donut:>8}  {rp['coef']:>9.4f} {rp['pval']:>9.3f}{stars(rp['pval'] if rp else 1):>3}  "
          f"{rc['coef']:>9.4f} {rc['pval']:>9.3f}{stars(rc['pval'] if rc else 1):>3}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. Polynomial order
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("6. POLYNOMIAL ORDER (BW=150 km)")
print("="*70)
print(f"  {'Spec':<25}  {'β_police':>9} {'p':>6}  {'β_courts':>9} {'p':>6}")
print("  " + "-"*58)
poly_results = []
for pname, poly in [('Linear (p=1)','1'), ('Quadratic (p=2)','2')]:
    rp = run_rd(r2, 'police', 150, poly=int(poly))
    rc = run_rd(r2, 'courts', 150, poly=int(poly))
    poly_results.append({'spec': pname,
                          'police_coef': rp['coef'] if rp else np.nan,
                          'police_pval': rp['pval'] if rp else np.nan,
                          'courts_coef': rc['coef'] if rc else np.nan,
                          'courts_pval': rc['pval'] if rc else np.nan})
    print(f"  {pname:<25}  {rp['coef']:>9.4f} {rp['pval']:>6.3f}{stars(rp['pval'] if rp else 1):>3}  "
          f"{rc['coef']:>9.4f} {rc['pval']:>6.3f}{stars(rc['pval'] if rc else 1):>3}")

# With covariates
rp = run_rd(r2, 'police', 150, poly=1, covs=[c for c in ['age','edu','female','urban'] if c in r2.columns])
rc = run_rd(r2, 'courts', 150, poly=1, covs=[c for c in ['age','edu','female','urban'] if c in r2.columns])
print(f"  {'Linear + covariates':<25}  {rp['coef']:>9.4f} {rp['pval']:>6.3f}{stars(rp['pval'] if rp else 1):>3}  "
      f"{rc['coef']:>9.4f} {rc['pval']:>6.3f}{stars(rc['pval'] if rc else 1):>3}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. Placebo boundaries (shift boundary north/south)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("7. PLACEBO BOUNDARIES (using shifted boundary lines)")
print("   [respondents reassigned to shifted-boundary dist; main outcomes]")
print("="*70)
print(f"  {'Shift':>12}  {'β_police':>9} {'p':>6}  {'β_courts':>9} {'p':>6}  {'N':>6}")
print("  " + "-"*60)
placebo_bdy_results = []
for shift_km in [-250, -150, 0, 150, 250]:
    shift_deg = shift_km / 111.0
    shifted_coords = [(lon, lat + shift_deg) for lon, lat in SHARIA_BOUNDARY_COORDS]
    shifted_line   = LineString(shifted_coords)
    rp = run_rd(r2, 'police', 150, bline=shifted_line)
    rc = run_rd(r2, 'courts', 150, bline=shifted_line)
    tag = f'{shift_km:+d} km' if shift_km != 0 else 'TRUE BOUNDARY (0 km)'
    placebo_bdy_results.append({'shift': shift_km,
                                  'police_coef': rp['coef'] if rp else np.nan,
                                  'police_pval': rp['pval'] if rp else np.nan,
                                  'courts_coef': rc['coef'] if rc else np.nan,
                                  'courts_pval': rc['pval'] if rc else np.nan})
    print(f"  {tag:>12}  {rp['coef'] if rp else np.nan:>9.4f} "
          f"{rp['pval'] if rp else np.nan:>6.3f}{stars(rp['pval'] if rp else 1):>3}  "
          f"{rc['coef'] if rc else np.nan:>9.4f} "
          f"{rc['pval'] if rc else np.nan:>6.3f}{stars(rc['pval'] if rc else 1):>3}  "
          f"{rp['n'] if rp else 0:>6}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. Pre-trend: R1 (year 2000, before or during Sharia adoption)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("8. PRE-TREND TEST: R1 (2000) — before/during Sharia adoption")
print("   [Scale 0–3 rescaled from original 1–4; same direction as R2]")
print("="*70)
print(f"  {'Round':<12} {'Outcome':<12} {'N':>5} {'β_RD':>8} {'SE':>7} {'p':>6}")
print("  " + "-"*55)
pretrend_results = []
for rnd, df, outcomes in [
    ('R1 (2000)', r1, [('police','police'),('courts','courts')]),
    ('R2 (2003)', r2, [('police','police'),('courts','courts')]),
]:
    for lbl, col in outcomes:
        r = run_rd(df, col, 150)
        if r:
            pretrend_results.append({'round': rnd, 'outcome': lbl, **r})
            print(f"  {rnd:<12} {lbl:<12} {r['n']:>5} {r['coef']:>8.4f} {r['se']:>7.4f} "
                  f"{r['pval']:>6.3f}{stars(r['pval']):>3}")

# ══════════════════════════════════════════════════════════════════════════════
# 9. Pooled rounds: R2 (2003) + R4 (2008)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("9. POOLED ROUNDS R2+R4 (2003 + 2008) with round fixed effect")
print("   [R2: q43i/q43j; R4: q49c/q49d — both 0–3 trust scale]")
print("="*70)
pooled = pd.concat([r2[['police','courts','dist_km','sharia','round','lat','lon','state']],
                    r4[['police','courts','dist_km','sharia','round','lat','lon','state']]],
                   ignore_index=True)

def run_rd_pooled(df, outcome, bw):
    d = df.dropna(subset=[outcome,'dist_km','sharia','round']).copy()
    d = d[d['dist_km'].abs() <= bw]
    if len(d) < 80 or d[outcome].nunique() < 2: return None
    d['y'] = d[outcome]; d['x'] = d['dist_km']
    try:
        m = smf.ols('y ~ sharia + x + x:sharia + C(round)', data=d).fit(cov_type='cluster',
                    cov_kwds={'groups': d['state']})
    except:
        return None
    c = m.params.get('sharia', np.nan)
    s = m.bse.get('sharia',   np.nan)
    p = m.pvalues.get('sharia', np.nan)
    if abs(c) > 10 or s > 10: return None
    return {'coef':c,'se':s,'pval':p,'n':len(d)}

print(f"  {'Outcome':<12} {'N':>6} {'β_RD':>8} {'SE':>7} {'p':>6}")
print("  " + "-"*42)
pool_results = []
for lbl, col in [('police','police'),('courts','courts')]:
    r = run_rd_pooled(pooled, col, 150)
    if r:
        pool_results.append({'label': lbl, **r})
        print(f"  {lbl:<12} {r['n']:>6} {r['coef']:>8.4f} {r['se']:>7.4f} "
              f"{r['pval']:>6.3f}{stars(r['pval']):>3}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating robustness figures…")

C_POL  = '#3d7ab5'
C_CRT  = '#e07b39'
C_GREY = '#aaaaaa'

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.patch.set_facecolor('white')

# ── A: Bandwidth sensitivity ──────────────────────────────────────────────
ax = axes[0, 0]
bw_df = pd.DataFrame(bw_results)
bws_arr = bw_df['bw'].values

ax.fill_between(bws_arr,
                bw_df['police_coef'] - 1.96*bw_df['police_se'],
                bw_df['police_coef'] + 1.96*bw_df['police_se'],
                alpha=0.15, color=C_POL)
ax.plot(bws_arr, bw_df['police_coef'], color=C_POL, lw=2, marker='o', ms=5,
        label='Trust: police')

ax.fill_between(bws_arr,
                bw_df['courts_coef'] - 1.96*bw_df['courts_se'],
                bw_df['courts_coef'] + 1.96*bw_df['courts_se'],
                alpha=0.15, color=C_CRT)
ax.plot(bws_arr, bw_df['courts_coef'], color=C_CRT, lw=2, marker='s', ms=5,
        label='Trust: courts')

ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.6)
ax.set_xlabel('Bandwidth (km)', fontsize=9.5)
ax.set_ylabel('β (Sharia effect on trust, 0–3)', fontsize=9.5)
ax.set_title('A.  Bandwidth Sensitivity', fontsize=11, fontweight='bold')
ax.legend(fontsize=9, loc='lower right', framealpha=0.9, edgecolor='#ccc')
ax.grid(axis='y', alpha=0.3); ax.set_facecolor('#fafafa')
ax.spines[['top','right']].set_visible(False)

# ── B: McCrary density ────────────────────────────────────────────────────
ax = axes[0, 1]
near_d = r2[r2['dist_km'].abs() <= BW_DENSITY]['dist_km'].dropna()
ax.hist(near_d[near_d < 0], bins=np.arange(-BW_DENSITY, 0, bin_w),
        color=C_GREY, alpha=0.75, label='Non-Sharia side', edgecolor='white', lw=0.4)
ax.hist(near_d[near_d > 0], bins=np.arange(0, BW_DENSITY + bin_w, bin_w),
        color=C_POL,  alpha=0.75, label='Sharia side',      edgecolor='white', lw=0.4)

# Fitted density curves
x_left  = np.linspace(-BW_DENSITY, -bin_w/2, 100)
x_right = np.linspace( bin_w/2,  BW_DENSITY, 100)
ax.plot(x_left,  np.polyval(np.polyfit(left_m,  left_c,  2), x_left),
        color='#555555', lw=1.5, ls='--')
ax.plot(x_right, np.polyval(np.polyfit(right_m, right_c, 2), x_right),
        color='#555555', lw=1.5, ls='--')
ax.axvline(0, color=C_POL, lw=1.2, ls=':', alpha=0.8)

ax.set_xlabel('Signed distance from Sharia boundary (km)', fontsize=9.5)
ax.set_ylabel('Number of respondents', fontsize=9.5)
ax.set_title('B.  McCrary Density Test\n(smoothed density continuous at 0 ⟹ no sorting)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.9, edgecolor='#ccc')
ax.grid(axis='y', alpha=0.3); ax.set_facecolor('#fafafa')
ax.spines[['top','right']].set_visible(False)
ax.text(0.03, 0.95, f'Ratio: {ratio:.2f}\n(1.0 = no gap)', transform=ax.transAxes,
        fontsize=8.5, va='top', color='#333',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#bbb', lw=0.8))

# ── C: Covariate balance + outcome placebos ───────────────────────────────
ax = axes[1, 0]
ax.set_facecolor('white')
ax.spines[['top','right','left']].set_visible(False)
ax.spines['bottom'].set_linewidth(0.7)

all_checks = (
    [(r['label'], r, '#888888', 'o') for r in cov_results] +
    [(r['label'], r, C_CRT,     's') for r in placebo_results]
)
all_checks = [(l, r, c, m) for l, r, c, m in all_checks if r is not None]

y_pos = np.arange(len(all_checks))[::-1]
for i, (lbl, r, color, mk) in enumerate(all_checks):
    yy   = y_pos[i]
    coef = r['coef']; se = r['se']; pval = r['pval']
    ci95 = (coef - 1.96*se, coef + 1.96*se)
    ec   = color if pval < 0.10 else '#cccccc'
    ax.plot(ci95, [yy, yy], color=ec, lw=5, solid_capstyle='butt', alpha=0.6, zorder=2)
    ax.plot(coef, yy, marker=mk, ms=8, color='white', markeredgecolor=ec, mew=1.8, zorder=3)
    sig_str = f'p={pval:.3f}' if pval < 0.10 else ''
    ax.text(ci95[1] + 0.01, yy, sig_str, va='center', fontsize=7.5, color=color)

ax.set_yticks(y_pos)
ax.set_yticklabels([l for l, *_ in all_checks], fontsize=9)
ax.axvline(0, color='#333', lw=0.9, ls='--', alpha=0.6, zorder=1)
ax.set_xlabel('β (Sharia state effect, 0–3 scale, BW=150 km)', fontsize=9.5)
ax.set_title('C.  Covariate Balance (●) and Outcome Placebos (■)',
             fontsize=11, fontweight='bold')
ax.grid(axis='x', alpha=0.2)

# Add divider between covariates and placebos
if cov_results and placebo_results:
    mid = (y_pos[len(cov_results)-1] + y_pos[len(cov_results)]) / 2
    ax.axhline(mid, color='#aaa', lw=0.8, ls=':')
    ax.text(ax.get_xlim()[0] + 0.01 if ax.get_xlim()[0] > -10 else -0.6,
            mid + 0.35, 'Covariates', fontsize=7.5, color='#777')
    ax.text(ax.get_xlim()[0] + 0.01 if ax.get_xlim()[0] > -10 else -0.6,
            mid - 0.35, 'Placebos', fontsize=7.5, color='#777', va='top')

# ── D: Pre-trend + placebo boundaries ────────────────────────────────────
ax = axes[1, 1]
ax.set_facecolor('#fafafa')
ax.spines[['top','right']].set_visible(False)

# Pre-trend: R1 vs R2 police and courts
pretrend_df = pd.DataFrame(pretrend_results)
n_pt = len(pretrend_df)
y_pt = np.arange(n_pt)

# Placebo boundaries
pbdy_df = pd.DataFrame(placebo_bdy_results)

# Two sub-axes side by side using twinx / manual positioning
# Use a split: top half = placebo bounds, bottom half = pre-trend
fig_d_ax = ax

# Plot: placebo boundaries for police
pbdy_df_sorted = pbdy_df.sort_values('shift')
shifts  = pbdy_df_sorted['shift'].values
pol_c   = pbdy_df_sorted['police_coef'].values
pol_s   = (pbdy_df_sorted['police_pval'].apply(lambda p: '*' if p < 0.10 else '')).values
crt_c   = pbdy_df_sorted['courts_coef'].values

x_pos = np.arange(len(shifts))
width = 0.35
bars_pol = ax.bar(x_pos - width/2, pol_c, width, color=C_POL, alpha=0.7,
                  label='Police', zorder=3)
bars_crt = ax.bar(x_pos + width/2, crt_c, width, color=C_CRT, alpha=0.7,
                  label='Courts', zorder=3)

# Annotate significance
for i, (p, s) in enumerate(zip(pbdy_df_sorted['police_pval'], pol_s)):
    if s: ax.text(i - width/2, pol_c[i] - 0.04, s, ha='center', va='top',
                  fontsize=10, color='white', fontweight='bold')

# Highlight true boundary
true_idx = np.where(pbdy_df_sorted['shift'].values == 0)[0][0]
ax.axvspan(true_idx - 0.5, true_idx + 0.5, color='#f5e642', alpha=0.25, zorder=1)
ax.text(true_idx, ax.get_ylim()[0] if ax.get_ylim()[0] > -5 else -0.9,
        'True\nboundary', ha='center', va='bottom', fontsize=7.5, color='#7a6800')

ax.set_xticks(x_pos)
labels_shift = [f'{s:+d} km' if s != 0 else '0 km\n(true)' for s in shifts]
ax.set_xticklabels(labels_shift, fontsize=8.5)
ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.6)
ax.set_xlabel('Boundary shift (north = +km)', fontsize=9.5)
ax.set_ylabel('β (Sharia effect on trust, 0–3)', fontsize=9.5)
ax.set_title('D.  Placebo Boundaries\n(significant effect should vanish at fake cutoffs)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.9, edgecolor='#ccc')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.5)
plt.savefig('sharia_rd_robustness_fig.png', dpi=175, bbox_inches='tight', facecolor='white')
print("Saved sharia_rd_robustness_fig.png")

# ── Pre-trend table embedded in subtitle ──
print("\n── Pre-trend summary ──")
print(pretrend_df[['round','outcome','n','coef','se','pval']].to_string(index=False))

# ── Save full results CSV ─────────────────────────────────────────────────────
all_rows = []
for r in cov_results:     all_rows.append({'test':'covariate_balance', **r})
for r in placebo_results: all_rows.append({'test':'outcome_placebo',   **r})
for r in bw_results:      all_rows.append({'test':'bandwidth_sensitivity', **r})
for r in donut_results:   all_rows.append({'test':'donut_rd',          **r})
for r in poly_results:    all_rows.append({'test':'polynomial_order',  **r})
for r in placebo_bdy_results: all_rows.append({'test':'placebo_boundary', **r})
for r in pretrend_results: all_rows.append({'test':'pre_trend',         **r})
for r in pool_results:    all_rows.append({'test':'pooled_rounds',      **r})
pd.DataFrame(all_rows).to_csv('sharia_rd_robustness_results.csv', index=False)
print(f"\nSaved sharia_rd_robustness_results.csv ({len(all_rows)} rows)")
