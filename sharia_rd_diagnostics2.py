"""
Supplemental diagnostics: corrected Kaduna exclusion and placebo boundary.
"""
import os, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
import statsmodels.formula.api as smf
from rdrobust import rdrobust
warnings.filterwarnings('ignore')

os.chdir('/home/user/nig')

SHARIA_STATES = {'Zamfara','Kano','Sokoto','Katsina','Bauchi','Niger',
                 'Jigawa','Kebbi','Yobe','Borno','Kaduna','Gombe'}
BW_MAIN = 150

nga = gpd.read_file('/tmp/nga_admin1.gpkg')
NE_FIX = {'Federal Capital Territory': 'FCT', 'Nassarawa': 'Nasarawa'}
nga['state'] = nga['name'].replace(NE_FIX)
nga['sharia'] = nga['state'].isin(SHARIA_STATES)
sharia_union    = unary_union(nga[nga['sharia']].geometry)
nonsharia_union = unary_union(nga[~nga['sharia']].geometry)
official_boundary = sharia_union.boundary.intersection(nonsharia_union.boundary)

def signed_dist_km(lat, lon):
    p = Point(lon, lat)
    lat_rad = np.radians(lat)
    dlat = np.pi * 6371.0 / 180.0
    dlon = dlat * np.cos(lat_rad)
    km = official_boundary.distance(p) * np.sqrt(dlat**2 + dlon**2) / np.sqrt(2)
    sign = 1.0 if sharia_union.contains(p) else -1.0
    return sign * km

def extract_state(s):
    parts = str(s).split('|')
    if len(parts) < 4: return None
    return parts[3].replace(' State','').strip()

def star(p):
    if pd.isna(p): return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.10:  return '†'
    return ''

def rd_ols(df, outcome, bw=150):
    d = df.dropna(subset=[outcome,'dist_km']).copy()
    d = d[d['dist_km'].abs() <= bw]
    if len(d) < 50 or d[outcome].nunique() < 2: return None
    if d['sharia'].nunique() < 2: return None
    n_cl = d['ea_id'].nunique() if 'ea_id' in d.columns else 0
    cov = ('cluster', {'groups': d['ea_id']}) if n_cl >= 5 else ('HC3', {})
    try:
        m = smf.ols('y ~ sharia + x + x:sharia',
                    data=d.assign(y=d[outcome], x=d['dist_km'])
                    ).fit(cov_type=cov[0], cov_kwds=cov[1])
        c = m.params.get('sharia', np.nan)
        s = m.bse.get('sharia', np.nan)
        p = m.pvalues.get('sharia', np.nan)
        if abs(c) > 3 or s > 2: return None
        return {'coef': c, 'se': s, 'pval': p, 'n': len(d), 'n_cl': n_cl}
    except Exception:
        return None

def load_round(fname, outcomes_map):
    df = pd.read_excel(fname)
    df.columns = df.columns.str.lower().str.strip()
    lat_col = next((c for c in df.columns if c in ('latitude','lat')), None)
    lon_col = next((c for c in df.columns if c in ('longitude','lon')), None)
    df = df.rename(columns={lat_col:'lat', lon_col:'lon'})
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df = df.dropna(subset=['lat','lon']).reset_index(drop=True)
    if 'geoname_adm_name' in df.columns:
        df['state_survey'] = df['geoname_adm_name'].apply(extract_state)
    else:
        df['state_survey'] = None
    df['sharia'] = df['state_survey'].isin(SHARIA_STATES).astype(float)
    df['dist_km'] = [signed_dist_km(r['lat'], r['lon']) for _,r in df.iterrows()]
    ea_col = next((c for c in ['uniqueea','eanumb'] if c in df.columns
                   and df[c].nunique() > 1), None)
    df['ea_id'] = df[ea_col] if ea_col else np.arange(len(df))
    for name, (col, recode) in outcomes_map.items():
        if col not in df.columns:
            df[name] = np.nan; continue
        v = pd.to_numeric(df[col], errors='coerce')
        if recode == '0-3':   v = v.where(v <= 3)
        elif recode == '1-5->0-3': v = (v.where(v <= 5) - 1) * (3/4)
        df[name] = v
    return df

print("Loading...")
r2 = load_round('NIG_r2.csv.xlsx', {
    'police':('q43i','0-3'), 'courts':('q43j','0-3'), 'relig':('q43g','0-3'),
    'urban':('urbrur','0-3'),
})
r4 = load_round('NIG_r4.csv.xlsx', {
    'police':('q49c','0-3'), 'courts':('q49d','0-3'), 'relig':('q49i','0-3'),
    'urban':('urbrur','0-3'),
})
r3 = load_round('NIG_r3.csv.xlsx', {
    'police':('q52c','1-5->0-3'), 'courts':('q52d','1-5->0-3'),
    'relig':('q52b','1-5->0-3'),
})

# ════════════════════════════════════════════════════════════════════════════
# A. KADUNA: DROP (not reclassify) from both sides
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('KADUNA EXCLUSION — drop all Kaduna respondents (not reclassify)')
print('This removes the non-uniform-Sharia-exposure problem at boundary')
print('='*80)

for label, df in [('R2',r2),('R4',r4)]:
    df_nok = df[df['state_survey'] != 'Kaduna'].copy()
    near_full = df[df['dist_km'].abs() <= BW_MAIN]
    near_nok  = df_nok[df_nok['dist_km'].abs() <= BW_MAIN]
    print(f"\n  {label}: full N within {BW_MAIN}km={len(near_full)}, "
          f"no-Kaduna N={len(near_nok)} "
          f"({len(near_full)-len(near_nok)} dropped)")
    print(f"  {'Outcome':10}  {'With Kaduna β':>16}  {'No Kaduna β':>14}  {'Δβ':>8}")
    print(f"  {'-'*55}")
    for outcome in ['police','courts','relig']:
        if outcome not in df.columns: continue
        r_f = rd_ols(df, outcome, BW_MAIN)
        r_n = rd_ols(df_nok, outcome, BW_MAIN)
        fs = f"{r_f['coef']:>+8.3f}{star(r_f['pval']):<3}" if r_f else '   —'
        ns = f"{r_n['coef']:>+8.3f}{star(r_n['pval']):<3}" if r_n else '   —'
        delta = (r_n['coef'] - r_f['coef']) if (r_f and r_n) else float('nan')
        print(f"  {outcome:<10}  {fs:>16}  {ns:>14}  {delta:>+8.3f}" if not np.isnan(delta)
              else f"  {outcome:<10}  {fs:>16}  {ns:>14}     —")

# ════════════════════════════════════════════════════════════════════════════
# B. PROPER PLACEBO BOUNDARY: shift cutoff to dist_km = ±50 km INSIDE each side
#    (i.e., test for discontinuity at a false boundary inside Sharia territory)
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('PLACEBO CUTOFF (correct implementation)')
print('Test for discontinuity at dist_km = +75 (50-100 km inside Sharia territory)')
print('Respondents within [50,150] km on Sharia side; cutoff at 75 km')
print('Should show β ≈ 0 if there is no within-Sharia gradient at that location')
print('='*80)

def rd_placebo_cutoff(df, outcome, bw=150, cutoff=75, window=50):
    """
    Test discontinuity at dist_km = cutoff (inside Sharia territory).
    Use respondents with dist_km in [cutoff-window, cutoff+window] on Sharia side.
    Running variable is dist_km - cutoff (centered at false cutoff).
    Treatment = dist_km > cutoff (deeper inside Sharia vs. boundary zone).
    """
    d = df.dropna(subset=[outcome,'dist_km']).copy()
    d = d[(d['dist_km'] >= cutoff - window) & (d['dist_km'] <= cutoff + window)]
    if len(d) < 50 or d[outcome].nunique() < 2: return None
    d['x_c'] = d['dist_km'] - cutoff
    d['treat'] = (d['dist_km'] > cutoff).astype(float)
    if d['treat'].nunique() < 2: return None
    n_cl = d['ea_id'].nunique() if 'ea_id' in d.columns else 0
    cov = ('cluster', {'groups': d['ea_id']}) if n_cl >= 5 else ('HC3', {})
    try:
        m = smf.ols('y ~ treat + x_c + x_c:treat',
                    data=d.assign(y=d[outcome])
                    ).fit(cov_type=cov[0], cov_kwds=cov[1])
        c = m.params.get('treat', np.nan)
        s = m.bse.get('treat', np.nan)
        p = m.pvalues.get('treat', np.nan)
        return {'coef': c, 'se': s, 'pval': p, 'n': len(d)}
    except Exception:
        return None

# Also test on non-Sharia side (cutoff at dist_km = -75)
def rd_placebo_cutoff_ns(df, outcome, bw=150, cutoff=-75, window=50):
    d = df.dropna(subset=[outcome,'dist_km']).copy()
    d = d[(d['dist_km'] >= cutoff - window) & (d['dist_km'] <= cutoff + window)]
    if len(d) < 50 or d[outcome].nunique() < 2: return None
    d['x_c'] = d['dist_km'] - cutoff
    d['treat'] = (d['dist_km'] > cutoff).astype(float)
    if d['treat'].nunique() < 2: return None
    n_cl = d['ea_id'].nunique() if 'ea_id' in d.columns else 0
    cov = ('cluster', {'groups': d['ea_id']}) if n_cl >= 5 else ('HC3', {})
    try:
        m = smf.ols('y ~ treat + x_c + x_c:treat',
                    data=d.assign(y=d[outcome])
                    ).fit(cov_type=cov[0], cov_kwds=cov[1])
        c = m.params.get('treat', np.nan)
        s = m.bse.get('treat', np.nan)
        p = m.pvalues.get('treat', np.nan)
        return {'coef': c, 'se': s, 'pval': p, 'n': len(d)}
    except Exception:
        return None

print(f"\n  {'Round':5}  {'Outcome':10}  {'Real β':>10}  {'Placebo+75 β':>14}  "
      f"{'Placebo-75 β':>14}")
print(f"  {'-'*60}")
for label, df in [('R2',r2),('R4',r4)]:
    for outcome in ['police','courts','relig']:
        if outcome not in df.columns: continue
        r_real = rd_ols(df, outcome, BW_MAIN)
        r_p75  = rd_placebo_cutoff(df, outcome, window=50)
        r_n75  = rd_placebo_cutoff_ns(df, outcome, window=50)
        rs = f"{r_real['coef']:>+8.3f}{star(r_real['pval']):<2}" if r_real else '   —'
        ps = f"{r_p75['coef']:>+8.3f}{star(r_p75['pval']):<2}" if r_p75 else '   —'
        ns = f"{r_n75['coef']:>+8.3f}{star(r_n75['pval']):<2}" if r_n75 else '   —'
        print(f"  {label:<5}  {outcome:<10}  {rs:>10}  {ps:>14}  {ns:>14}")

# ════════════════════════════════════════════════════════════════════════════
# C. R4 URBAN BALANCE — is urbanization driving the R4 effect?
#    Control for urban in the OLS specification
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('R4 URBAN CONTROL — include urban indicator in RD regression')
print('Addresses: R4 shows significant urban discontinuity (β=+0.51, p=0.021)')
print('='*80)

def rd_ols_urban(df, outcome, bw=150):
    d = df.dropna(subset=[outcome,'dist_km','urban']).copy()
    d = d[d['dist_km'].abs() <= bw]
    if len(d) < 50 or d[outcome].nunique() < 2: return None
    if d['sharia'].nunique() < 2: return None
    n_cl = d['ea_id'].nunique() if 'ea_id' in d.columns else 0
    cov = ('cluster', {'groups': d['ea_id']}) if n_cl >= 5 else ('HC3', {})
    try:
        m = smf.ols('y ~ sharia + x + x:sharia + urban',
                    data=d.assign(y=d[outcome], x=d['dist_km'])
                    ).fit(cov_type=cov[0], cov_kwds=cov[1])
        c = m.params.get('sharia', np.nan)
        s = m.bse.get('sharia', np.nan)
        p = m.pvalues.get('sharia', np.nan)
        return {'coef': c, 'se': s, 'pval': p, 'n': len(d)}
    except Exception:
        return None

print(f"\n  {'Outcome':10}  {'Without urban':>17}  {'With urban β':>16}")
print(f"  {'-'*50}")
for outcome in ['police','courts','relig']:
    if outcome not in r4.columns: continue
    r_no  = rd_ols(r4, outcome, BW_MAIN)
    r_urb = rd_ols_urban(r4, outcome, BW_MAIN)
    ns = f"{r_no['coef']:>+8.3f}{star(r_no['pval']):<3}" if r_no else '   —'
    us = f"{r_urb['coef']:>+8.3f}{star(r_urb['pval']):<3}" if r_urb else '   —'
    print(f"  {outcome:<10}  {ns:>17}  {us:>16}")

# ════════════════════════════════════════════════════════════════════════════
# D. POOLED R2-R4 no-Kaduna
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('POOLED R2-R4 WITHOUT KADUNA — standardized, round FE')
print('='*80)

def standardize(s): m, sd = s.mean(), s.std(); return (s-m)/sd if sd>0 else s*0

for outcome, label in [('police','Police'),('courts','Courts'),('relig','Relig leaders')]:
    rows = []
    for rnd, yr, df in [('R2',2003,r2),('R3',2005,r3),('R4',2008,r4)]:
        if outcome not in df.columns: continue
        near = df[(df['dist_km'].abs() <= BW_MAIN) &
                  (df['state_survey'] != 'Kaduna')].copy()
        near = near.dropna(subset=[outcome,'dist_km'])
        near['round'] = rnd; near['year'] = yr
        rows.append(near[['sharia','dist_km','ea_id',outcome,'round','year']].copy())
    if not rows: continue
    pool = pd.concat(rows, ignore_index=True)
    pool['y_std'] = standardize(pool[outcome].dropna())
    pool = pool.dropna(subset=['y_std'])
    if len(pool) < 100: continue
    n_cl = pool['ea_id'].nunique()
    cov = ('cluster', {'groups': pool['ea_id']}) if n_cl >= 5 else ('HC3', {})
    try:
        m = smf.ols('y_std ~ sharia + x + x:sharia + C(round)',
                    data=pool.assign(x=pool['dist_km'])
                   ).fit(cov_type=cov[0], cov_kwds=cov[1])
        c = m.params.get('sharia', np.nan)
        p = m.pvalues.get('sharia', np.nan)
        print(f"  {label}: β={c:+.3f} p={p:.3f}{star(p)} N={len(pool)} EAs={n_cl}")
    except Exception as e:
        print(f"  {label}: error {e}")

# ════════════════════════════════════════════════════════════════════════════
# E. SCTRUST in R1: what is it? Can we use it as a pre-trend?
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('R1 (2000) sctrust — value distribution and pre-trend check')
print('NOTE: R1 overlaps with Sharia rollout (Zamfara Jan 2000); not a clean pre-period')
print('='*80)

r1_raw = pd.read_excel('NIG_r1.csv.xlsx')
r1_raw.columns = r1_raw.columns.str.lower()
print(f"  sctrust values: {pd.to_numeric(r1_raw['sctrust'],errors='coerce').value_counts().sort_index().to_dict()}")

r1 = load_round('NIG_r1.csv.xlsx', {'sctrust': ('sctrust','raw')})
r1['sctrust_c'] = pd.to_numeric(r1['sctrust'], errors='coerce')
print(f"  sctrust range: {r1['sctrust_c'].min():.0f}–{r1['sctrust_c'].max():.0f}")
r = rd_ols(r1, 'sctrust_c', BW_MAIN)
if r:
    print(f"  R1 sctrust RD: β={r['coef']:+.3f} SE={r['se']:.3f} "
          f"p={r['pval']:.3f}{star(r['pval'])} N={r['n']}")
else:
    print("  R1 sctrust: insufficient data for RD")

print('\nDiagnostics2 complete.')
