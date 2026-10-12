"""
Comprehensive diagnostics for Sharia RD analysis.
Addresses reviewer concerns 1-8.
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

# ── Boundary ──────────────────────────────────────────────────────────────────
nga = gpd.read_file('/tmp/nga_admin1.gpkg')
NE_FIX = {'Federal Capital Territory': 'FCT', 'Nassarawa': 'Nasarawa'}
nga['state'] = nga['name'].replace(NE_FIX)
nga['sharia'] = nga['state'].isin(SHARIA_STATES)
sharia_union    = unary_union(nga[nga['sharia']].geometry)
nonsharia_union = unary_union(nga[~nga['sharia']].geometry)
official_boundary = sharia_union.boundary.intersection(nonsharia_union.boundary)

def signed_dist_km(lat, lon):
    """Positive = inside Sharia state (right side for rdrobust)."""
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

def rd_ols(df, outcome, bw=150, cluster_col='ea_id', exclude_kaduna=False):
    d = df.dropna(subset=[outcome,'dist_km']).copy()
    d = d[d['dist_km'].abs() <= bw]
    if exclude_kaduna:
        d = d[~d.get('state_survey','').str.contains('Kaduna', na=False)]
    if len(d) < 50 or d[outcome].nunique() < 2: return None
    if d['sharia'].nunique() < 2: return None
    n_cl = d[cluster_col].nunique() if cluster_col in d.columns else 0
    cov = ('cluster', {'groups': d[cluster_col]}) if n_cl >= 5 else ('HC3', {})
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

def rd_cct(df, outcome, bw=None, use_adjust=True):
    """rdrobust with masspoints='adjust' (correct) and cluster."""
    d = df.dropna(subset=[outcome,'dist_km']).copy()
    if bw: d = d[d['dist_km'].abs() <= bw]
    if len(d) < 80 or d[outcome].nunique() < 2: return None
    if d['sharia'].nunique() < 2: return None
    mp = 'adjust' if use_adjust else 'off'
    # Build cluster vector if available
    try:
        if 'ea_id' in d.columns and d['ea_id'].nunique() >= 5:
            r = rdrobust(d[outcome].values, d['dist_km'].values,
                         kernel='triangular', bwselect='mserd',
                         masspoints=mp, cluster=d['ea_id'].values)
        else:
            r = rdrobust(d[outcome].values, d['dist_km'].values,
                         kernel='triangular', bwselect='mserd',
                         masspoints=mp)
        bw_l = float(r.bws.loc['h','left'])
        bw_r = float(r.bws.loc['h','right'])
        coef  = float(r.coef.loc['Robust','Coeff'])
        se    = float(r.se.loc['Robust','Std. Err.'])
        pval  = float(r.pv.loc['Robust','P>|z|'])
        return {'bw_l': bw_l, 'bw_r': bw_r, 'coef': coef, 'se': se, 'pval': pval}
    except Exception as e:
        return {'error': str(e)}

def load_round(fname, outcomes_map, state_col=None):
    df = pd.read_excel(fname)
    df.columns = df.columns.str.lower().str.strip()
    lat_col = next((c for c in df.columns if c in ('latitude','lat')), None)
    lon_col = next((c for c in df.columns if c in ('longitude','lon')), None)
    df = df.rename(columns={lat_col:'lat', lon_col:'lon'})
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df = df.dropna(subset=['lat','lon']).reset_index(drop=True)

    if state_col and state_col in df.columns:
        df['state_survey'] = df[state_col].astype(str).str.strip()
    elif 'geoname_adm_name' in df.columns:
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
        elif recode == '0-4': v = v.where(v <= 4)
        elif recode == '1-4': v = v.where(v <= 4) - 1
        elif recode == '1-5': v = v.where(v <= 5) - 1
        elif recode == 'raw': pass
        df[name] = v

    return df

# ════════════════════════════════════════════════════════════════════════════
# LOAD MAIN ROUNDS
# ════════════════════════════════════════════════════════════════════════════
print("Loading rounds...")

r2 = load_round('NIG_r2.csv.xlsx', {
    'police':('q43i','0-3'), 'courts':('q43j','0-3'),
    'relig': ('q43g','0-3'),
    # Covariates: education q97 (1-9), age q1 (numeric), urban urbrur (1=urban)
    'educ':  ('q97','raw'),  'age':('q1','raw'),   'urban':('urbrur','raw'),
    'econ_personal': ('q4a','raw'),   # personal economic conditions
    'econ_country':  ('q4b','raw'),   # country economic conditions
})
# Muslim share proxy: religion column
for rc in ['q98','q99','religion']:
    if rc in pd.read_excel('NIG_r2.csv.xlsx').columns.str.lower():
        tmp = pd.read_excel('NIG_r2.csv.xlsx')
        tmp.columns = tmp.columns.str.lower()
        r2['religion_raw'] = pd.to_numeric(tmp[rc], errors='coerce').values[:len(r2)]
        break

r3 = load_round('NIG_r3.csv.xlsx', {
    'police':('q52c','1-5->0-3'), 'courts':('q52d','1-5->0-3'),
    'relig': ('q52b','1-5->0-3'),
    'educ':  ('q97','raw'),  'age':('q1','raw'),
})

r4 = load_round('NIG_r4.csv.xlsx', {
    'police':('q49c','0-3'), 'courts':('q49d','0-3'),
    'relig': ('q49i','0-3'),
    'educ':  ('q97','raw'),  'age':('q1','raw'), 'urban':('urbrur','raw'),
    'econ_personal': ('q4a','raw'),
})

print(f"  R2: {len(r2)}, R3: {len(r3)}, R4: {len(r4)}")

# ════════════════════════════════════════════════════════════════════════════
# 1. SIGN VERIFICATION
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('SIGN VERIFICATION')
print('NOTE: dist_km > 0 = Sharia side (positive = Sharia, negative = non-Sharia)')
print('rdrobust estimates right−left = Sharia−non-Sharia (same sign as OLS)')
print('='*80)

for label, df in [('R2', r2), ('R4', r4)]:
    near = df[df['dist_km'].abs() <= 50].copy()
    sharia_side = near[near['sharia'] == 1]
    nonsharia_side = near[near['sharia'] == 0]
    print(f"\n  {label} raw means within 50 km:")
    print(f"  {'Outcome':12}  {'Sharia':>8}  {'Non-Sharia':>12}  {'Diff (S-NS)':>14}")
    print(f"  {'-'*55}")
    for outcome in ['police','courts','relig']:
        if outcome not in df.columns: continue
        ms = sharia_side[outcome].mean()
        mn = nonsharia_side[outcome].mean()
        print(f"  {outcome:<12}  {ms:>8.3f}  {mn:>12.3f}  {ms-mn:>+14.3f}")

# ════════════════════════════════════════════════════════════════════════════
# 2. R4 SCALE CHECK
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('R4 SCALE CHECK — value distributions for q49c (police), q49d (courts), q49i (relig)')
print('='*80)
r4_raw = pd.read_excel('NIG_r4.csv.xlsx')
r4_raw.columns = r4_raw.columns.str.lower()
for col, label in [('q49c','Police'),('q49d','Courts'),('q49i','Relig leaders')]:
    if col in r4_raw.columns:
        vc = pd.to_numeric(r4_raw[col], errors='coerce').value_counts().sort_index()
        print(f"  {label} ({col}): {dict(vc)}")

# ════════════════════════════════════════════════════════════════════════════
# 3. R1 PRE-TREND (placebo — Sharia not yet implemented)
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('R1 PRE-TREND (2000 — Sharia not yet implemented, should show β ≈ 0)')
print('='*80)
r1_raw = pd.read_excel('NIG_r1.csv.xlsx')
r1_raw.columns = r1_raw.columns.str.lower()
print(f"  R1 columns sample: {list(r1_raw.columns[:30])}")
# Find trust-relevant columns in R1
trust_candidates = [c for c in r1_raw.columns if
                    any(s in c for s in ['trust','q43','q52','q49','police','court','relig'])]
print(f"  Trust-related columns: {trust_candidates}")

r1 = load_round('NIG_r1.csv.xlsx', {})  # load geometry only first
print(f"  R1: {len(r1)} obs, sharia side: {(r1['sharia']==1).sum()}, "
      f"non-sharia: {(r1['sharia']==0).sum()}")

# Find numeric outcome columns in R1 that look like trust scales
numeric_cols = []
for col in r1_raw.columns:
    v = pd.to_numeric(r1_raw[col], errors='coerce').dropna()
    if len(v) > 100 and v.min() >= 0 and v.max() <= 5 and v.max() >= 2:
        numeric_cols.append((col, v.min(), v.max(), v.mean()))
print(f"\n  Candidate 0-4/0-3 columns in R1:")
for c, mn, mx, avg in numeric_cols[:30]:
    print(f"    {c}: min={mn:.0f} max={mx:.0f} mean={avg:.2f}")

# ════════════════════════════════════════════════════════════════════════════
# 4. COVARIATE BALANCE (R2 and R4)
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('COVARIATE BALANCE (RD on predetermined covariates, BW=150 km)')
print('If RD shows significant discontinuity in covariates → sorting concern')
print('='*80)

for label, df in [('R2',r2),('R4',r4)]:
    print(f"\n  {label}:")
    print(f"  {'Covariate':20}  {'β':>8}  {'SE':>7}  {'p':>8}  {'N':>5}")
    print(f"  {'-'*55}")
    for cov_name in ['educ','age','urban','econ_personal','econ_country']:
        if cov_name not in df.columns: continue
        r = rd_ols(df, cov_name, BW_MAIN)
        if r:
            print(f"  {cov_name:<20}  {r['coef']:>+8.3f}  {r['se']:>7.3f}  "
                  f"{r['pval']:>7.3f}{star(r['pval'])}  {r['n']:>5}")
        else:
            print(f"  {cov_name:<20}  insufficient data")

# Muslim/religion balance
print()
for col_r, df, label in [('religion_raw', r2, 'R2 religion')]:
    if col_r not in df.columns:
        print(f"  {label}: column not found"); continue
    r = rd_ols(df, col_r, BW_MAIN)
    if r:
        print(f"  {label}:  β={r['coef']:+.3f}  SE={r['se']:.3f}  "
              f"p={r['pval']:.3f}{star(r['pval'])}  N={r['n']}")

# ════════════════════════════════════════════════════════════════════════════
# 5. MCCRARY DENSITY TEST (manipulation test)
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('DENSITY TEST (manipulation check — no sorting should produce smooth density at 0)')
print('='*80)

try:
    from rddensity import rddensity as rddens
    for label, df in [('R2',r2),('R4',r4)]:
        x = df['dist_km'].dropna()
        x = x[x.abs() <= BW_MAIN]
        res = rddens(x.values)
        # rddensity output
        print(f"  {label}: p-value = {float(res.test.p_jk):.3f}")
except ImportError:
    # Fallback: simple bin-count comparison
    print("  rddensity not available — using bin-count comparison")
    for label, df in [('R2',r2),('R4',r4)]:
        x = df['dist_km'].dropna()
        x = x[x.abs() <= BW_MAIN]
        bin_edges = np.linspace(-BW_MAIN, BW_MAIN, 31)
        counts, _ = np.histogram(x.values, bins=bin_edges)
        left_bins  = counts[:15]
        right_bins = counts[15:]
        # Expect counts to be similar on both sides if no manipulation
        print(f"  {label}: left={left_bins.sum()} obs, right={right_bins.sum()} obs, "
              f"ratio={right_bins.sum()/max(left_bins.sum(),1):.2f}")
        print(f"    Left mean per bin={left_bins.mean():.1f}, "
              f"right mean per bin={right_bins.mean():.1f}")

# ════════════════════════════════════════════════════════════════════════════
# 6. DONUT (exclude ±20 km around boundary)
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('DONUT RD (exclude ±20 km, BW=150 km) — robust to sorting near boundary')
print('='*80)

def rd_ols_donut(df, outcome, bw=150, donut=20):
    d = df.dropna(subset=[outcome,'dist_km']).copy()
    d = d[(d['dist_km'].abs() <= bw) & (d['dist_km'].abs() >= donut)]
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
        return {'coef': c, 'se': s, 'pval': p, 'n': len(d)}
    except Exception:
        return None

print(f"\n  {'Round':5}  {'Outcome':10}  {'Full β':>10}  {'Donut β':>10}  {'Δ':>8}")
print(f"  {'-'*50}")
for label, df in [('R2',r2),('R4',r4)]:
    for outcome in ['police','courts','relig']:
        if outcome not in df.columns: continue
        r_full  = rd_ols(df, outcome, BW_MAIN)
        r_donut = rd_ols_donut(df, outcome, BW_MAIN, donut=20)
        if r_full and r_donut:
            delta = r_donut['coef'] - r_full['coef']
            print(f"  {label:<5}  {outcome:<10}  "
                  f"{r_full['coef']:>+8.3f}{star(r_full['pval']):<2}  "
                  f"{r_donut['coef']:>+8.3f}{star(r_donut['pval']):<2}  "
                  f"{delta:>+8.3f}")

# ════════════════════════════════════════════════════════════════════════════
# 7. PLACEBO BOUNDARY (shift all points +0.5° north)
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('PLACEBO BOUNDARY (shift all respondent locations +0.5° north)')
print('Should show β ≈ 0 if result is not a general north-south gradient')
print('='*80)

def add_placebo_dist(df, lat_shift=0.5):
    """Recompute dist_km as if respondents were shifted north by lat_shift degrees."""
    d = df.copy()
    d['dist_km_placebo'] = [
        signed_dist_km(row['lat'] + lat_shift, row['lon'])
        for _, row in d.iterrows()
    ]
    return d

# Only run for R4 (most significant round) to save time
print("  Computing placebo distances for R4 (may take ~30s)...")
r4_placebo = add_placebo_dist(r4, lat_shift=0.5)

for outcome in ['police','courts','relig']:
    if outcome not in r4_placebo.columns: continue
    r4_p = r4_placebo.rename(columns={'dist_km_placebo':'dist_km_orig',
                                       'dist_km':'dist_km_true'})
    r4_p = r4_p.copy()
    r4_p['dist_km'] = r4_p['dist_km_orig']  # use placebo dist
    # Need to re-assign sharia based on placebo location
    # Use placebo sharia assignment based on shifted location
    r4_p['sharia_placebo'] = (r4_p['dist_km'] > 0).astype(float)

    # OLS with placebo sharia
    d = r4_p.dropna(subset=[outcome,'dist_km']).copy()
    d = d[d['dist_km'].abs() <= BW_MAIN]
    if len(d) < 50 or d['sharia_placebo'].nunique() < 2:
        print(f"  R4 {outcome}: insufficient data for placebo"); continue
    n_cl = d['ea_id'].nunique() if 'ea_id' in d.columns else 0
    cov = ('cluster', {'groups': d['ea_id']}) if n_cl >= 5 else ('HC3', {})
    try:
        m = smf.ols('y ~ sharia_placebo + x + x:sharia_placebo',
                    data=d.assign(y=d[outcome], x=d['dist_km'])
                    ).fit(cov_type=cov[0], cov_kwds=cov[1])
        c = m.params.get('sharia_placebo', np.nan)
        s = m.bse.get('sharia_placebo', np.nan)
        p = m.pvalues.get('sharia_placebo', np.nan)
        print(f"  R4 {outcome} PLACEBO (+0.5° north): β={c:+.3f} SE={s:.3f} "
              f"p={p:.3f}{star(p)} N={len(d)}")
    except Exception as e:
        print(f"  R4 {outcome} placebo error: {e}")

# ════════════════════════════════════════════════════════════════════════════
# 8. PLACEBO OUTCOMES (non-trust economic variables, should not be affected)
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('PLACEBO OUTCOMES (economic conditions — should not be affected by Sharia courts)')
print('="If Sharia affected these, results reflect a general development gap"')
print('='*80)

for label, df, outcomes in [
    ('R2', r2, [('econ_personal','Personal econ cond.'),
                ('econ_country','Country econ cond.')]),
    ('R4', r4, [('econ_personal','Personal econ cond.')]),
]:
    print(f"\n  {label}:")
    for outcome, desc in outcomes:
        if outcome not in df.columns: continue
        r = rd_ols(df, outcome, BW_MAIN)
        if r:
            print(f"  {desc:25}  β={r['coef']:+.3f} SE={r['se']:.3f} "
                  f"p={r['pval']:.3f}{star(r['pval'])} N={r['n']}")

# ════════════════════════════════════════════════════════════════════════════
# 9. KADUNA SENSITIVITY
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('KADUNA SENSITIVITY (exclude Kaduna from Sharia side)')
print('Southern Kaduna is largely Christian — may dilute "treated" boundary')
print('='*80)

# Count how many Kaduna respondents are in the boundary sample
for label, df in [('R2',r2),('R4',r4)]:
    near = df[df['dist_km'].abs() <= BW_MAIN].copy()
    kaduna_sharia = ((near['state_survey'] == 'Kaduna') & (near['sharia'] == 1)).sum()
    print(f"\n  {label}: Kaduna respondents on Sharia side within {BW_MAIN}km: {kaduna_sharia} "
          f"(of {(near['sharia']==1).sum()} total Sharia-side)")
    print(f"  {'Outcome':10}  {'With Kaduna':>15}  {'Without Kaduna':>18}  {'Δβ':>10}")
    print(f"  {'-'*60}")
    for outcome in ['police','courts','relig']:
        if outcome not in df.columns: continue
        r_full = rd_ols(df, outcome, BW_MAIN)
        # Exclude Kaduna by temporarily setting sharia=0 for Kaduna respondents
        df_no_kad = df.copy()
        df_no_kad.loc[df_no_kad['state_survey'] == 'Kaduna', 'sharia'] = 0
        r_nk = rd_ols(df_no_kad, outcome, BW_MAIN)
        if r_full and r_nk:
            delta = r_nk['coef'] - r_full['coef']
            print(f"  {outcome:<10}  {r_full['coef']:>+8.3f}{star(r_full['pval']):<6}  "
                  f"{r_nk['coef']:>+8.3f}{star(r_nk['pval']):<6}  {delta:>+10.3f}")

# ════════════════════════════════════════════════════════════════════════════
# 10. POOLED R2-R4 — standardized, round FE, R4 interaction test
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('POOLED R2-R4: Standardized outcomes (SD units), round FE, R4 interaction')
print('Addresses "immediate vs delayed" story properly')
print('="R4 interaction = test whether R4 coefficient significantly larger than R2/R3"')
print('='*80)

def standardize(series):
    m, s = series.mean(), series.std()
    return (series - m) / s if s > 0 else series * 0

for outcome, label in [('police','Police trust'),('courts','Courts trust'),
                        ('relig','Relig leaders')]:
    rows = []
    for rnd, yr, df in [('R2',2003,r2),('R3',2005,r3),('R4',2008,r4)]:
        if outcome not in df.columns: continue
        near = df[df['dist_km'].abs() <= BW_MAIN].copy()
        near = near.dropna(subset=[outcome,'dist_km'])
        near['round'] = rnd
        near['year'] = yr
        near['is_R4'] = (yr == 2008).real
        rows.append(near[['sharia','dist_km','ea_id',outcome,'round','year','is_R4',
                           'state_survey']].copy())
    if not rows: continue
    pool = pd.concat(rows, ignore_index=True)
    # Standardize outcome across all rounds
    pool['y_std'] = standardize(pool[outcome].dropna())
    pool = pool.dropna(subset=['y_std'])
    if len(pool) < 100: continue

    # Pooled OLS: y ~ sharia + dist_km + dist_km:sharia + C(round) + sharia:is_R4
    n_cl = pool['ea_id'].nunique()
    cov = ('cluster', {'groups': pool['ea_id']}) if n_cl >= 5 else ('HC3', {})
    try:
        m = smf.ols('y_std ~ sharia + x + x:sharia + C(round) + sharia:is_R4',
                    data=pool.assign(x=pool['dist_km'])
                   ).fit(cov_type=cov[0], cov_kwds=cov[1])
        c_base    = m.params.get('sharia', np.nan)
        p_base    = m.pvalues.get('sharia', np.nan)
        c_r4      = m.params.get('sharia:is_R4', np.nan)
        p_r4      = m.pvalues.get('sharia:is_R4', np.nan)
        print(f"\n  {label}:")
        print(f"    Pooled Sharia effect (R2/R3 base):  β={c_base:+.3f} p={p_base:.3f}{star(p_base)}")
        print(f"    R4 additional interaction:           β={c_r4:+.3f} p={p_r4:.3f}{star(p_r4)}")
        print(f"    N={len(pool)}, EAs={n_cl}")
    except Exception as e:
        print(f"  {label}: pooled model error: {e}")

# ════════════════════════════════════════════════════════════════════════════
# 11. CORRECTED CCT with masspoints='adjust' + cluster
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print("CCT OPTIMAL BW with masspoints='adjust' + EA cluster (reviewer fix #7)")
print('='*80)

print(f"\n  {'Round':5}  {'Outcome':10}  {'BW_h':>7}  {'β':>8}  {'SE':>7}  {'p':>8}")
print(f"  {'-'*55}")
for label, df in [('R2',r2),('R4',r4)]:
    for outcome in ['police','courts','relig']:
        if outcome not in df.columns: continue
        r = rd_cct(df, outcome, use_adjust=True)
        if r and 'error' not in r:
            h = (r['bw_l'] + r['bw_r']) / 2
            print(f"  {label:<5}  {outcome:<10}  {h:>7.1f}  {r['coef']:>+8.3f}  "
                  f"{r['se']:>7.3f}  {r['pval']:>7.3f}{star(r['pval'])}")
        elif r:
            print(f"  {label:<5}  {outcome:<10}  error: {r['error'][:60]}")

print('\nDiagnostics complete.')
