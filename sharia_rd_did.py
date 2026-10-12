"""
Difference-in-Discontinuities (DiD-RD) test.

Design: R1 (2000, pre-Sharia baseline) vs R2 (2003) and R4 (2008) post-treatment.
R1 trspol (1-4 → 0-3) and trscts (1-4 → 0-3) are scale-comparable to R2 police/courts.
The DiD removes any time-invariant confound at the boundary (North/Middle Belt gradient).

DiD estimate: (Sharia-side jump at Rt) - (Sharia-side jump at R1)
If R1 shows a negative pre-existing discontinuity and R2/R4 show positive effects,
the DiD amplifies the estimate and strengthens the causal interpretation.

Key assumption: The pre-treatment discontinuity is stable over time in the absence of
treatment (parallel trends in discontinuities).

Three estimators:
  A. OLS DiD: pool R1+R2 (or R1+R4), sharia×post interaction in local-linear OLS
  B. rdrobust DiD: run rdrobust separately on R1 and R2/R4, compute DiD manually
  C. Show R1-only RD to document the baseline discontinuity with scale-comparable items
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
nga['is_sharia'] = nga['state'].isin(SHARIA_STATES)
sharia_union    = unary_union(nga[nga['is_sharia']].geometry)
nonsharia_union = unary_union(nga[~nga['is_sharia']].geometry)
official_boundary = sharia_union.boundary.intersection(nonsharia_union.boundary)

def signed_dist_km(lat, lon):
    p = Point(lon, lat)
    lat_rad = np.radians(lat)
    dlat = np.pi * 6371.0 / 180.0
    dlon = dlat * np.cos(lat_rad)
    km = official_boundary.distance(p) * np.sqrt(dlat**2 + dlon**2) / np.sqrt(2)
    return (1.0 if sharia_union.contains(p) else -1.0) * km

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
    df['dist_km'] = [signed_dist_km(r['lat'], r['lon']) for _, r in df.iterrows()]
    ea_col = next((c for c in ['uniqueea','eanumb'] if c in df.columns
                   and df[c].nunique() > 1), None)
    df['ea_id'] = df[ea_col].astype(str) if ea_col else np.arange(len(df)).astype(str)
    for name, (col, recode) in outcomes_map.items():
        if col not in df.columns:
            df[name] = np.nan; continue
        v = pd.to_numeric(df[col], errors='coerce')
        if recode == '0-3':          v = v.where(v <= 3)
        elif recode == '1-5->0-3':   v = (v.where(v <= 5) - 1) * (3/4)
        elif recode == '1-4->0-3':   v = v.where(v <= 4) - 1   # R1: 1-4 → 0-3
        elif recode == '0-4':        v = v.where(v <= 4)
        df[name] = v
    return df

def rd_ols(df, outcome, bw=BW_MAIN):
    d = df.dropna(subset=[outcome,'dist_km']).copy()
    d = d[d['dist_km'].abs() <= bw]
    if len(d) < 30 or d[outcome].nunique() < 2: return None
    if d['sharia'].nunique() < 2: return None
    n_cl = d['ea_id'].nunique()
    cov = ('cluster',{'groups':d['ea_id']}) if n_cl >= 5 else ('HC3',{})
    try:
        m = smf.ols('y ~ sharia + x + x:sharia',
                    data=d.assign(y=d[outcome], x=d['dist_km'])
                    ).fit(cov_type=cov[0], cov_kwds=cov[1])
        c = m.params.get('sharia', np.nan)
        s = m.bse.get('sharia', np.nan)
        p = m.pvalues.get('sharia', np.nan)
        return {'coef': c, 'se': s, 'pval': p, 'n': len(d), 'n_cl': n_cl}
    except Exception: return None

def rd_ccт(df, outcome, bw=BW_MAIN):
    d = df.dropna(subset=[outcome,'dist_km','ea_id']).copy()
    d = d[d['dist_km'].abs() <= bw]
    if len(d) < 50 or d[outcome].nunique() < 2: return None
    try:
        r = rdrobust(d[outcome].values, d['dist_km'].values,
                     masspoints='adjust',
                     cluster=d['ea_id'].values,
                     bwselect='mserd')
        try:    coef = float(r.coef.loc['Robust','Coeff'])
        except: coef = float(r.coef.iloc[2,0])
        try:    se   = float(r.se.loc['Robust','Std. Err.'])
        except: se   = float(r.se.iloc[2,0])
        try:    pval = float(r.pv.loc['Robust','P>|z|'])
        except: pval = float(r.pv.iloc[2,0])
        try:    bw_h = float(r.bws.loc['h','left'])
        except: bw_h = float(r.bws.iloc[0,0])
        return {'coef': coef, 'se': se, 'pval': pval, 'bw': bw_h}
    except Exception as e:
        return {'error': str(e)}

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading rounds...")
r1 = load_round('NIG_r1.csv.xlsx', {
    'police': ('trspol', '1-4->0-3'),
    'courts': ('trscts', '1-4->0-3'),
})
r2 = load_round('NIG_r2.csv.xlsx', {
    'police': ('q43i', '0-3'),
    'courts': ('q43j', '0-3'),
})
r4 = load_round('NIG_r4.csv.xlsx', {
    'police': ('q49c', '0-3'),
    'courts': ('q49d', '0-3'),
})
r1['round'] = 'R1';  r2['round'] = 'R2';  r4['round'] = 'R4'

for df, lbl in [(r1,'R1'),(r2,'R2'),(r4,'R4')]:
    bw_df = df[df['dist_km'].abs() <= BW_MAIN]
    n_s  = (bw_df['sharia']==1).sum()
    n_ns = (bw_df['sharia']==0).sum()
    print(f"  {lbl}: total={len(df)}, boundary sample (±{BW_MAIN}km): "
          f"Sharia={n_s}, non-Sharia={n_ns}")

# ════════════════════════════════════════════════════════════════════════════════
# STEP 1: R1 BASELINE — scale-comparable trust items
# ════════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('STEP 1: R1 (2000) RD ON SCALE-COMPARABLE ITEMS (trspol, trscts → 0–3)')
print('This is the pre-treatment or very-early-rollout baseline.')
print('Zamfara adopted January 2000 but was NOT sampled in R1.')
print('All other Sharia states adopted May 2000 or later; R1 field dates unknown.')
print('='*80)
print()
print(f"  {'Outcome':<10} {'OLS β':<12} {'SE':<8} {'p':<10} {'CCT β':<12} {'CCT p'}")
print("  " + "-"*55)

r1_cct = {}
r1_ols = {}
for outcome in ['police','courts']:
    ols = rd_ols(r1, outcome)
    cct = rd_ccт(r1, outcome)
    r1_ols[outcome] = ols
    r1_cct[outcome] = cct
    ols_str = (f"{ols['coef']:+.3f}{star(ols['pval'])}" if ols else 'n/a')
    ols_se  = (f"{ols['se']:.3f}" if ols else '')
    ols_p   = (f"{ols['pval']:.3f}" if ols else '')
    cct_str = (f"{cct['coef']:+.3f}{star(cct['pval'])}" if cct and 'coef' in cct else
               f"err:{cct.get('error','')[:20]}" if cct else 'n/a')
    cct_p   = (f"{cct['pval']:.3f}" if cct and 'coef' in cct else '')
    print(f"  {outcome:<10} {ols_str:<12} {ols_se:<8} {ols_p:<10} {cct_str:<12} {cct_p}")

# ════════════════════════════════════════════════════════════════════════════════
# STEP 2: DiD-RD OLS — pool pre (R1) and post (R2 or R4)
# Estimating equation:
#   y = α + τ·sharia + γ·post + δ·(sharia×post)
#       + β₁·x + β₂·(x·sharia) + β₃·(x·post) + β₄·(x·sharia·post) + ε
# δ = DiD = τ_post - τ_pre
# Cluster: ea×round (separate respondents across rounds)
# ════════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('STEP 2: DiD-RD OLS (pool R1 with R2 or R4)')
print('DiD = sharia × post coefficient = τ_post - τ_pre')
print('Removes time-invariant confound at boundary (cultural gradient, geography).')
print('Cluster: EA×round (R1 and R2 respondents in same EA treated as independent).')
print('='*80)

for post_label, post_df in [('R2 (2003)', r2), ('R4 (2008)', r4)]:
    print(f"\n  Pre=R1 vs Post={post_label}:")
    print(f"  {'Outcome':<10} {'τ_R1 (pre)':<14} {'τ_post':<14} {'DiD (δ)':<14} {'p(DiD)'}")
    print("  " + "-"*60)

    for outcome in ['police','courts']:
        # Stack R1 + post round
        d_pre  = r1[['dist_km','sharia','ea_id',outcome,'round']].copy()
        d_post = post_df[['dist_km','sharia','ea_id',outcome,'round']].copy()
        d_pre['post']  = 0.0
        d_post['post'] = 1.0
        stacked = pd.concat([d_pre, d_post], ignore_index=True)
        stacked = stacked.dropna(subset=[outcome,'dist_km']).copy()
        stacked = stacked[stacked['dist_km'].abs() <= BW_MAIN]
        stacked['ea_round'] = stacked['ea_id'].astype(str) + '_' + stacked['round']

        if len(stacked) < 80 or stacked['sharia'].nunique() < 2: continue
        n_cl = stacked['ea_round'].nunique()
        cov = ('cluster',{'groups':stacked['ea_round']}) if n_cl >= 5 else ('HC3',{})

        try:
            m = smf.ols(
                'y ~ sharia + post + sharia:post + x + x:sharia + x:post + x:sharia:post',
                data=stacked.assign(y=stacked[outcome], x=stacked['dist_km'])
            ).fit(cov_type=cov[0], cov_kwds=cov[1])

            tau_pre  = m.params.get('sharia', np.nan)
            tau_post = tau_pre + m.params.get('sharia:post', np.nan)
            did      = m.params.get('sharia:post', np.nan)
            p_did    = m.pvalues.get('sharia:post', np.nan)
            se_did   = m.bse.get('sharia:post', np.nan)

            pre_str  = f"{tau_pre:+.3f}"
            post_str = f"{tau_post:+.3f}"
            did_str  = f"{did:+.3f}{star(p_did)}"
            p_str    = f"p={p_did:.3f}"
            print(f"  {outcome:<10} {pre_str:<14} {post_str:<14} {did_str:<14} {p_str}")

            # Also print n and clusters
            n_pre  = (stacked['post']==0).sum()
            n_post = (stacked['post']==1).sum()
            print(f"            N_pre={n_pre}, N_post={n_post}, clusters={n_cl}")

        except Exception as e:
            print(f"  {outcome}: FAILED — {e}")

# ════════════════════════════════════════════════════════════════════════════════
# STEP 3: DiD-RD MANUAL (rdrobust separately, delta method for SE)
# ════════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('STEP 3: DiD USING rdrobust SEPARATELY ON R1 AND R2/R4')
print('DiD = β_post - β_R1 (CCT estimates, masspoints=adjust, EA cluster)')
print('SE(DiD) = sqrt(SE_post² + SE_R1²)  [assumes independence across rounds]')
print('='*80)

for post_label, post_df in [('R2', r2), ('R4', r4)]:
    print(f"\n  R1 vs {post_label}:")
    print(f"  {'Outcome':<10} {'β_R1 (CCT)':<16} {'β_post (CCT)':<16} {'DiD':<12} {'Z':<10} {'p'}")
    print("  " + "-"*68)
    for outcome in ['police','courts']:
        cct_pre  = r1_cct.get(outcome)
        cct_post = rd_ccт(post_df, outcome)
        if cct_pre is None or 'coef' not in cct_pre:
            print(f"  {outcome:<10} R1 CCT failed"); continue
        if cct_post is None or 'coef' not in cct_post:
            print(f"  {outcome:<10} post CCT failed"); continue

        b_pre  = cct_pre['coef'];   se_pre  = cct_pre['se']
        b_post = cct_post['coef'];  se_post = cct_post['se']
        did    = b_post - b_pre
        se_did = np.sqrt(se_pre**2 + se_post**2)
        z      = did / se_did
        from scipy import stats
        p_did  = 2 * (1 - stats.norm.cdf(abs(z)))

        pre_str  = f"{b_pre:+.3f} (SE={se_pre:.3f})"
        post_str = f"{b_post:+.3f} (SE={se_post:.3f})"
        did_str  = f"{did:+.3f}{star(p_did)}"
        z_str    = f"{z:+.2f}"
        p_str    = f"p={p_did:.3f}"
        print(f"  {outcome:<10} {pre_str:<16} {post_str:<16} {did_str:<12} {z_str:<10} {p_str}")

# ════════════════════════════════════════════════════════════════════════════════
# STEP 4: Parallel-trends check
# Show R1 RD separately by state (Sharia vs non-Sharia side) to see if pre-
# existing discontinuity is uniform across the boundary or concentrated in
# specific stretches (which would indicate it's not a stable baseline).
# Also: compare covariate balance at R1 vs R2 to assess stability.
# ════════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('STEP 4: COVARIATE BALANCE AT R1 — does the pre-treatment discontinuity')
print('appear in placebo outcomes (economic conditions) as well as trust?')
print('If economic conditions also jump at R1, the confound is general, not trust-specific.')
print('='*80)

# Load R1 with economic condition items if available
r1_full = pd.read_excel('NIG_r1.csv.xlsx')
r1_full.columns = r1_full.columns.str.lower().str.strip()
lat_col = next((c for c in r1_full.columns if c in ('latitude','lat')), None)
lon_col = next((c for c in r1_full.columns if c in ('longitude','lon')), None)
r1_full = r1_full.rename(columns={lat_col:'lat', lon_col:'lon'})
r1_full['lat'] = pd.to_numeric(r1_full['lat'], errors='coerce')
r1_full['lon'] = pd.to_numeric(r1_full['lon'], errors='coerce')
r1_full = r1_full.dropna(subset=['lat','lon']).reset_index(drop=True)
r1_full['state_survey'] = r1_full['geoname_adm_name'].apply(extract_state)
r1_full['sharia'] = r1_full['state_survey'].isin(SHARIA_STATES).astype(float)
r1_full['dist_km'] = [signed_dist_km(row['lat'], row['lon'])
                      for _, row in r1_full.iterrows()]
ea_col = next((c for c in ['uniqueea','eanumb'] if c in r1_full.columns
               and r1_full[c].nunique() > 1), None)
r1_full['ea_id'] = r1_full[ea_col].astype(str) if ea_col else np.arange(len(r1_full)).astype(str)

# Potential placebo outcomes in R1
placebos = {}
for col, label in [('povfoo','food poverty'), ('povinc','income poverty'),
                   ('urbrur','urban'), ('age','age'), ('educ','education')]:
    if col in r1_full.columns:
        v = pd.to_numeric(r1_full[col], errors='coerce')
        # Recode: urbrur 1=urban 2=rural → is_urban
        if col == 'urbrur': v = (v == 1).astype(float)
        elif col in ('povfoo','povinc'): v = v.where(v <= 4)  # 1-4 scale
        r1_full[label] = v
        placebos[label] = label

print(f"\n  {'Outcome':<20} {'β':<12} {'SE':<8} {'p'}")
print("  " + "-"*45)
for label in placebos.values():
    res = rd_ols(r1_full, label)
    if res:
        sig = star(res['pval'])
        print(f"  {label:<20} {res['coef']:+.3f}{sig:<8}  {res['se']:.3f}   p={res['pval']:.3f}")
    else:
        print(f"  {label:<20} n/a")

# Also test R1 trspol/trscts covariate balance
r1_full['police'] = pd.to_numeric(r1_full['trspol'], errors='coerce').pipe(
    lambda v: v.where(v <= 4) - 1)
r1_full['courts'] = pd.to_numeric(r1_full['trscts'], errors='coerce').pipe(
    lambda v: v.where(v <= 4) - 1)

print()
print("  Trust outcomes at R1 (for comparison):")
for outcome in ['police','courts']:
    res = rd_ols(r1_full, outcome)
    if res:
        sig = star(res['pval'])
        print(f"  {outcome:<20} {res['coef']:+.3f}{sig:<8}  {res['se']:.3f}   "
              f"p={res['pval']:.3f}  (N={res['n']}, clusters={res['n_cl']})")

print('\nDiD-RD tests complete.')
