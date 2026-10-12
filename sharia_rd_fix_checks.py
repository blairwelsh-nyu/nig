"""
Diagnostic and fix script for six factual errors in the memo.

1. Muslim variable: q86 vs q111 in R2 — which is religion?
2. Muslim as RD outcome (religion balance check)
3. Early adopter recoding — Sokoto is Jan 31 2001, not January 2000
4. R4 interaction vs split-sample discrepancy
5. R6 courts discrepancy: +0.147 vs +0.321*
6. Religion balance in §8 (p=0.47) vs §9.2 (80.8% vs 32.7%)
"""
import os, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')

os.chdir('/home/user/nig')

SHARIA_STATES = {'Zamfara','Kano','Sokoto','Katsina','Bauchi','Niger',
                 'Jigawa','Kebbi','Yobe','Borno','Kaduna','Gombe'}

# Correct adopter timing (from Ostien):
# Zamfara: January 2000
# Niger: May 2000, Katsina: August 2000, Kano: November 2000, Kebbi: December 2000
# Sokoto: January 31, 2001 — NOT January 2000
# Jigawa: February 2001, Gombe: June 2001, Borno: June 2001, Bauchi: July 2001
# Yobe: December 2001, Kaduna: April 2002, Kano already in 2000

EARLY_SHARIA_WRONG  = {'Zamfara','Sokoto','Niger','Kano','Katsina','Kebbi'}  # current (wrong)
EARLY_SHARIA_FIXED  = {'Zamfara','Niger','Kano','Katsina','Kebbi'}            # 2000 only
LATE_SHARIA_FIXED   = {'Sokoto','Jigawa','Gombe','Bauchi','Yobe','Borno','Kaduna'}

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
        return {'coef': c, 'se': s, 'pval': p, 'n': len(d), 'n_cl': n_cl}
    except Exception: return None

# ── Load R2 ───────────────────────────────────────────────────────────────────
print("Loading R2...")
r2_raw = pd.read_excel('NIG_r2.csv.xlsx')
r2_raw.columns = r2_raw.columns.str.lower().str.strip()
r2_raw['lat'] = pd.to_numeric(r2_raw.get('latitude', r2_raw.get('lat')), errors='coerce')
r2_raw['lon'] = pd.to_numeric(r2_raw.get('longitude', r2_raw.get('lon')), errors='coerce')
r2_raw = r2_raw.dropna(subset=['lat','lon']).reset_index(drop=True)
r2_raw['state_survey'] = r2_raw['geoname_adm_name'].apply(extract_state)
r2_raw['sharia'] = r2_raw['state_survey'].isin(SHARIA_STATES).astype(float)
r2_raw['dist_km'] = [signed_dist_km(r['lat'], r['lon']) for _, r in r2_raw.iterrows()]
ea_col = next((c for c in ['uniqueea','eanumb'] if c in r2_raw.columns and r2_raw[c].nunique() > 1), None)
r2_raw['ea_id'] = r2_raw[ea_col] if ea_col else np.arange(len(r2_raw))
for col in ['police','courts']:
    src = 'q43i' if col == 'police' else 'q43j'
    v = pd.to_numeric(r2_raw[src], errors='coerce')
    r2_raw[col] = v.where(v <= 3)

near_r2 = r2_raw[r2_raw['dist_km'].abs() <= 150].copy()

# ── Load R4 ───────────────────────────────────────────────────────────────────
print("Loading R4...")
r4_raw = pd.read_excel('NIG_r4.csv.xlsx')
r4_raw.columns = r4_raw.columns.str.lower().str.strip()
r4_raw['lat'] = pd.to_numeric(r4_raw.get('latitude', r4_raw.get('lat')), errors='coerce')
r4_raw['lon'] = pd.to_numeric(r4_raw.get('longitude', r4_raw.get('lon')), errors='coerce')
r4_raw = r4_raw.dropna(subset=['lat','lon']).reset_index(drop=True)
r4_raw['state_survey'] = r4_raw['geoname_adm_name'].apply(extract_state)
r4_raw['sharia'] = r4_raw['state_survey'].isin(SHARIA_STATES).astype(float)
r4_raw['dist_km'] = [signed_dist_km(r['lat'], r['lon']) for _, r in r4_raw.iterrows()]
ea_col4 = next((c for c in ['uniqueea','eanumb'] if c in r4_raw.columns and r4_raw[c].nunique() > 1), None)
r4_raw['ea_id'] = r4_raw[ea_col4] if ea_col4 else np.arange(len(r4_raw))
for col, src in [('police','q49c'),('courts','q49d'),('relig','q49i')]:
    v = pd.to_numeric(r4_raw[src], errors='coerce')
    r4_raw[col] = v.where(v <= 3)

near_r4 = r4_raw[r4_raw['dist_km'].abs() <= 150].copy()

# ── Load R6 ───────────────────────────────────────────────────────────────────
print("Loading R6...")
r6_raw = pd.read_excel('NIG_r6.csv.xlsx')
r6_raw.columns = r6_raw.columns.str.lower().str.strip()
r6_raw['lat'] = pd.to_numeric(r6_raw.get('latitude', r6_raw.get('lat')), errors='coerce')
r6_raw['lon'] = pd.to_numeric(r6_raw.get('longitude', r6_raw.get('lon')), errors='coerce')
r6_raw = r6_raw.dropna(subset=['lat','lon']).reset_index(drop=True)
r6_raw['state_survey'] = r6_raw['geoname_adm_name'].apply(extract_state)
r6_raw['sharia'] = r6_raw['state_survey'].isin(SHARIA_STATES).astype(float)
r6_raw['dist_km'] = [signed_dist_km(r['lat'], r['lon']) for _, r in r6_raw.iterrows()]
ea_col6 = next((c for c in ['uniqueea','eanumb'] if c in r6_raw.columns and r6_raw[c].nunique() > 1), None)
r6_raw['ea_id'] = r6_raw[ea_col6] if ea_col6 else np.arange(len(r6_raw))
for col, src in [('courts_q52d','q52d'), ('courts_alt','q52c')]:  # q52c=police, q52d=courts
    v = pd.to_numeric(r6_raw.get(src, pd.Series(dtype=float)), errors='coerce')
    r6_raw[col] = v.where(v <= 3)
# Also check what columns exist for trust
r6_trust_cols = [c for c in r6_raw.columns if c.startswith('q52')]
print(f"  R6 q52* columns: {r6_trust_cols[:10]}")

near_r6 = r6_raw[r6_raw['dist_km'].abs() <= 150].copy()

print(f"  R2 boundary N={len(near_r2)}, R4 boundary N={len(near_r4)}, R6 boundary N={len(near_r6)}")

# ════════════════════════════════════════════════════════════════════════════
# CHECK 1: q86 vs q111 — which is religion in R2?
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('CHECK 1: Muslim variable — q86 vs q111 in R2')
print('='*80)

for col_name in ['q86','q111','q112','q113']:
    if col_name not in r2_raw.columns:
        print(f"  {col_name}: not in R2 data")
        continue
    v = pd.to_numeric(r2_raw[col_name], errors='coerce')
    print(f"\n  {col_name}: {v.nunique()} distinct values, range {v.min():.0f}–{v.max():.0f}")
    vc = near_r2[col_name].value_counts().sort_index() if col_name in near_r2.columns else pd.Series()
    for code, cnt in vc.head(10).items():
        s_pct = near_r2[(near_r2['sharia']==1) & (near_r2[col_name]==code)].shape[0]
        n_pct = near_r2[(near_r2['sharia']==0) & (near_r2[col_name]==code)].shape[0]
        s_tot = (near_r2['sharia']==1).sum()
        n_tot = (near_r2['sharia']==0).sum()
        print(f"    code={code}: total={cnt:3d}  sharia={s_pct/s_tot:.1%}  non-sharia={n_pct/n_tot:.1%}")

# ════════════════════════════════════════════════════════════════════════════
# CHECK 2: Muslim as RD outcome (religion balance)
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('CHECK 2: Muslim as RD outcome — is the religion gap at boundary significant?')
print('§8 claims religion balance p=0.47; §9.2 reports 80.8% vs 32.7%')
print('='*80)

# Try both q86 and q111 as Muslim indicators
for col, muslim_code in [('q86', 6), ('q111', 6)]:
    if col not in r2_raw.columns:
        print(f"\n  {col}: not in R2 data"); continue
    r2_raw[f'muslim_{col}'] = (pd.to_numeric(r2_raw[col], errors='coerce') == muslim_code).astype(float)
    r = rd_ols(r2_raw, f'muslim_{col}')
    if r:
        s_mean = near_r2[near_r2['sharia']==1][col].apply(lambda x: 1 if x==muslim_code else 0).mean()
        n_mean = near_r2[near_r2['sharia']==0][col].apply(lambda x: 1 if x==muslim_code else 0).mean()
        print(f"\n  {col}==6 as Muslim proxy:")
        print(f"    Sharia-side mean={s_mean:.2%}, non-Sharia mean={n_mean:.2%}")
        print(f"    RD: β={r['coef']:>+.3f}{star(r['pval'])} SE={r['se']:.3f} p={r['pval']:.3f} N={r['n']} clusters={r['n_cl']}")
    else:
        print(f"\n  {col}: RD failed")

# Also check R4 Muslim
print("\n  R4 religion check (q85):")
if 'q85' in r4_raw.columns:
    v85 = pd.to_numeric(r4_raw['q85'], errors='coerce')
    print(f"  R4 q85 value counts (boundary sample):")
    vc = pd.to_numeric(r4_raw.loc[r4_raw['dist_km'].abs()<=150, 'q85'], errors='coerce').value_counts().sort_index()
    for code, cnt in vc.items():
        r4_tmp = r4_raw[r4_raw['dist_km'].abs()<=150].copy()
        r4_tmp['tmp_v'] = (pd.to_numeric(r4_tmp['q85'], errors='coerce') == code).astype(float)
        s = r4_tmp[r4_tmp['sharia']==1]['tmp_v'].mean()
        n = r4_tmp[r4_tmp['sharia']==0]['tmp_v'].mean()
        print(f"    code={code}: N={cnt}  sharia={s:.1%}  non-sharia={n:.1%}")
    for muslim_code in [1, 6]:
        col_name = f'muslim_r4_{muslim_code}'
        r4_raw[col_name] = (pd.to_numeric(r4_raw['q85'], errors='coerce') == muslim_code).astype(float)
        r = rd_ols(r4_raw, col_name)
        if r:
            r4_bnd = r4_raw[r4_raw['dist_km'].abs()<=150]
            s_mean = r4_bnd[r4_bnd['sharia']==1][col_name].mean()
            n_mean = r4_bnd[r4_bnd['sharia']==0][col_name].mean()
            print(f"    q85=={muslim_code}: sharia={s_mean:.2%}, non-sharia={n_mean:.2%}, β={r['coef']:>+.3f}{star(r['pval'])} p={r['pval']:.3f}")

# ════════════════════════════════════════════════════════════════════════════
# CHECK 3: Corrected early adopter coding
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('CHECK 3: Corrected early adopter coding (Sokoto = Jan 31 2001, not Jan 2000)')
print('Old EARLY: Zamfara, Sokoto, Niger, Kano, Katsina, Kebbi')
print('New EARLY (year-2000): Zamfara, Niger, Kano, Katsina, Kebbi')
print('New LATE  (2001+):     Sokoto, Jigawa, Gombe, Bauchi, Yobe, Borno, Kaduna')
print('='*80)

for rnd, df_raw in [('R2', r2_raw), ('R4', r4_raw)]:
    d = df_raw[df_raw['dist_km'].abs() <= 150].copy()
    # Old coding
    d['early_old'] = ((d['sharia']==1) & d['state_survey'].isin(EARLY_SHARIA_WRONG)).astype(float)
    d['late_old']  = ((d['sharia']==1) & ~d['state_survey'].isin(EARLY_SHARIA_WRONG)).astype(float)
    # New coding
    d['early_new'] = ((d['sharia']==1) & d['state_survey'].isin(EARLY_SHARIA_FIXED)).astype(float)
    d['late_new']  = ((d['sharia']==1) & ~d['state_survey'].isin(EARLY_SHARIA_FIXED)).astype(float)

    # Show Sokoto N
    sokoto_n = int((d['state_survey']=='Sokoto').sum())
    print(f"\n  {rnd}: Sokoto boundary N={sokoto_n}")
    print(f"  Old: early N={int(d['early_old'].sum())}, late N={int(d['late_old'].sum())}")
    print(f"  New: early N={int(d['early_new'].sum())}, late N={int(d['late_new'].sum())}")

    outcomes = [o for o in ['police','courts','relig'] if o in d.columns]
    print(f"\n  {rnd} corrected early/late results:")
    print(f"  {'Outcome':10}  {'Early (2000) β':>16}  {'Late (2001+) β':>16}  {'F-test p':>10}")
    for outcome in outcomes:
        dd = d.dropna(subset=[outcome,'dist_km'])
        n_cl = dd['ea_id'].nunique()
        cov = ('cluster', {'groups': dd['ea_id']}) if n_cl >= 5 else ('HC3', {})
        try:
            m = smf.ols('y ~ early_new + late_new + x + x:early_new + x:late_new',
                        data=dd.assign(y=dd[outcome], x=dd['dist_km'])
                        ).fit(cov_type=cov[0], cov_kwds=cov[1])
            c_e = m.params.get('early_new', np.nan)
            c_l = m.params.get('late_new', np.nan)
            p_e = m.pvalues.get('early_new', np.nan)
            p_l = m.pvalues.get('late_new', np.nan)
            try:
                fp = float(m.f_test(['early_new - late_new = 0']).pvalue)
            except Exception: fp = np.nan
            print(f"  {outcome:<10}  {c_e:>+8.3f}{star(p_e):<6}  {c_l:>+8.3f}{star(p_l):<6}  {fp:.3f}")
        except Exception as e:
            print(f"  {outcome:<10}  error: {e}")

# ════════════════════════════════════════════════════════════════════════════
# CHECK 4: R4 interaction vs split-sample — slope specification
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('CHECK 4: R4 interaction specification — why is int p=0.79 when gap is ~0.7?')
print('Testing full interaction spec (slopes also interacted) vs partial (level only)')
print('='*80)

# Determine R4 Muslim variable
if 'q85' in r4_raw.columns:
    v85 = pd.to_numeric(r4_raw['q85'], errors='coerce')
    # Most likely code 1=Muslim
    print(f"\n  R4 q85 value counts (boundary sample):")
    vc = near_r4['q85'].value_counts().sort_index() if 'q85' in near_r4.columns else pd.Series()
    for code, cnt in vc.items():
        print(f"    code={code}: N={cnt}")

    # Try code 1 as Muslim (Islam)
    r4_raw['muslim_r4'] = (pd.to_numeric(r4_raw['q85'], errors='coerce') == 1).astype(float)
    near_r4['muslim_r4'] = (pd.to_numeric(near_r4['q85'], errors='coerce') == 1).astype(float)

    s_pct = near_r4[near_r4['sharia']==1]['muslim_r4'].mean()
    n_pct = near_r4[near_r4['sharia']==0]['muslim_r4'].mean()
    print(f"\n  R4 q85==1 Muslim: sharia={s_pct:.2%}, non-sharia={n_pct:.2%}")

    d = near_r4.dropna(subset=['courts','dist_km','muslim_r4'])
    n_cl = d['ea_id'].nunique()
    cov = ('cluster', {'groups': d['ea_id']}) if n_cl >= 5 else ('HC3', {})

    print(f"\n  Model A: partial interaction (level only, slopes NOT interacted):")
    print(f"    y ~ sharia + muslim + sharia:muslim + x + x:sharia")
    try:
        m_a = smf.ols('y ~ sharia + muslim_r4 + sharia:muslim_r4 + x + x:sharia',
                      data=d.assign(y=d['courts'], x=d['dist_km'])
                      ).fit(cov_type=cov[0], cov_kwds=cov[1])
        print(f"    sharia (non-Muslim): β={m_a.params.get('sharia',np.nan):>+.3f}{star(m_a.pvalues.get('sharia',np.nan))}")
        print(f"    sharia:muslim:      β={m_a.params.get('sharia:muslim_r4',np.nan):>+.3f}{star(m_a.pvalues.get('sharia:muslim_r4',np.nan))} p={m_a.pvalues.get('sharia:muslim_r4',np.nan):.3f}")
        print(f"    implied Muslim eff: {m_a.params.get('sharia',np.nan)+m_a.params.get('sharia:muslim_r4',np.nan):>+.3f}")
    except Exception as e:
        print(f"    failed: {e}")

    print(f"\n  Model B: full interaction (slopes also interacted):")
    print(f"    y ~ sharia + muslim + sharia:muslim + x + x:sharia + x:muslim + x:sharia:muslim")
    try:
        m_b = smf.ols('y ~ sharia + muslim_r4 + sharia:muslim_r4 + x + x:sharia + x:muslim_r4 + x:sharia:muslim_r4',
                      data=d.assign(y=d['courts'], x=d['dist_km'])
                      ).fit(cov_type=cov[0], cov_kwds=cov[1])
        print(f"    sharia (non-Muslim): β={m_b.params.get('sharia',np.nan):>+.3f}{star(m_b.pvalues.get('sharia',np.nan))}")
        print(f"    sharia:muslim:      β={m_b.params.get('sharia:muslim_r4',np.nan):>+.3f}{star(m_b.pvalues.get('sharia:muslim_r4',np.nan))} p={m_b.pvalues.get('sharia:muslim_r4',np.nan):.3f}")
    except Exception as e:
        print(f"    failed: {e}")

    print(f"\n  Split-sample for comparison:")
    mus  = near_r4[near_r4['muslim_r4']==1].copy()
    nmus = near_r4[near_r4['muslim_r4']==0].copy()
    rm = rd_ols(mus, 'courts')
    rn = rd_ols(nmus, 'courts')
    if rm: print(f"    Muslim:     β={rm['coef']:>+.3f}{star(rm['pval'])} N={rm['n']}")
    if rn: print(f"    Non-Muslim: β={rn['coef']:>+.3f}{star(rn['pval'])} N={rn['n']}")

# ════════════════════════════════════════════════════════════════════════════
# CHECK 5: R6 courts discrepancy — +0.147 vs +0.321*
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('CHECK 5: R6 courts discrepancy — §5 says +0.147, §9.1 says +0.321*')
print('Testing both q52c and q52d columns, checking BW')
print('='*80)

for col_name, label in [('courts_q52d','q52d (courts)'), ('courts_alt','q52c (police?)')]:
    if col_name not in r6_raw.columns: continue
    r_150 = rd_ols(r6_raw, col_name, bw=150)
    r_100 = rd_ols(r6_raw, col_name, bw=100)
    r_200 = rd_ols(r6_raw, col_name, bw=200)
    if r_150:
        print(f"\n  R6 {label}:")
        print(f"    BW=150: β={r_150['coef']:>+.3f}{star(r_150['pval'])} p={r_150['pval']:.3f} N={r_150['n']} cl={r_150['n_cl']}")
    if r_100:
        print(f"    BW=100: β={r_100['coef']:>+.3f}{star(r_100['pval'])} p={r_100['pval']:.3f} N={r_100['n']}")
    if r_200:
        print(f"    BW=200: β={r_200['coef']:>+.3f}{star(r_200['pval'])} p={r_200['pval']:.3f} N={r_200['n']}")

# Also check if the §9.1 +0.321* is coming from a different subsample
# NW test uses lon < 9/10/11°E restriction — the "Full" R6 result
print("\n  R6 full sample check (all lon):")
r_full = rd_ols(r6_raw, 'courts_q52d', bw=150)
if r_full:
    print(f"    Full BW=150: β={r_full['coef']:>+.3f}{star(r_full['pval'])} p={r_full['pval']:.3f} N={r_full['n']}")
    s_side = r6_raw[r6_raw['dist_km'].abs() <= 150].groupby('sharia').size()
    print(f"    R6 boundary sample: Sharia={s_side.get(1.0,0)}, non-Sharia={s_side.get(0.0,0)}")

print()
print('All checks complete.')
