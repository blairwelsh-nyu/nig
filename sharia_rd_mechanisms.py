"""
Mechanism tests and northwest boundary check for Sharia RD analysis.

Tests:
  1. Northwest boundary — R5/R6 restricted to lon < 10°E (away from Boko Haram core)
  2. Muslim HTE — Sharia × Muslim interaction
  3. Non-Muslim spillover — is there an effect for non-Muslim respondents?
  4. Fear of crime (R4) — security mechanism
  5. Urban/rural HTE (R4) — are effects concentrated in urban centres?
  6. Early vs. late Sharia adopter HTE — exposure-duration test
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
# States that adopted Sharia in 2000 (earliest cohort)
EARLY_SHARIA = {'Zamfara','Sokoto','Niger','Kano','Katsina','Kebbi'}
# States in the Boko Haram zone (northeast)
BH_STATES = {'Borno','Yobe','Adamawa','Gombe'}

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
        if abs(c) > 3 or s > 2: return None
        return {'coef': c, 'se': s, 'pval': p, 'n': len(d), 'n_cl': n_cl}
    except Exception: return None

def rd_hte(df, outcome, moderator, bw=150):
    """
    HTE: y ~ sharia + sharia:mod + x + x:sharia + mod (+ x:mod optional)
    Returns effect for mod=0, mod=1, and interaction coefficient.
    """
    d = df.dropna(subset=[outcome,'dist_km',moderator]).copy()
    d = d[d['dist_km'].abs() <= bw]
    if len(d) < 80 or d[outcome].nunique() < 2: return None
    if d['sharia'].nunique() < 2: return None
    n_cl = d['ea_id'].nunique() if 'ea_id' in d.columns else 0
    cov = ('cluster', {'groups': d['ea_id']}) if n_cl >= 5 else ('HC3', {})
    try:
        m = smf.ols('y ~ sharia + sharia:mod + x + x:sharia + mod',
                    data=d.assign(y=d[outcome], x=d['dist_km'],
                                  mod=d[moderator])
                    ).fit(cov_type=cov[0], cov_kwds=cov[1])
        c_main = m.params.get('sharia', np.nan)
        c_int  = m.params.get('sharia:mod', np.nan)
        p_main = m.pvalues.get('sharia', np.nan)
        p_int  = m.pvalues.get('sharia:mod', np.nan)
        # Effect at mod=0 (baseline) and mod=1
        eff_0 = c_main
        eff_1 = c_main + c_int
        return {'base': c_main, 'int': c_int, 'p_base': p_main, 'p_int': p_int,
                'eff_0': eff_0, 'eff_1': eff_1, 'n': len(d)}
    except Exception: return None

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
        if recode == '0-3':        v = v.where(v <= 3)
        elif recode == '1-5->0-3': v = (v.where(v <= 5) - 1) * (3/4)
        elif recode == '0-4':      v = v.where(v <= 4)
        elif recode == '1-4':      v = v.where(v <= 4) - 1
        elif recode == 'raw':      pass
        df[name] = v
    return df

# ── Probe religion columns in each round ─────────────────────────────────────
def find_religion_col(fname):
    df = pd.read_excel(fname)
    df.columns = df.columns.str.lower()
    candidates = [c for c in df.columns if
                  any(k in c for k in ['relig','q87','q88','q89','q90','q95',
                                        'q97','q98','q99','q100'])]
    results = {}
    for c in candidates:
        v = pd.to_numeric(df[c], errors='coerce').dropna()
        if len(v) > 100:
            vc = v.value_counts().sort_index().head(8).to_dict()
            results[c] = vc
    return results

print("Probing religion columns...")
for rnd, fname in [('R2','NIG_r2.csv.xlsx'),('R3','NIG_r3.csv.xlsx'),
                   ('R4','NIG_r4.csv.xlsx'),('R5','NIG_r5.csv.xlsx'),
                   ('R6','NIG_r6.csv.xlsx')]:
    rc = find_religion_col(fname)
    for col, vc in rc.items():
        print(f"  {rnd} {col}: {vc}")

print("\nLoading rounds (this may take 2-3 minutes)...")

r2 = load_round('NIG_r2.csv.xlsx', {
    'police':('q43i','0-3'), 'courts':('q43j','0-3'),
    'relig': ('q43g','0-3'), 'trad':  ('q43e','0-3'),
    'urban': ('urbrur','raw'),
    'relig_q': ('q87','raw'),   # religion variable candidate
})
r3 = load_round('NIG_r3.csv.xlsx', {
    'police':('q52c','1-5->0-3'), 'courts':('q52d','1-5->0-3'),
    'relig': ('q52b','1-5->0-3'),
})
r4 = load_round('NIG_r4.csv.xlsx', {
    'police':     ('q49c','0-3'), 'courts':('q49d','0-3'),
    'relig':      ('q49i','0-3'), 'trad':  ('q49h','0-3'),
    'fear_crime': ('q43', '0-4'),
    'urban':      ('urbrur','raw'),
    'relig_q':    ('q87','raw'),
})
r5 = load_round('NIG_r5.csv.xlsx', {
    'police':     ('q52c','0-3'), 'courts':('q52d','0-3'),
    'fear_crime': ('q43','0-4'),
    'relig_q':    ('q87','raw'),
})
r6 = load_round('NIG_r6.csv.xlsx', {
    'police': ('q52c','0-3'), 'courts':('q52d','0-3'),
    'relig':  ('q52k','0-3'), 'trad':  ('q52j','0-3'),
    'relig_q':('q87','raw'),
})

# Add early-adopter flag
for df in [r2, r3, r4, r5, r6]:
    df['early_sharia'] = df['state_survey'].isin(EARLY_SHARIA).astype(float)
    df['bh_zone']      = df['state_survey'].isin(BH_STATES).astype(float)

print(f"Loaded: R2={len(r2)}, R3={len(r3)}, R4={len(r4)}, R5={len(r5)}, R6={len(r6)}")

# ── Build Muslim indicator ────────────────────────────────────────────────────
# In Afrobarometer the typical coding for Nigeria:
# 1=Christian, 2=Muslim, 3=No religion, etc.
# We'll check by printing distributions and seeing if code 2 = Islam is plausible
# by checking high prevalence in known Muslim-majority Sharia states.
print("\nReligion variable distributions by sharia status (R4, within 150km):")
for col in ['relig_q']:
    if col not in r4.columns: continue
    near = r4[r4['dist_km'].abs() <= BW_MAIN].copy()
    near['relig_v'] = pd.to_numeric(near[col], errors='coerce')
    sharia_side = near[near['sharia']==1]['relig_v'].value_counts().sort_index()
    nonsh_side  = near[near['sharia']==0]['relig_v'].value_counts().sort_index()
    print(f"  Sharia side: {sharia_side.to_dict()}")
    print(f"  Non-Sharia:  {nonsh_side.to_dict()}")

# ════════════════════════════════════════════════════════════════════════════
# 1. NORTHWEST BOUNDARY — R5 and R6 restricted to lon < 10°E
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('TEST 1: NORTHWEST BOUNDARY CHECK (R5 2012, R6 2015)')
print('Restriction: lon < 10°E (away from Boko Haram core: Borno ~13°, Yobe ~12°)')
print('If Boko Haram explains R5/R6 null, northwest effects should survive')
print('='*80)

LON_CUTOFFS = [9.0, 10.0, 11.0]

for rnd, df, outcomes in [
    ('R5', r5, ['police','courts']),
    ('R6', r6, ['police','courts','relig']),
]:
    print(f"\n  {rnd}:")
    print(f"  {'Outcome':10}  {'Full sample':>14}  {'lon<9°':>10}  {'lon<10°':>10}  {'lon<11°':>10}")
    print(f"  {'-'*60}")
    for outcome in outcomes:
        if outcome not in df.columns: continue
        r_full = rd_ols(df, outcome, BW_MAIN)
        fs = f"{r_full['coef']:>+8.3f}{star(r_full['pval']):<2}" if r_full else '     —'
        parts = [fs]
        for lon_cut in LON_CUTOFFS:
            sub = df[df['lon'] < lon_cut].copy()
            r = rd_ols(sub, outcome, BW_MAIN)
            if r:
                parts.append(f"{r['coef']:>+8.3f}{star(r['pval']):<2}")
            else:
                parts.append('     —')
        print(f"  {outcome:<10}  {'  '.join(parts)}")

# Also: show how many obs remain by lon cutoff
print("\n  Sample sizes by lon cutoff:")
for rnd, df in [('R5',r5),('R6',r6)]:
    near = df[df['dist_km'].abs() <= BW_MAIN]
    for lon_cut in LON_CUTOFFS:
        sub = near[near['lon'] < lon_cut]
        print(f"  {rnd} lon<{lon_cut}: N={len(sub)} (sharia={int((sub['sharia']==1).sum())}, "
              f"non-sharia={int((sub['sharia']==0).sum())})")

# ════════════════════════════════════════════════════════════════════════════
# 2. MUSLIM HTE
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('TEST 2: MUSLIM HTE (Sharia × Muslim interaction)')
print('Religious-legitimacy mechanism predicts larger effect for Muslim respondents')
print('='*80)

# Build Muslim indicator: try multiple column candidates
for rnd, df in [('R2',r2),('R4',r4),('R6',r6)]:
    # Find the religion column
    relig_col = None
    for c in ['relig_q','q87','q88','q89']:
        if c in df.columns and df[c].notna().sum() > 200:
            relig_col = c; break
    if relig_col is None:
        print(f"\n  {rnd}: no religion column found"); continue

    near = df[df['dist_km'].abs() <= BW_MAIN].copy()
    near['relig_v'] = pd.to_numeric(near[relig_col], errors='coerce')

    # Determine Muslim code: in Afrobarometer Nigeria, typically 2=Muslim
    # Validate: Sharia side should have higher Muslim %
    for muslim_code in [2, 1]:
        near['muslim'] = (near['relig_v'] == muslim_code).astype(float)
        sharia_muslim  = near[near['sharia']==1]['muslim'].mean()
        nonsh_muslim   = near[near['sharia']==0]['muslim'].mean()
        if sharia_muslim > nonsh_muslim + 0.05:
            break   # found a code where Sharia side is more Muslim

    print(f"\n  {rnd}: religion col={relig_col}, Muslim code={muslim_code}, "
          f"Sharia side Muslim%={sharia_muslim:.2f}, "
          f"non-Sharia Muslim%={nonsh_muslim:.2f}")

    for outcome in ['police','courts','relig']:
        if outcome not in df.columns: continue
        r = rd_hte(df, outcome, 'muslim', BW_MAIN)
        if r:
            print(f"  {outcome:<10}: Sharia effect for non-Muslim={r['eff_0']:>+.3f}{star(r['p_base'])} | "
                  f"Muslim={r['eff_1']:>+.3f} | interaction={r['int']:>+.3f}{star(r['p_int'])} "
                  f"(N={r['n']})")
        else:
            print(f"  {outcome:<10}: insufficient data")

# ════════════════════════════════════════════════════════════════════════════
# 3. NON-MUSLIM SPILLOVER — is there an effect among non-Muslims?
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('TEST 3: NON-MUSLIM SPILLOVER')
print('Security/order channel would raise trust even for non-Muslims near boundary')
print('Note: near the boundary in northern Nigeria, most respondents are Muslim')
print('="Non-Muslim" here = the minority non-Muslim sample near the border')
print('='*80)

for rnd, df in [('R4',r4)]:
    relig_col = None
    for c in ['relig_q','q87','q88','q89']:
        if c in df.columns and df[c].notna().sum() > 200:
            relig_col = c; break
    if relig_col is None:
        print(f"  {rnd}: no religion column"); continue

    near = df[df['dist_km'].abs() <= BW_MAIN].copy()
    near['relig_v'] = pd.to_numeric(near[relig_col], errors='coerce')
    for muslim_code in [2, 1]:
        near['muslim'] = (near['relig_v'] == muslim_code).astype(float)
        if near[near['sharia']==1]['muslim'].mean() > near[near['sharia']==0]['muslim'].mean() + 0.05:
            break

    non_muslim = near[near['muslim']==0].copy()
    muslim_s   = near[near['muslim']==1].copy()

    print(f"\n  {rnd} Muslim sample (N={len(muslim_s)}):")
    for outcome in ['police','courts','relig']:
        if outcome not in muslim_s.columns: continue
        r = rd_ols(muslim_s, outcome, BW_MAIN)
        if r: print(f"    {outcome}: β={r['coef']:>+.3f}{star(r['pval'])} SE={r['se']:.3f} N={r['n']}")

    print(f"\n  {rnd} Non-Muslim sample (N={len(non_muslim)}):")
    for outcome in ['police','courts','relig']:
        if outcome not in non_muslim.columns: continue
        r = rd_ols(non_muslim, outcome, BW_MAIN)
        if r: print(f"    {outcome}: β={r['coef']:>+.3f}{star(r['pval'])} SE={r['se']:.3f} N={r['n']}")
        else: print(f"    {outcome}: insufficient data (N={len(non_muslim)})")

# ════════════════════════════════════════════════════════════════════════════
# 4. FEAR OF CRIME (R4) — security mechanism
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('TEST 4: FEAR OF CRIME (R4, 0=never afraid, 4=always afraid)')
print('Sharia hudud penalties → lower crime → lower fear (mechanism test)')
print('Prediction: β < 0 (Sharia side feels safer)')
print('='*80)

r = rd_ols(r4, 'fear_crime', BW_MAIN)
if r:
    print(f"\n  R4 fear of crime: β={r['coef']:>+.3f}{star(r['pval'])} "
          f"SE={r['se']:.3f} p={r['pval']:.3f} N={r['n']}")
    print(f"  Interpretation: {'NEGATIVE (lower fear on Sharia side) — consistent with security mechanism' if r['coef'] < 0 else 'POSITIVE (higher fear) — inconsistent with security mechanism; if null, mechanism inconclusive'}")

# Also at R5 (post-insurgency)
r_r5 = rd_ols(r5, 'fear_crime', BW_MAIN)
if r_r5:
    print(f"\n  R5 fear of crime (post-Boko Haram): β={r_r5['coef']:>+.3f}{star(r_r5['pval'])} "
          f"SE={r_r5['se']:.3f} p={r_r5['pval']:.3f} N={r_r5['n']}")

# ════════════════════════════════════════════════════════════════════════════
# 5. URBAN / RURAL HTE (R4)
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('TEST 5: URBAN/RURAL HTE (R4)')
print('urban=1 (urban), urban=2 (rural) in Afrobarometer urbrur')
print('If Sharia courts operate mainly in cities, urban effect > rural')
print('='*80)

r4['is_urban'] = (r4['urban'] == 1).astype(float)  # 1=urban, 2=rural → binary

for outcome in ['police','courts','relig']:
    if outcome not in r4.columns: continue
    r = rd_hte(r4, outcome, 'is_urban', BW_MAIN)
    if r:
        print(f"  {outcome}: rural β={r['eff_0']:>+.3f}{star(r['p_base'])} | "
              f"urban β={r['eff_1']:>+.3f} | interaction={r['int']:>+.3f}{star(r['p_int'])}")

# ════════════════════════════════════════════════════════════════════════════
# 6. EARLY vs. LATE SHARIA ADOPTER HTE
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('TEST 6: EARLY vs. LATE SHARIA ADOPTER HTE')
print('Early (2000): Zamfara, Sokoto, Niger, Kano, Katsina, Kebbi')
print('Late (2001-2002): Jigawa, Gombe, Bauchi, Yobe, Borno, Kaduna')
print('If exposure duration matters, early-adopter states should show larger R2 effect')
print('="early_sharia" = Sharia side AND in early adopter state')
print('='*80)

for rnd, df in [('R2',r2),('R4',r4)]:
    print(f"\n  {rnd}:")
    print(f"  {'Outcome':10}  {'Early adopter β':>18}  {'Late adopter β':>18}  {'Interaction p':>14}")
    print(f"  {'-'*70}")
    for outcome in ['police','courts','relig']:
        if outcome not in df.columns: continue

        # Restrict to Sharia side, compare early vs. late adopters
        d_sharia = df[(df['dist_km'].abs() <= BW_MAIN) &
                      (df['sharia'] == 1)].copy().dropna(subset=[outcome,'dist_km'])

        # Also: HTE on full sample, early_sharia = 1 for early states on sharia side
        d_full = df[df['dist_km'].abs() <= BW_MAIN].copy().dropna(subset=[outcome,'dist_km'])
        d_full['is_early'] = ((d_full['sharia']==1) &
                              d_full['state_survey'].isin(EARLY_SHARIA)).astype(float)
        d_full['is_late']  = ((d_full['sharia']==1) &
                              ~d_full['state_survey'].isin(EARLY_SHARIA)).astype(float)

        # OLS with early/late indicators
        n_cl = d_full['ea_id'].nunique()
        cov = ('cluster', {'groups': d_full['ea_id']}) if n_cl >= 5 else ('HC3', {})
        try:
            m = smf.ols(
                'y ~ is_early + is_late + x + x:is_early + x:is_late',
                data=d_full.assign(y=d_full[outcome], x=d_full['dist_km'])
            ).fit(cov_type=cov[0], cov_kwds=cov[1])
            c_e = m.params.get('is_early', np.nan)
            c_l = m.params.get('is_late',  np.nan)
            p_e = m.pvalues.get('is_early', np.nan)
            p_l = m.pvalues.get('is_late',  np.nan)
            # F-test: is_early != is_late
            from scipy import stats
            f_pval = np.nan
            try:
                f_stat = m.f_test(['is_early - is_late = 0'])
                f_pval = float(f_stat.pvalue)
            except Exception: pass
            print(f"  {outcome:<10}  {c_e:>+8.3f}{star(p_e):<9}  "
                  f"{c_l:>+8.3f}{star(p_l):<9}  {f_pval:>13.3f}" if not np.isnan(f_pval)
                  else f"  {outcome:<10}  {c_e:>+8.3f}{star(p_e):<9}  {c_l:>+8.3f}{star(p_l):<9}  —")
        except Exception as e:
            print(f"  {outcome:<10}: error {e}")

# ════════════════════════════════════════════════════════════════════════════
# 7. RELIGIOUS VS. TRADITIONAL LEADERS — channel specificity
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('TEST 7: CHANNEL SPECIFICITY — religious vs. traditional leader trust')
print('Religious-legitimacy mechanism predicts: relig effect > trad effect')
print('If relig ≈ trad, effect reflects a general "north vs. middle belt" sentiment')
print('='*80)

for rnd, df in [('R2',r2),('R4',r4),('R6',r6)]:
    rr = rd_ols(df, 'relig', BW_MAIN) if 'relig' in df.columns else None
    rt = rd_ols(df, 'trad',  BW_MAIN) if 'trad'  in df.columns else None
    rs = f"relig={rr['coef']:>+.3f}{star(rr['pval'])}" if rr else "relig=—"
    ts = f"trad={rt['coef']:>+.3f}{star(rt['pval'])}"  if rt else "trad=—"
    if rr and rt:
        diff = rr['coef'] - rt['coef']
        print(f"  {rnd}: {rs}  |  {ts}  |  relig-trad difference={diff:>+.3f}")
        # Test difference: add both to model
        d = df[df['dist_km'].abs() <= BW_MAIN].copy()
        d = d.dropna(subset=['relig','trad','dist_km'])
        if len(d) > 50:
            n_cl = d['ea_id'].nunique()
            cov = ('cluster', {'groups': d['ea_id']}) if n_cl >= 5 else ('HC3', {})
            try:
                # Stack outcomes (long form)
                d1 = d.copy(); d1['y'] = d1['relig']; d1['is_relig'] = 1
                d2 = d.copy(); d2['y'] = d2['trad'];  d2['is_relig'] = 0
                dl = pd.concat([d1,d2], ignore_index=True)
                ml = smf.ols('y ~ sharia + is_relig + sharia:is_relig + x + x:sharia',
                             data=dl.assign(x=dl['dist_km'])
                             ).fit(cov_type='HC3')  # can't cluster stacked easily
                c_diff = ml.params.get('sharia:is_relig', np.nan)
                p_diff = ml.pvalues.get('sharia:is_relig', np.nan)
                print(f"       Relig - Trad interaction: {c_diff:>+.3f}{star(p_diff)} (p={p_diff:.3f})")
            except Exception: pass
    else:
        avail = [s for s in [rs, ts] if '—' not in s]
        print(f"  {rnd}: {', '.join(avail)}")

print('\nMechanism tests complete.')
