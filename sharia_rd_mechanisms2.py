"""
Mechanism tests — revised. Fixes Muslim HTE column bug, adds smarter religion detection.
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
EARLY_SHARIA  = {'Zamfara','Sokoto','Niger','Kano','Katsina','Kebbi'}

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
        c, s, p = (m.params.get(k, np.nan) for k in ['sharia','sharia','sharia'])
        c = m.params.get('sharia', np.nan)
        s = m.bse.get('sharia', np.nan)
        p = m.pvalues.get('sharia', np.nan)
        if abs(c) > 3 or s > 2: return None
        return {'coef': c, 'se': s, 'pval': p, 'n': len(d), 'n_cl': n_cl}
    except Exception: return None

def rd_hte(df, outcome, moderator, bw=150):
    """HTE: returns effect at mod=0 and mod=1, plus interaction."""
    d = df.dropna(subset=[outcome,'dist_km',moderator]).copy()
    d = d[d['dist_km'].abs() <= bw]
    if len(d) < 80 or d[outcome].nunique() < 2: return None
    if d['sharia'].nunique() < 2 or d[moderator].nunique() < 2: return None
    n_cl = d['ea_id'].nunique() if 'ea_id' in d.columns else 0
    cov = ('cluster', {'groups': d['ea_id']}) if n_cl >= 5 else ('HC3', {})
    try:
        m = smf.ols('y ~ sharia + sharia:mod + x + x:sharia + mod',
                    data=d.assign(y=d[outcome], x=d['dist_km'], mod=d[moderator])
                    ).fit(cov_type=cov[0], cov_kwds=cov[1])
        c_base = m.params.get('sharia', np.nan)
        c_int  = m.params.get('sharia:mod', np.nan)
        p_base = m.pvalues.get('sharia', np.nan)
        p_int  = m.pvalues.get('sharia:mod', np.nan)
        return {'base': c_base, 'int': c_int, 'p_base': p_base, 'p_int': p_int,
                'eff_0': c_base, 'eff_1': c_base + c_int, 'n': len(d)}
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
        elif recode == 'raw':      pass
        df[name] = v
    return df

def find_muslim_indicator(df, bw=BW_MAIN):
    """
    Identify the best religion proxy column by finding which (column, value) pair
    has the largest Sharia-side enrichment within the boundary sample.
    Returns (column_name, muslim_code, sharia_pct, nonsharia_pct) or None.
    """
    near = df[df['dist_km'].abs() <= bw].copy()
    sharia_mask = near['sharia'] == 1
    candidates = []
    # Try all columns with values in 1-15 range (likely coded religion)
    for col in df.columns:
        if col in ('sharia','dist_km','ea_id','lat','lon','state_survey'): continue
        v = pd.to_numeric(near[col], errors='coerce')
        if v.isna().mean() > 0.6: continue
        vals = v.dropna().unique()
        if len(vals) < 2 or len(vals) > 15: continue
        if v.min() < 0 or v.max() > 20: continue
        for code in vals:
            if pd.isna(code): continue
            ind = (v == code).astype(float)
            sp = ind[sharia_mask].mean()
            np_ = ind[~sharia_mask].mean()
            if np_ > 0 and sp > np_ + 0.10:  # at least 10pp Sharia enrichment
                candidates.append((col, code, sp, np_, sp - np_))
    if not candidates: return None
    # Pick the one with largest enrichment
    best = max(candidates, key=lambda x: x[4])
    return best  # (col, code, sharia_pct, nonsharia_pct, diff)

print("Loading rounds...")
r2 = load_round('NIG_r2.csv.xlsx', {
    'police':('q43i','0-3'), 'courts':('q43j','0-3'),
    'relig': ('q43g','0-3'), 'trad':  ('q43e','0-3'),
    'urban': ('urbrur','raw'),
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
})
r5 = load_round('NIG_r5.csv.xlsx', {
    'police':     ('q52c','0-3'), 'courts':('q52d','0-3'),
    'fear_crime': ('q43','0-4'),
})
r6 = load_round('NIG_r6.csv.xlsx', {
    'police': ('q52c','0-3'), 'courts':('q52d','0-3'),
    'relig':  ('q52k','0-3'), 'trad':  ('q52j','0-3'),
})

for df in [r2, r3, r4, r5, r6]:
    df['early_sharia'] = df['state_survey'].isin(EARLY_SHARIA).astype(float)

print(f"Loaded: R2={len(r2)}, R3={len(r3)}, R4={len(r4)}, R5={len(r5)}, R6={len(r6)}")

# ════════════════════════════════════════════════════════════════════════════
# Find Muslim indicator columns
# ════════════════════════════════════════════════════════════════════════════
print("\nFinding Muslim proxy column (largest Sharia-side enrichment)...")
muslim_cols = {}
for rnd, df in [('R2',r2),('R3',r3),('R4',r4)]:
    result = find_muslim_indicator(df)
    if result:
        col, code, sp, np_, diff = result
        print(f"  {rnd}: col={col}, code={code:.0f}, "
              f"Sharia={sp:.2%}, non-Sharia={np_:.2%}, diff={diff:.2%}")
        muslim_cols[rnd] = (col, code)
    else:
        print(f"  {rnd}: no Muslim proxy found")

# ════════════════════════════════════════════════════════════════════════════
# 1. NORTHWEST BOUNDARY — R5 and R6 restricted to lon < 10°E
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('TEST 1: NORTHWEST BOUNDARY CHECK — R5 (2012) and R6 (2015)')
print('lon < 10°E excludes Borno (~13°E) and Yobe (~12°E) — Boko Haram core')
print('If BH explains R5/R6 null, northwest-boundary effects should survive')
print('='*80)

for rnd, df, outcomes in [('R5',r5,['police','courts']),
                           ('R6',r6,['police','courts','relig'])]:
    print(f"\n  {rnd}:")
    hdr = f"  {'Outcome':10}  {'Full':>12}  {'lon<9°':>10}  {'lon<10°':>10}  {'lon<11°':>10}"
    print(hdr); print(f"  {'-'*60}")
    for outcome in outcomes:
        if outcome not in df.columns: continue
        r_full = rd_ols(df, outcome, BW_MAIN)
        fs = f"{r_full['coef']:>+8.3f}{star(r_full['pval']):<2}" if r_full else '     —'
        parts = [fs]
        for lon_cut in [9.0, 10.0, 11.0]:
            sub = df[df['lon'] < lon_cut].copy()
            r = rd_ols(sub, outcome, BW_MAIN)
            parts.append(f"{r['coef']:>+8.3f}{star(r['pval']):<2}" if r else '     —')
        print(f"  {outcome:<10}  {'  '.join(parts)}")

print("\n  Sample sizes by lon cutoff:")
for rnd, df in [('R5',r5),('R6',r6)]:
    near = df[df['dist_km'].abs() <= BW_MAIN]
    for lon_cut in [9.0, 10.0, 11.0]:
        sub = near[near['lon'] < lon_cut]
        s = int((sub['sharia']==1).sum()); ns = int((sub['sharia']==0).sum())
        print(f"  {rnd} lon<{lon_cut:.0f}: N={len(sub)} (sharia={s}, non-sharia={ns})")

# ════════════════════════════════════════════════════════════════════════════
# 2. MUSLIM HTE
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('TEST 2: MUSLIM HTE (Sharia × Muslim interaction)')
print('Religious-legitimacy mechanism: effect should be larger for Muslims')
print('="non-Muslim effect" tests whether spillover to other-religion respondents exists')
print('='*80)

for rnd, df in [('R2',r2),('R4',r4)]:
    if rnd not in muslim_cols:
        print(f"\n  {rnd}: skipped (no Muslim proxy)"); continue
    col, code = muslim_cols[rnd]
    df['muslim'] = (pd.to_numeric(df[col], errors='coerce') == code).astype(float)
    near = df[df['dist_km'].abs() <= BW_MAIN]
    n_mus = int((near['muslim']==1).sum())
    n_non = int((near['muslim']==0).sum())
    print(f"\n  {rnd} ({col}, code={code:.0f}): Muslim N={n_mus}, non-Muslim N={n_non}")
    print(f"  {'Outcome':10}  {'non-Muslim β':>14}  {'Muslim β':>12}  {'interaction β':>15}  {'int p':>8}")
    print(f"  {'-'*65}")
    for outcome in ['police','courts','relig']:
        if outcome not in df.columns: continue
        r = rd_hte(df, outcome, 'muslim', BW_MAIN)
        if r:
            print(f"  {outcome:<10}  {r['eff_0']:>+8.3f}{star(r['p_base']):<5}  "
                  f"{r['eff_1']:>+8.3f}       {r['int']:>+8.3f}{star(r['p_int']):<6}  "
                  f"{r['p_int']:>8.3f}")
        else:
            print(f"  {outcome:<10}  insufficient data")

# ════════════════════════════════════════════════════════════════════════════
# 3. NON-MUSLIM SUBSAMPLE TEST (R4 — largest overall effects)
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('TEST 3: WITHIN-RELIGION SPLIT (R4)')
print('Run RD separately on Muslim and non-Muslim subsamples')
print('='*80)

if 'R4' in muslim_cols:
    col, code = muslim_cols['R4']
    r4['muslim'] = (pd.to_numeric(r4[col], errors='coerce') == code).astype(float)
    mus  = r4[r4['muslim']==1].copy()
    nmus = r4[r4['muslim']==0].copy()
    print(f"\n  R4 Muslim subsample (col={col}, code={code:.0f}):")
    for outcome in ['police','courts','relig']:
        if outcome not in mus.columns: continue
        rm = rd_ols(mus,  outcome, BW_MAIN)
        rn = rd_ols(nmus, outcome, BW_MAIN)
        ms = f"β={rm['coef']:>+.3f}{star(rm['pval'])}" if rm else "β=—"
        ns = f"β={rn['coef']:>+.3f}{star(rn['pval'])}" if rn else "β=—"
        print(f"  {outcome}: Muslim: {ms} (N={rm['n'] if rm else '—'})  |  "
              f"Non-Muslim: {ns} (N={rn['n'] if rn else '—'})")
else:
    print("  R4 Muslim proxy not available; skipping split")

# ════════════════════════════════════════════════════════════════════════════
# 4. FEAR OF CRIME (R4 and R5) — security mechanism
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('TEST 4: FEAR OF CRIME — security / deterrence mechanism')
print('Sharia hudud penalties → lower crime → lower fear (β < 0 expected)')
print('R4 (2008) = pre-insurgency; R5 (2012) = post-Boko Haram')
print('='*80)

for rnd, df in [('R4',r4),('R5',r5)]:
    r = rd_ols(df, 'fear_crime', BW_MAIN)
    if r:
        sign = ('LOWER fear on Sharia side ✓ security mechanism' if r['coef'] < 0
                else 'HIGHER or same fear — no security mechanism evidence')
        print(f"\n  {rnd} fear of crime: β={r['coef']:>+.3f}{star(r['pval'])} "
              f"SE={r['se']:.3f} p={r['pval']:.3f} N={r['n']}")
        print(f"  → {sign}")

# ════════════════════════════════════════════════════════════════════════════
# 5. URBAN/RURAL HTE (R4)
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('TEST 5: URBAN/RURAL HTE (R4)')
print('urbrur: 1=urban, 2=rural → is_urban = 1 if urban')
print('Sharia courts mainly in cities → predict urban effect > rural')
print('='*80)

r4['is_urban'] = (r4['urban'] == 1).astype(float)
near = r4[r4['dist_km'].abs() <= BW_MAIN]
print(f"  Urban N={int((near['is_urban']==1).sum())}, Rural N={int((near['is_urban']==0).sum())}")
print(f"  {'Outcome':10}  {'Rural β':>12}  {'Urban β':>12}  {'interaction β':>15}  {'int p':>8}")
print(f"  {'-'*65}")
for outcome in ['police','courts','relig']:
    if outcome not in r4.columns: continue
    r = rd_hte(r4, outcome, 'is_urban', BW_MAIN)
    if r:
        print(f"  {outcome:<10}  {r['eff_0']:>+8.3f}{star(r['p_base']):<3}  "
              f"{r['eff_1']:>+8.3f}       {r['int']:>+8.3f}{star(r['p_int']):<6}  "
              f"{r['p_int']:>8.3f}")

# ════════════════════════════════════════════════════════════════════════════
# 6. EARLY vs. LATE ADOPTER HTE
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('TEST 6: EARLY vs. LATE SHARIA ADOPTER HTE')
print('Early (2000): Zamfara, Sokoto, Niger, Kano, Katsina, Kebbi')
print('Late (2001-2002): Jigawa, Gombe, Bauchi, Yobe, Borno, Kaduna')
print('By R2 (2003): early adopters had ~3 yrs Sharia vs. 1-2 for late adopters')
print('="early_sharia" = 1 for respondents in early-adopter Sharia states')
print('This compares two sets of Sharia-side respondents (non-Sharia = reference)')
print('='*80)

for rnd, df in [('R2',r2),('R4',r4)]:
    print(f"\n  {rnd}:")
    # Early = 1 for Sharia-side respondents in early adopter states
    d = df[df['dist_km'].abs() <= BW_MAIN].copy()
    d['early'] = ((d['sharia']==1) & d['state_survey'].isin(EARLY_SHARIA)).astype(float)
    d['late']  = ((d['sharia']==1) & ~d['state_survey'].isin(EARLY_SHARIA)).astype(float)
    early_n = int(d['early'].sum())
    late_n  = int(d['late'].sum())
    print(f"  Early-adopter Sharia N={early_n}, Late-adopter Sharia N={late_n}, "
          f"Non-Sharia N={int((d['sharia']==0).sum())}")
    print(f"  {'Outcome':10}  {'Early β':>12}  {'Late β':>12}  {'F-test p':>10}")
    print(f"  {'-'*50}")
    for outcome in ['police','courts','relig']:
        if outcome not in d.columns: continue
        dd = d.dropna(subset=[outcome,'dist_km'])
        if len(dd) < 80: continue
        n_cl = dd['ea_id'].nunique()
        cov = ('cluster', {'groups': dd['ea_id']}) if n_cl >= 5 else ('HC3', {})
        try:
            m = smf.ols('y ~ early + late + x + x:early + x:late',
                        data=dd.assign(y=dd[outcome], x=dd['dist_km'])
                        ).fit(cov_type=cov[0], cov_kwds=cov[1])
            c_e = m.params.get('early', np.nan)
            c_l = m.params.get('late',  np.nan)
            p_e = m.pvalues.get('early', np.nan)
            p_l = m.pvalues.get('late',  np.nan)
            try:
                fp = float(m.f_test(['early - late = 0']).pvalue)
            except Exception: fp = np.nan
            fp_s = f"{fp:.3f}" if not np.isnan(fp) else "—"
            print(f"  {outcome:<10}  {c_e:>+8.3f}{star(p_e):<3}  "
                  f"{c_l:>+8.3f}{star(p_l):<3}  {fp_s:>10}")
        except Exception as e:
            print(f"  {outcome:<10}  error: {e}")

# ════════════════════════════════════════════════════════════════════════════
# 7. RELIGIOUS vs. TRADITIONAL LEADERS — channel specificity
# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('TEST 7: CHANNEL SPECIFICITY — religious vs. traditional leader trust')
print('Religious-legitimacy mechanism: relig effect > trad effect')
print('If relig ≈ trad, effect is a general north/Middle-Belt sentiment')
print('='*80)

for rnd, df, outcomes in [('R2',r2,('relig','trad')),
                           ('R4',r4,('relig','trad')),
                           ('R6',r6,('relig','trad'))]:
    rr = rd_ols(df, outcomes[0], BW_MAIN) if outcomes[0] in df.columns else None
    rt = rd_ols(df, outcomes[1], BW_MAIN) if outcomes[1] in df.columns else None
    if rr and rt:
        diff = rr['coef'] - rt['coef']
        print(f"\n  {rnd}: relig={rr['coef']:>+.3f}{star(rr['pval'])}  "
              f"trad={rt['coef']:>+.3f}{star(rt['pval'])}  "
              f"relig-trad={diff:>+.3f}")
        # Stack and test in one model
        d = df[df['dist_km'].abs() <= BW_MAIN].copy().dropna(subset=[outcomes[0],outcomes[1],'dist_km'])
        if len(d) > 50:
            d1 = d.copy(); d1['y'] = d1[outcomes[0]]; d1['is_relig'] = 1
            d2 = d.copy(); d2['y'] = d2[outcomes[1]]; d2['is_relig'] = 0
            dl = pd.concat([d1,d2], ignore_index=True)
            try:
                ml = smf.ols('y ~ sharia + is_relig + sharia:is_relig + x + x:sharia',
                             data=dl.assign(x=dl['dist_km'])
                             ).fit(cov_type='HC3')
                c_diff = ml.params.get('sharia:is_relig', np.nan)
                p_diff = ml.pvalues.get('sharia:is_relig', np.nan)
                print(f"  Stacked test: relig−trad sharia interaction "
                      f"{c_diff:>+.3f}{star(p_diff)} (p={p_diff:.3f})")
            except Exception: pass
    else:
        relig_str = ('relig='+f'{rr["coef"]:>+.3f}{star(rr["pval"])}') if rr else 'relig=—'
        trad_str  = ('trad=' +f'{rt["coef"]:>+.3f}{star(rt["pval"])}') if rt else 'trad=—'
        print(f"\n  {rnd}: {relig_str}  {trad_str}")

print('\nMechanism tests complete.')
