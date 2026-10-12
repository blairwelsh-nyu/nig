"""
Three reviewer tests:

A. Pooled rdrobust R2-R4 with masspoints='adjust', cluster=EA, round FE covariates.
   If significant, rdrobust replaces OLS as primary estimator (Issue 1 resolved).

B. Boundary-segment fixed effects. Divide the official Sharia/non-Sharia boundary
   by longitude of nearest boundary point. Compare northwestern segments (away from
   the North/Middle-Belt confound) vs Middle Belt segments (Kaduna/Plateau,
   Bauchi/Plateau). If effect appears only in Middle Belt, it is regional gradient.

C. Non-Muslim subsample. If Christians just inside the Sharia boundary gain trust in
   religious leaders and courts at the same rate as Muslims, the Sharia mechanism is
   hard to sustain. Show both subsamples and the interaction explicitly.
"""
import os, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union, nearest_points
import statsmodels.formula.api as smf
from rdrobust import rdrobust
warnings.filterwarnings('ignore')

os.chdir('/home/user/nig')

# ── constants ─────────────────────────────────────────────────────────────────
SHARIA_STATES = {'Zamfara','Kano','Sokoto','Katsina','Bauchi','Niger',
                 'Jigawa','Kebbi','Yobe','Borno','Kaduna','Gombe'}
BW_MAIN = 150

# ── boundary geometry ─────────────────────────────────────────────────────────
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

def nearest_boundary_lon(lat, lon):
    """Longitude of the nearest point on the official boundary (for segment assignment)."""
    p = Point(lon, lat)
    pt_on_bdy, _ = nearest_points(official_boundary, p)
    return pt_on_bdy.x

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

# ── data loader ───────────────────────────────────────────────────────────────
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
        if recode == '0-3':        v = v.where(v <= 3)
        elif recode == '1-5->0-3': v = (v.where(v <= 5) - 1) * (3/4)
        elif recode == '0-4':      v = v.where(v <= 4)
        df[name] = v
    return df

def find_muslim_col(df, bw=BW_MAIN):
    near = df[df['dist_km'].abs() <= bw].copy()
    sm = near['sharia'] == 1
    best = None
    for col in df.columns:
        if col in ('sharia','dist_km','ea_id','lat','lon','state_survey'): continue
        v = pd.to_numeric(near[col], errors='coerce')
        if v.isna().mean() > 0.6: continue
        vals = v.dropna().unique()
        if len(vals) < 2 or len(vals) > 15: continue
        if v.min() < 0 or v.max() > 20: continue
        for code in vals:
            if pd.isna(code): continue
            sp  = (v == code)[sm].mean()
            nsp = (v == code)[~sm].mean()
            diff = sp - nsp
            if nsp > 0 and diff > 0.10:
                if best is None or diff > best[4]:
                    best = (col, code, sp, nsp, diff)
    return best   # (col, code, sharia_pct, nonsharia_pct, diff) or None

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading rounds...")
r2 = load_round('NIG_r2.csv.xlsx', {
    'police':('q43i','0-3'), 'courts':('q43j','0-3'), 'relig':('q43g','0-3'),
})
r3 = load_round('NIG_r3.csv.xlsx', {
    'police':('q52c','1-5->0-3'), 'courts':('q52d','1-5->0-3'),
    'relig': ('q52b','1-5->0-3'),
})
r4 = load_round('NIG_r4.csv.xlsx', {
    'police':('q49c','0-3'), 'courts':('q49d','0-3'), 'relig':('q49i','0-3'),
})
print(f"Loaded: R2={len(r2)}, R3={len(r3)}, R4={len(r4)}")

# ════════════════════════════════════════════════════════════════════════════════
# TEST A: POOLED rdrobust R2-R4
# masspoints='adjust', cluster=EA×round, round FE via covs argument
# ════════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('TEST A: POOLED rdrobust (R2–R4), masspoints=adjust, cluster=EA, round FE')
print('CCT bandwidth selected on pooled data. EA×round cluster (different respondents')
print('same EA across rounds treated as independent — less conservative than pooled EA).')
print('='*80)

OUTCOMES_COMMON = ['police','courts','relig']

for outcome in OUTCOMES_COMMON:
    dfs = []
    for rnd_label, df_r in [('R2',r2),('R3',r3),('R4',r4)]:
        d = df_r[['dist_km','sharia','ea_id',outcome]].dropna().copy()
        d = d[d['dist_km'].abs() <= BW_MAIN]
        if len(d) < 30 or d[outcome].nunique() < 2: continue
        mu, sd = d[outcome].mean(), d[outcome].std()
        if sd == 0: continue
        d['y_std'] = (d[outcome] - mu) / sd
        d['R3'] = float(rnd_label == 'R3')
        d['R4'] = float(rnd_label == 'R4')
        d['ea_round'] = d['ea_id'].astype(str) + '_' + rnd_label
        dfs.append(d[['y_std','dist_km','R3','R4','ea_round']])

    if not dfs:
        print(f"  {outcome}: no data"); continue
    pooled = pd.concat(dfs, ignore_index=True).dropna()
    covs = pooled[['R3','R4']].values

    try:
        r = rdrobust(pooled['y_std'].values,
                     pooled['dist_km'].values,
                     covs=covs,
                     masspoints='adjust',
                     cluster=pooled['ea_round'].values,
                     bwselect='mserd')
        try:
            coef = float(r.coef.loc['Robust','Coeff'])
            se   = float(r.se.loc['Robust','Std. Err.'])
            pval = float(r.pv.loc['Robust','P>|z|'])
        except Exception:
            coef = float(r.coef.iloc[2,0])
            se   = float(r.se.iloc[2,0])
            pval = float(r.pv.iloc[2,0])
        try:
            bw = float(r.bws.loc['h','left'])
        except Exception:
            bw = float(r.bws.iloc[0,0])
        N_h = r.N_h if isinstance(r.N_h, (list, tuple)) else [r.N_h]
        n_l = int(N_h[0]) if len(N_h) > 0 else 0
        n_r = int(N_h[1]) if len(N_h) > 1 else 0
        print(f"\n  {outcome}: β={coef:+.3f} SE={se:.3f} p={pval:.3f}{star(pval)}"
              f"  BW={bw:.0f}km  N(left={n_l}, right={n_r})")
    except Exception as e:
        print(f"\n  {outcome}: FAILED — {e}")

# ════════════════════════════════════════════════════════════════════════════════
# TEST B: BOUNDARY SEGMENT FIXED EFFECTS
# Assign each respondent to a boundary segment by nearest-boundary-point longitude.
# Northwestern segment (seg_lon < 7.5°E): Niger State / Kwara–Kogi–FCT
#   — Far from the North/Middle-Belt cultural divide; mainly Muslim–Muslim interface.
# Middle Belt segment (7.5–10.5°E): Kaduna / FCT–Plateau, Bauchi / Plateau–Nasarawa
#   — Where the North/Middle-Belt confound is strongest.
# Eastern segment (>10.5°E): Gombe–Borno / Adamawa–Taraba (Boko Haram zone)
# ════════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('TEST B: BOUNDARY SEGMENT ANALYSIS (R4, 2008 — sharpest pre-insurgency round)')
print('If effect appears only in Middle Belt but not Northwest, it is a regional')
print('gradient. If Northwest also shows positive effects, Sharia mechanism is')
print('more credible independent of the cultural divide.')
print('='*80)

def assign_segment(seg_lon):
    if seg_lon < 7.5:  return 'NW'    # Northwestern: Niger/Kwara–FCT
    if seg_lon < 10.5: return 'MB'    # Middle Belt:  Kaduna–Bauchi/Plateau
    return 'E'                        # Eastern:      Gombe–Borno/Adamawa

# Compute boundary-point longitude for R4 respondents within BW
print("\n  Computing nearest-boundary-point longitude for R4...")
r4_bw = r4[r4['dist_km'].abs() <= BW_MAIN].copy()
r4_bw['seg_lon'] = [nearest_boundary_lon(row['lat'], row['lon'])
                    for _, row in r4_bw.iterrows()]
r4_bw['segment'] = r4_bw['seg_lon'].apply(assign_segment)

seg_counts = r4_bw.groupby(['segment','sharia']).size().unstack(fill_value=0)
print(f"\n  Segment sample sizes (R4, within {BW_MAIN}km):")
for seg, row in seg_counts.iterrows():
    nS = row.get(0.0, row.get(0, 0))
    S  = row.get(1.0, row.get(1, 0))
    label = {'NW':'Northwestern (lon<7.5°)', 'MB':'Middle Belt (7.5–10.5°)',
             'E': 'Eastern (lon>10.5°)'}[seg]
    print(f"    {label}: Sharia={int(S)}, non-Sharia={int(nS)}")

# ── B1: segment-by-segment OLS ────────────────────────────────────────────────
print()
print("  B1. Segment-by-segment OLS (police, courts, relig):")
print(f"  {'Outcome':<10} {'NW β':<12} {'MB β':<12} {'East β':<12}")
print("  " + "-"*45)

for outcome in ['police','courts','relig']:
    row_vals = {}
    for seg in ['NW','MB','E']:
        d = r4_bw[r4_bw['segment'] == seg].dropna(subset=[outcome]).copy()
        if len(d) < 20 or d[outcome].nunique() < 2 or d['sharia'].nunique() < 2:
            row_vals[seg] = '  n/a'
            continue
        n_cl = d['ea_id'].nunique()
        cov = ('cluster',{'groups':d['ea_id']}) if n_cl >= 5 else ('HC3',{})
        try:
            m = smf.ols('y ~ sharia + x + x:sharia',
                        data=d.assign(y=d[outcome], x=d['dist_km'])
                        ).fit(cov_type=cov[0], cov_kwds=cov[1])
            c = m.params.get('sharia', np.nan)
            p = m.pvalues.get('sharia', np.nan)
            row_vals[seg] = f'{c:+.3f}{star(p)}'
        except Exception:
            row_vals[seg] = ' err'
    print(f"  {outcome:<10} {row_vals.get('NW','n/a'):<12} "
          f"{row_vals.get('MB','n/a'):<12} {row_vals.get('E','n/a'):<12}")

# ── B2: segment FE model (pool R4 with segment dummies) ──────────────────────
print()
print("  B2. Pooled R4 with segment FEs (segment dummies in regression):")
print("      Controls for segment-level baseline differences in trust.")
print(f"  {'Outcome':<10} {'β (Sharia)':<14} {'SE':<10} {'p'}")
print("  " + "-"*45)

for outcome in ['police','courts','relig']:
    d = r4_bw.dropna(subset=[outcome,'segment']).copy()
    if len(d) < 50 or d[outcome].nunique() < 2: continue
    # segment dummies (NW reference)
    d['seg_MB'] = (d['segment'] == 'MB').astype(float)
    d['seg_E']  = (d['segment'] == 'E').astype(float)
    n_cl = d['ea_id'].nunique()
    cov = ('cluster',{'groups':d['ea_id']}) if n_cl >= 5 else ('HC3',{})
    try:
        m = smf.ols('y ~ sharia + x + x:sharia + seg_MB + seg_E',
                    data=d.assign(y=d[outcome], x=d['dist_km'])
                    ).fit(cov_type=cov[0], cov_kwds=cov[1])
        c = m.params.get('sharia', np.nan)
        s = m.bse.get('sharia', np.nan)
        p = m.pvalues.get('sharia', np.nan)
        print(f"  {outcome:<10} {c:+.3f}{star(p):<10}  {s:.3f}     p={p:.3f}")
    except Exception as e:
        print(f"  {outcome}: FAILED — {e}")

# Also run B1 and B2 for R2 (earlier round, cleaner pre-Sharia)
print()
print("  B3. Segment-by-segment OLS R2 (2003 — first post-Sharia round):")
print(f"  {'Outcome':<10} {'NW β':<12} {'MB β':<12} {'East β':<12}")
print("  " + "-"*45)

r2_bw = r2[r2['dist_km'].abs() <= BW_MAIN].copy()
r2_bw['seg_lon'] = [nearest_boundary_lon(row['lat'], row['lon'])
                    for _, row in r2_bw.iterrows()]
r2_bw['segment'] = r2_bw['seg_lon'].apply(assign_segment)

for outcome in ['police','courts','relig']:
    row_vals = {}
    for seg in ['NW','MB','E']:
        d = r2_bw[r2_bw['segment'] == seg].dropna(subset=[outcome]).copy()
        if len(d) < 20 or d[outcome].nunique() < 2 or d['sharia'].nunique() < 2:
            row_vals[seg] = '  n/a'
            continue
        n_cl = d['ea_id'].nunique()
        cov = ('cluster',{'groups':d['ea_id']}) if n_cl >= 5 else ('HC3',{})
        try:
            m = smf.ols('y ~ sharia + x + x:sharia',
                        data=d.assign(y=d[outcome], x=d['dist_km'])
                        ).fit(cov_type=cov[0], cov_kwds=cov[1])
            c = m.params.get('sharia', np.nan)
            p = m.pvalues.get('sharia', np.nan)
            row_vals[seg] = f'{c:+.3f}{star(p)}'
        except Exception:
            row_vals[seg] = ' err'
    print(f"  {outcome:<10} {row_vals.get('NW','n/a'):<12} "
          f"{row_vals.get('MB','n/a'):<12} {row_vals.get('E','n/a'):<12}")

# ════════════════════════════════════════════════════════════════════════════════
# TEST C: MUSLIM vs NON-MUSLIM SUBSAMPLE
# Key diagnostic: if the Sharia mechanism is driving trust via religious legitimacy,
# Muslims should show larger effects. Christians gaining the same trust in religious
# leaders and courts as Muslims would suggest a regional confound.
# ════════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('TEST C: MUSLIM vs NON-MUSLIM SUBSAMPLES')
print('If Christians inside the Sharia boundary gain trust in religious leaders /courts')
print('at the same rate as Muslims, that is hard to explain via Sharia mechanism.')
print('="Regional gradient" predicts: non-Muslim effect ≈ Muslim effect.')
print('="Sharia mechanism" predicts: Muslim >> non-Muslim (or non-Muslim ≈ 0 / negative).')
print('='*80)

for rnd_label, df_r in [('R2',r2),('R4',r4)]:
    result = find_muslim_col(df_r)
    if result is None:
        print(f"\n  {rnd_label}: no Muslim proxy found"); continue
    mcol, mcode, sp, nsp, diff = result
    print(f"\n  {rnd_label}: Muslim proxy = {mcol} code {mcode:.0f}  "
          f"(Sharia={sp:.1%}, non-Sharia={nsp:.1%}, Δ={diff:.1%})")

    # Assign muslim indicator on full df
    df_r['muslim'] = (pd.to_numeric(df_r[mcol], errors='coerce') == mcode).astype(float)

    d = df_r[df_r['dist_km'].abs() <= BW_MAIN].dropna(
        subset=['dist_km','ea_id','muslim']).copy()

    print(f"\n  {rnd_label} — Muslim subsample:")
    print(f"  {'Outcome':<10} {'Muslim β':<16} {'Non-Muslim β':<16} {'Interaction β':<14} {'int p'}")
    print("  " + "-"*65)

    for outcome in ['police','courts','relig']:
        d_out = d.dropna(subset=[outcome]).copy()
        if len(d_out) < 80 or d_out[outcome].nunique() < 2: continue
        n_cl = d_out['ea_id'].nunique()
        cov = ('cluster',{'groups':d_out['ea_id']}) if n_cl >= 5 else ('HC3',{})

        # Interaction model: Sharia × Muslim
        try:
            m_int = smf.ols(
                'y ~ sharia + sharia:muslim + x + x:sharia + muslim',
                data=d_out.assign(y=d_out[outcome], x=d_out['dist_km'])
            ).fit(cov_type=cov[0], cov_kwds=cov[1])
            c_s   = m_int.params.get('sharia', np.nan)      # effect for non-Muslim
            c_int = m_int.params.get('sharia:muslim', np.nan)  # additional for Muslim
            p_s   = m_int.pvalues.get('sharia', np.nan)
            p_int = m_int.pvalues.get('sharia:muslim', np.nan)
            c_mus = c_s + c_int
        except Exception:
            c_s = c_int = c_mus = p_s = p_int = np.nan

        # Muslim-only subsample
        d_mus  = d_out[d_out['muslim'] == 1].copy()
        d_non  = d_out[d_out['muslim'] == 0].copy()
        n_mus  = len(d_mus[d_mus['dist_km'].abs() <= BW_MAIN])
        n_non  = len(d_non[d_non['dist_km'].abs() <= BW_MAIN])

        mus_str  = f"{c_mus:+.3f}{star(p_s)}"   # Muslim total effect
        nmus_str = f"{c_s:+.3f}{star(p_s)}"     # non-Muslim effect (= base)
        int_str  = f"{c_int:+.3f}{star(p_int)}" if not np.isnan(c_int) else "n/a"
        int_p_str = f"{p_int:.3f}" if not np.isnan(p_int) else "n/a"
        print(f"  {outcome:<10} {mus_str:<16} {nmus_str:<16} {int_str:<14} {int_p_str}")

    # Separate subsample OLS (for N/sig transparency)
    print(f"\n  Separate-subsample OLS {rnd_label}:")
    print(f"  {'Outcome':<10} {'Muslim β (N)':<22} {'Non-Muslim β (N)':<22}")
    print("  " + "-"*50)
    for outcome in ['police','courts','relig']:
        def sub_ols(dff):
            dff = dff.dropna(subset=[outcome]).copy()
            dff = dff[dff['dist_km'].abs() <= BW_MAIN]
            if len(dff) < 20 or dff['sharia'].nunique() < 2: return 'n/a', np.nan
            n_cl = dff['ea_id'].nunique()
            cov = ('cluster',{'groups':dff['ea_id']}) if n_cl >= 5 else ('HC3',{})
            try:
                m = smf.ols('y ~ sharia + x + x:sharia',
                            data=dff.assign(y=dff[outcome], x=dff['dist_km'])
                            ).fit(cov_type=cov[0], cov_kwds=cov[1])
                c = m.params.get('sharia', np.nan)
                p = m.pvalues.get('sharia', np.nan)
                n = len(dff)
                return f'{c:+.3f}{star(p)} (N={n})', p
            except:
                return 'err', np.nan

        mus_df = df_r[df_r['muslim'] == 1]
        non_df = df_r[df_r['muslim'] == 0]
        ms, _ = sub_ols(mus_df)
        ns, _ = sub_ols(non_df)
        print(f"  {outcome:<10} {ms:<22} {ns:<22}")

print('\nReviewer tests complete.')
