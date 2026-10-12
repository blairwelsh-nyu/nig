"""
Revised Sharia RD addressing six reviewer concerns:
  1. State assignment from survey variable + official NE admin1 boundary
  2. EA-clustered SEs + wild cluster bootstrap by state
  3. EA concentration diagnostic (distance bands)
  4. rdrobust MSE-optimal bandwidth (CCT)
  5. Compound treatment balance (education, Muslim share, conflict)
  6. DiD-in-discontinuities pooling R1 + R2
  7. Multi-round RD (R2 / R4 / R5 / R6)
"""
import os, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
import statsmodels.formula.api as smf
from rdrobust import rdrobust
from wildboottest.wildboottest import wildboottest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')

os.chdir('/home/user/nig')

# ── Constants ─────────────────────────────────────────────────────────────────
SHARIA_STATES_FULL   = {'Zamfara','Kano','Sokoto','Katsina','Bauchi','Niger',
                         'Jigawa','Kebbi','Yobe','Borno','Kaduna','Gombe'}
SHARIA_STATES_STRICT = {'Zamfara','Kano','Sokoto','Katsina','Bauchi','Niger',
                         'Jigawa','Kebbi','Kaduna'}
BW_MAIN = 150

# ── Official boundary from Natural Earth admin1 ───────────────────────────────
print("Building official Sharia boundary from Natural Earth admin1...")
nga = gpd.read_file('/tmp/nga_admin1.gpkg')
# NE name map to match our state set
NE_FIX = {'Federal Capital Territory': 'FCT', 'Nassarawa': 'Nasarawa'}
nga['state'] = nga['name'].replace(NE_FIX)
nga['sharia_full']   = nga['state'].isin(SHARIA_STATES_FULL)
nga['sharia_strict'] = nga['state'].isin(SHARIA_STATES_STRICT)

sharia_union    = unary_union(nga[nga['sharia_full']].geometry)
nonsharia_union = unary_union(nga[~nga['sharia_full']].geometry)
# Shared border: intersection of their boundaries
official_boundary = sharia_union.boundary.intersection(nonsharia_union.boundary)
print(f"  Official boundary geometry type: {official_boundary.geom_type}")

def signed_dist_km_official(lat, lon, sharia_union=sharia_union,
                             boundary=official_boundary, earth_r=6371.0):
    p = Point(lon, lat)
    lat_rad = np.radians(lat)
    dlat = np.pi * earth_r / 180.0
    dlon = dlat * np.cos(lat_rad)
    km = boundary.distance(p) * np.sqrt(dlat**2 + dlon**2) / np.sqrt(2)
    sign = 1.0 if sharia_union.contains(p) else -1.0
    return sign * km

# ── State extraction from survey geoname_adm_name ────────────────────────────
def extract_state(geoname_str):
    parts = str(geoname_str).split('|')
    if len(parts) < 4:
        return None
    raw = parts[3].replace(' State', '').strip()
    return raw

# ── Generic data loader ───────────────────────────────────────────────────────
def load_round(fname, trust_map, scale='0-3', extra_cols=None):
    """
    trust_map: {outcome_name: column_name}
    scale: '0-3', '1-4', '1-5'  (how to recode to 0-3)
    extra_cols: list of extra column names to carry through
    """
    df = pd.read_excel(fname)
    df.columns = df.columns.str.lower().str.strip()

    # Coordinates
    lat_col = next((c for c in df.columns if c in ('latitude','lat')), None)
    lon_col = next((c for c in df.columns if c in ('longitude','lon')), None)
    if lat_col is None or lon_col is None:
        raise ValueError(f"No lat/lon in {fname}")
    df = df.rename(columns={lat_col: 'lat', lon_col: 'lon'})
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df = df.dropna(subset=['lat','lon']).reset_index(drop=True)

    # State from survey variable
    if 'geoname_adm_name' in df.columns:
        df['state_survey'] = df['geoname_adm_name'].apply(extract_state)
    elif 'region' in df.columns and df['region'].dtype == object:
        df['state_survey'] = df['region'].str.strip()
    else:
        df['state_survey'] = None

    # Sharia treatment from survey state
    df['sharia'] = df['state_survey'].isin(SHARIA_STATES_FULL).astype(float)

    # Running variable from official boundary
    print(f"  Computing signed distances for {len(df)} respondents...")
    df['dist_km'] = [signed_dist_km_official(r['lat'], r['lon'])
                     for _, r in df.iterrows()]

    # EA identifier
    ea_col = next((c for c in ['uniqueea','eanumb'] if c in df.columns
                   and df[c].nunique() > 1), None)
    df['ea_id'] = df[ea_col] if ea_col else np.arange(len(df))

    # Trust outcomes
    for name, col in trust_map.items():
        if col not in df.columns:
            df[name] = np.nan
            continue
        v = pd.to_numeric(df[col], errors='coerce')
        if scale == '0-3':
            v = v.where(v <= 3)
        elif scale == '1-4':
            v = v.where(v <= 4) - 1        # → 0-3
        elif scale == '1-5':
            v = v.where(v <= 5) - 1        # → 0-4; rescale to 0-3
            v = v * (3/4)
        df[name] = v

    # Extra columns
    if extra_cols:
        for c in extra_cols:
            if c not in df.columns:
                df[c] = np.nan

    return df

# ── Core RD helpers ───────────────────────────────────────────────────────────
def rd_ols(df, outcome, bw, cluster_col=None):
    d = df.dropna(subset=[outcome,'dist_km','sharia']).copy()
    d = d[(d['dist_km'].abs() <= bw)]
    if len(d) < 50 or d[outcome].nunique() < 2:
        return None
    formula = 'y ~ sharia + x + x:sharia'
    data = d.assign(y=d[outcome], x=d['dist_km'])
    try:
        if cluster_col and d[cluster_col].nunique() >= 5:
            groups = d[cluster_col]
            m = smf.ols(formula, data=data).fit(
                cov_type='cluster', cov_kwds={'groups': groups})
        else:
            m = smf.ols(formula, data=data).fit(cov_type='HC3')
    except Exception:
        return None
    c = m.params.get('sharia', np.nan)
    s = m.bse.get('sharia', np.nan)
    p = m.pvalues.get('sharia', np.nan)
    if abs(c) > 3 or s > 2:
        return None
    return {'coef': c, 'se': s, 'pval': p, 'n': len(d),
            'n_cl': d[cluster_col].nunique() if cluster_col else None}

def rd_cct(df, outcome, bw_override=None):
    d = df.dropna(subset=[outcome,'dist_km','sharia']).copy()
    if bw_override:
        d = d[d['dist_km'].abs() <= bw_override]
    if len(d) < 50:
        return None
    try:
        r = rdrobust(d[outcome].values, d['dist_km'].values,
                     kernel='triangular', bwselect='mserd')
        return {
            'bw_l': r.bws[0,0], 'bw_r': r.bws[0,1],
            'coef': r.coef[0,0], 'se': r.se[2,0], 'pval': r.pv[2,0],
            'n': len(d),
        }
    except Exception as e:
        return {'error': str(e)}

def star(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.10:  return '†'
    return ''

# ════════════════════════════════════════════════════════════════════════════
print("\nLoading R2 (2003) with official boundary + survey states...")
r2 = load_round('NIG_r2.csv.xlsx',
                {'police':'q43i','courts':'q43j',
                 'trust_pres':'q43a','trust_parl':'q43b',
                 'trust_trad':'q43e','trust_relig':'q43g',
                 'dem_supply':'q47'},
                scale='0-3',
                extra_cols=['q86','q1b','uniqueea'])
# Muslim indicator
r2['muslim'] = (r2['q86'] == 6).astype(float)
# Education (q1b: 1=none, 2=primary, 3=secondary, 4=post-sec, 5=university)
r2['educ'] = pd.to_numeric(r2['q1b'], errors='coerce').where(lambda x: x <= 5)

# Dem supply recode: 0→NaN, 9→NaN, then -1 to get 0-3
r2['dem_supply'] = pd.to_numeric(r2['q47'], errors='coerce')
r2.loc[r2['dem_supply'].isin([0,9]), 'dem_supply'] = np.nan
r2['dem_supply'] -= 1

print(f"  R2: {len(r2)} obs, {r2['state_survey'].nunique()} states, "
      f"{r2['ea_id'].nunique()} EAs")
print(f"  Sharia states in R2 sample: "
      f"{sorted(r2[r2['sharia']==1]['state_survey'].dropna().unique())}")

# ════════════════════════════════════════════════════════════════════════════
print()
print('='*70)
print('FIX 1 + 2: REVISED MAIN ESTIMATES')
print('  State: survey geoname_adm_name | Boundary: Natural Earth admin1')
print('  Clustering: EA-clustered SEs (uniqueea)')
print('='*70)
print()
print(f'  {"Outcome":<20}  {"N":>5}  {"N_EA":>6}  {"β_OLS_HC3":>10}  '
      f'{"β_OLS_EA":>10}  {"p_HC3":>8}  {"p_EA":>8}')
print('  ' + '-'*75)

for col, label in [('police','Police'), ('courts','Courts'),
                   ('trust_pres','President'), ('trust_relig','Relig. leaders')]:
    r_hc3 = rd_ols(r2, col, BW_MAIN, cluster_col=None)
    r_ea  = rd_ols(r2, col, BW_MAIN, cluster_col='ea_id')
    if r_hc3 and r_ea:
        print(f'  {label:<20}  {r_hc3["n"]:>5}  {r_ea["n_cl"]:>6}  '
              f'{r_hc3["coef"]:>+10.3f}  {r_ea["coef"]:>+10.3f}  '
              f'{r_hc3["pval"]:>7.3f}{star(r_hc3["pval"]):2}  '
              f'{r_ea["pval"]:>7.3f}{star(r_ea["pval"]):2}')

# Wild cluster bootstrap by state
print()
print('  Wild cluster bootstrap by state (police, BW=150 km):')
d_pol = r2.dropna(subset=['police','dist_km','sharia']).copy()
d_pol = d_pol[d_pol['dist_km'].abs() <= BW_MAIN]
d_pol = d_pol.assign(y=d_pol['police'], x=d_pol['dist_km'],
                     xi=d_pol['dist_km']*d_pol['sharia'])
try:
    # Design matrix: sharia, x, x*sharia
    X = d_pol[['sharia','x','xi']].copy()
    X.insert(0, 'const', 1.0)
    wcb = wildboottest(X.values, d_pol['state_survey'].values,
                       d_pol['y'].values, param=1,
                       B=999, seed=42, impose_null=False)
    print(f'    β = {wcb[0]:+.3f}  p_WCB = {wcb[1]:.3f}')
except Exception as e:
    print(f'    Wild bootstrap error: {e}')

# ════════════════════════════════════════════════════════════════════════════
print()
print('='*70)
print('FIX 3: EA CONCENTRATION BY DISTANCE BAND')
print('='*70)
d_bw = r2.dropna(subset=['police','dist_km']).copy()
d_bw = d_bw[d_bw['dist_km'].abs() <= BW_MAIN]
bands = [(0,15),(15,50),(50,100),(100,150)]
print(f'\n  {"Band":>12}  {"EAs_Sharia":>12}  {"EAs_NonS":>10}  '
      f'{"N_Sharia":>10}  {"N_NonS":>8}')
print('  ' + '-'*58)
for lo, hi in bands:
    mask = (d_bw['dist_km'].abs() > lo) & (d_bw['dist_km'].abs() <= hi)
    sub = d_bw[mask]
    s  = sub[sub['sharia']==1]
    ns = sub[sub['sharia']==0]
    print(f'  {lo:>3}–{hi:>3} km   '
          f'{s["ea_id"].nunique():>12}  {ns["ea_id"].nunique():>10}  '
          f'{len(s):>10}  {len(ns):>8}')
print()
print('  0-15 km Sharia EAs:')
near_s = d_bw[(d_bw['dist_km'].abs() <= 15) & (d_bw['sharia']==1)]
print(f'    {near_s["ea_id"].nunique()} distinct EAs, '
      f'{len(near_s)} respondents, '
      f'states: {sorted(near_s["state_survey"].dropna().unique())}')
near_ns = d_bw[(d_bw['dist_km'].abs() <= 15) & (d_bw['sharia']==0)]
print(f'    Non-Sharia: {near_ns["ea_id"].nunique()} EAs, '
      f'{len(near_ns)} respondents')

# ════════════════════════════════════════════════════════════════════════════
print()
print('='*70)
print('FIX 4: rdrobust MSE-OPTIMAL BANDWIDTH (CCT)')
print('='*70)
print(f'\n  {"Outcome":<15}  {"BW_L":>7}  {"BW_R":>7}  {"β_CCT":>8}  '
      f'{"SE_CCT":>7}  {"p_CCT":>8}')
print('  ' + '-'*58)
for col, label in [('police','Police'), ('courts','Courts')]:
    r = rd_cct(r2, col)
    if r and 'error' not in r:
        print(f'  {label:<15}  {r["bw_l"]:>7.1f}  {r["bw_r"]:>7.1f}  '
              f'{r["coef"]:>+8.3f}  {r["se"]:>7.3f}  '
              f'{r["pval"]:>7.3f}{star(r["pval"])}')
    elif r:
        print(f'  {label:<15}  Error: {r["error"]}')

# ════════════════════════════════════════════════════════════════════════════
print()
print('='*70)
print('FIX 5: COMPOUND TREATMENT BALANCE')
print('  Education (q1b), Muslim (q86=6), democracy-demand proxy,')
print('  conflict exposure (ACLED events within 150 km / 3 years pre-survey)')
print('='*70)

# ACLED conflict exposure: count events within 150km of respondent in 2000-2003
acled = pd.read_csv('nigeria_acled.csv')
acled['year'] = pd.to_datetime(acled['event_date'], errors='coerce').dt.year
acled_pre = acled[acled['year'] <= 2003].copy()

def conflict_exposure(lat, lon, events=acled_pre, radius_km=150):
    """Count ACLED events within radius_km of respondent (haversine)."""
    R = 6371.0
    lat_r = np.radians(events['latitude'].values)
    lon_r = np.radians(events['longitude'].values)
    dlat = lat_r - np.radians(lat)
    dlon = lon_r - np.radians(lon)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat)) * np.cos(lat_r) * np.sin(dlon/2)**2
    dist = 2 * R * np.arcsin(np.sqrt(np.clip(a,0,1)))
    return (dist <= radius_km).sum()

print("  Computing conflict exposure (this takes ~30 seconds)...")
# Sample for speed — compute for BW subsample only
bw_mask = r2['dist_km'].abs() <= BW_MAIN
r2_bw = r2[bw_mask].copy()
r2_bw['conflict'] = [conflict_exposure(row['lat'], row['lon'])
                     for _, row in r2_bw.iterrows()]
# Merge back
r2.loc[bw_mask, 'conflict'] = r2_bw['conflict'].values

# Balance table
print()
print(f'  {"Covariate":<25}  {"N":>5}  {"β_RD":>8}  {"SE":>7}  {"p":>8}')
print('  ' + '-'*58)
for col, label in [('educ','Education (q1b, 1-5)'),
                   ('muslim','Muslim (q86=6)'),
                   ('conflict','Conflict events (≤150km)')]:
    r_b = rd_ols(r2, col, BW_MAIN, cluster_col='ea_id')
    if r_b:
        print(f'  {label:<25}  {r_b["n"]:>5}  {r_b["coef"]:>+8.3f}  '
              f'{r_b["se"]:>7.3f}  {r_b["pval"]:>7.3f}{star(r_b["pval"])}')
    else:
        print(f'  {label:<25}  insufficient data')

# ════════════════════════════════════════════════════════════════════════════
print()
print('='*70)
print('FIX 6: DiD-in-DISCONTINUITIES  (R1 pre + R2 post, pooled)')
print('  y ~ sharia*post + x*sharia + x*post + x*sharia*post')
print('  β(sharia×post) = DiD-RD causal estimate')
print('='*70)

print("\nLoading R1 (2000)...")
r1_raw = pd.read_excel('NIG_r1.csv.xlsx')
r1_raw.columns = r1_raw.columns.str.lower().str.strip()
# R1 state from region column
r1_raw['lat'] = pd.to_numeric(r1_raw['latitude'], errors='coerce')
r1_raw['lon'] = pd.to_numeric(r1_raw['longitude'], errors='coerce')
r1_raw = r1_raw.dropna(subset=['lat','lon']).reset_index(drop=True)
# Use region column directly (has state names like 'Kano', 'Borno')
r1_raw['state_survey'] = r1_raw['region'].astype(str).str.strip()
r1_raw['sharia'] = r1_raw['state_survey'].isin(SHARIA_STATES_FULL).astype(float)
print(f"  Computing signed distances for {len(r1_raw)} R1 respondents...")
r1_raw['dist_km'] = [signed_dist_km_official(r['lat'], r['lon'])
                     for _, r in r1_raw.iterrows()]
ea_col_r1 = next((c for c in ['uniqueea','eanumb'] if c in r1_raw.columns
                  and r1_raw[c].nunique() > 1), None)
r1_raw['ea_id'] = r1_raw[ea_col_r1] if ea_col_r1 else np.arange(len(r1_raw))
# Trust (1-4 → 0-3)
for col, name in [('trspol','police'),('trscts','courts')]:
    v = pd.to_numeric(r1_raw[col], errors='coerce')
    r1_raw[name] = v.where(v <= 4) - 1

r1_raw['post'] = 0.0
r2_did = r2.copy()
r2_did['post'] = 1.0

# Pool
pool = pd.concat([r1_raw, r2_did], ignore_index=True)
print(f"  Pooled dataset: {len(pool)} obs ({len(r1_raw)} R1 + {len(r2_did)} R2)")

print()
print(f'  {"Outcome":<12}  {"N":>5}  {"β(sharia)":>10}  '
      f'{"β(post)":>10}  {"β(sharia×post)":>15}  {"p(DiD)":>8}')
print('  ' + '-'*68)
for col, label in [('police','Police'),('courts','Courts')]:
    d = pool.dropna(subset=[col,'dist_km','sharia','post']).copy()
    d = d[d['dist_km'].abs() <= BW_MAIN]
    if len(d) < 100:
        print(f'  {label:<12}  too few obs'); continue
    try:
        m = smf.ols(
            'y ~ sharia + post + sharia:post + x + x:sharia + x:post + x:sharia:post',
            data=d.assign(y=d[col], x=d['dist_km'])
        ).fit(cov_type='cluster', cov_kwds={'groups': d['ea_id']})
        c_s   = m.params.get('sharia', np.nan)
        c_p   = m.params.get('post',   np.nan)
        c_did = m.params.get('sharia:post', np.nan)
        p_did = m.pvalues.get('sharia:post', np.nan)
        print(f'  {label:<12}  {len(d):>5}  {c_s:>+10.3f}  '
              f'{c_p:>+10.3f}  {c_did:>+15.3f}  '
              f'{p_did:>7.3f}{star(p_did)}')
    except Exception as e:
        print(f'  {label}: error — {e}')

# ════════════════════════════════════════════════════════════════════════════
print()
print('='*70)
print('FIX 7: MULTI-ROUND RD  (R2 2003 / R4 2008 / R5 2012 / R6 ~2015)')
print('  Trust scale 0-3 in all four rounds')
print('='*70)

round_specs = [
    ('R2', 'NIG_r2.csv.xlsx', {'police':'q43i','courts':'q43j'}, '0-3', 2003),
    ('R4', 'NIG_r4.csv.xlsx', {'police':'q49c','courts':'q49d'}, '0-3', 2008),
    ('R5', 'NIG_r5.csv.xlsx', {'police':'q52c','courts':'q52d'}, '0-3', 2012),
    ('R6', 'NIG_r6.csv.xlsx', {'police':'q52c','courts':'q52d'}, '0-3', 2015),
]

multi_rows = []
print()
print(f'  {"Round":<6}  {"Year":>5}  {"N_pol":>7}  {"β_pol":>8}  '
      f'{"p_pol":>7}  {"N_cts":>7}  {"β_cts":>8}  {"p_cts":>7}')
print('  ' + '-'*65)

for rnd, fname, tmap, scale, year in round_specs:
    if rnd == 'R2':
        df = r2  # already loaded
    else:
        print(f"  Loading {rnd} ({fname})...")
        df = load_round(fname, tmap, scale=scale)
    rp = rd_ols(df, 'police', BW_MAIN, cluster_col='ea_id')
    rc = rd_ols(df, 'courts', BW_MAIN, cluster_col='ea_id')
    np_ = rp['n'] if rp else 0
    nc_ = rc['n'] if rc else 0
    cp  = rp['coef'] if rp else np.nan
    pp  = rp['pval'] if rp else np.nan
    cc  = rc['coef'] if rc else np.nan
    pc  = rc['pval'] if rc else np.nan
    sp  = star(pp) if rp else ''
    sc  = star(pc) if rc else ''
    print(f'  {rnd:<6}  {year:>5}  {np_:>7}  {cp:>+8.3f}{sp:<3}  '
          f'{pp:>6.3f}  {nc_:>7}  {cc:>+8.3f}{sc:<3}  {pc:>6.3f}')
    multi_rows.append({'round': rnd, 'year': year,
                       'coef_pol': cp, 'pval_pol': pp,
                       'coef_cts': cc, 'pval_cts': pc})

# R3 (1-5 scale, rescaled)
print()
print('  R3 (2005) — 1-5 scale, rescaled to 0-3:')
df_r3 = load_round('NIG_r3.csv.xlsx',
                   {'police':'q52c','courts':'q52d'}, scale='1-5')
rp3 = rd_ols(df_r3, 'police', BW_MAIN, cluster_col='ea_id')
rc3 = rd_ols(df_r3, 'courts', BW_MAIN, cluster_col='ea_id')
if rp3 and rc3:
    print(f'    Police: β={rp3["coef"]:+.3f}  p={rp3["pval"]:.3f}{star(rp3["pval"])}  N={rp3["n"]}')
    print(f'    Courts: β={rc3["coef"]:+.3f}  p={rc3["pval"]:.3f}{star(rc3["pval"])}  N={rc3["n"]}')

# ── Figure: multi-round coefficient plot ─────────────────────────────────────
df_mr = pd.DataFrame(multi_rows).dropna(subset=['coef_pol'])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
fig.patch.set_facecolor('white')
C = '#3d7ab5'

for ax, coef_col, pval_col, title in [
    (ax1, 'coef_pol', 'pval_pol', 'Trust: Police'),
    (ax2, 'coef_cts', 'pval_cts', 'Trust: Courts'),
]:
    d = df_mr.dropna(subset=[coef_col, pval_col])
    for i, row in d.iterrows():
        c, p = row[coef_col], row[pval_col]
        # Approximate SE from p-value (two-sided t-test)
        from scipy import stats
        se_approx = abs(c) / max(stats.norm.ppf(1 - p/2), 0.01) if p < 0.99 else 0.3
        ci = 1.96 * se_approx
        ax.plot([c-ci, c+ci], [row['year'], row['year']],
                color=C, linewidth=6, solid_capstyle='butt', alpha=0.5)
        mk = 'o' if p < 0.05 else 's'
        ax.plot(c, row['year'], marker=mk, markersize=10,
                color='white', markeredgecolor=C, markeredgewidth=2, zorder=3)
        ax.text(c, row['year'] + 0.8, f'{c:+.2f}{star(p)}',
                ha='center', va='bottom', fontsize=8)
    ax.axvline(0, color='#333', lw=0.8, ls='--', alpha=0.6)
    ax.set_yticks(d['year'].tolist())
    ax.set_yticklabels([f'{rnd}\n({yr})' for rnd, yr in
                        zip(d['round'].tolist(), d['year'].tolist())], fontsize=9)
    ax.set_xlabel('β (Sharia effect, 0-3 scale)\nLocal-linear RD, EA-clustered SE, BW=150 km',
                  fontsize=8.5)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    ax.set_facecolor('white')

fig.suptitle('Multi-Round Sharia RD: R2 (2003) through R6 (2015)',
             fontsize=11, fontweight='bold')
plt.tight_layout(pad=1.5)
plt.savefig('sharia_rd_multiround_fig.png', dpi=200, bbox_inches='tight',
            facecolor='white')
print('\nSaved sharia_rd_multiround_fig.png')

# ── Save revised summary ──────────────────────────────────────────────────────
out_rows = []
for rnd, fname, tmap, scale, year in round_specs:
    df_use = r2 if rnd == 'R2' else None  # already loaded above? no - use multi_rows
    pass

print('\nAll revised checks complete.')
