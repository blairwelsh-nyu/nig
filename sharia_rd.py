"""
Spatial RD: Sharia law adoption boundary in Nigeria (year 2000).

Running variable: signed perpendicular distance (km) from each Afrobarometer
respondent to the approximate Sharia / non-Sharia state border.
  positive  = inside a Sharia state
  negative  = inside a non-Sharia state

Outcomes:  trust variables (police, courts, traditional leaders, elected leaders)
           from Afrobarometer R2 (2003) and later rounds where available.

Identification: respondents within a bandwidth of the border should be
similar on pre-determined characteristics; the discontinuity in outcomes
at distance=0 identifies the Sharia effect.

BOUNDARY SOURCE:
The Sharia/non-Sharia boundary is hand-encoded as a LineString tracing the
approximate state borders between the 12 Sharia states (adopted 2000-2001)
and the contiguous non-Sharia states in Nigeria's "Middle Belt". Waypoints
are verified against publicly available Nigeria state-boundary maps.

Sharia states (12):
  Zamfara, Kano, Sokoto, Katsina, Bauchi, Niger(state), Jigawa,
  Kebbi, Yobe, Borno, Kaduna, Gombe

Non-Sharia border states:
  Kwara, Kogi, Nassarawa, Plateau, Taraba, Adamawa
"""
import os, warnings
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, MultiLineString
from sklearn.neighbors import BallTree
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')

os.chdir('/home/user/nig')

# ── 1.  Approximate Sharia boundary ─────────────────────────────────────────
# Traced west → east along state-pair borders.
# Format: (lon, lat) as required by shapely.
# Segments are listed as separate LineStrings and unioned for robustness.

SHARIA_BOUNDARY_COORDS = [
    # Kebbi / Kwara border (west end, Benin Republic border ~3.4°E)
    (3.40, 11.85),
    (3.90, 11.20),
    (4.35, 10.95),
    # Kebbi / Kwara — Niger / Kwara confluence area
    (4.85, 10.65),
    (5.25, 10.40),
    # Niger state north / Kwara-Kogi border
    (5.65, 10.10),
    (6.05,  9.85),
    # Kaduna / Niger-state south and Nassarawa / Kaduna interface
    (6.55,  9.55),
    (7.00,  9.35),
    (7.45,  9.15),
    # Kaduna / Plateau, Bauchi / Nassarawa-Plateau triple point
    (7.95,  9.40),
    (8.50,  9.60),
    (9.00,  9.80),
    # Bauchi / Plateau border continues east
    (9.50,  9.75),
    (9.95,  9.65),
    # Gombe / Taraba and Bauchi / Taraba
    (10.40,  9.35),
    (10.95,  9.10),
    (11.45,  9.00),
    # Gombe / Taraba — Borno / Adamawa
    (12.00,  9.00),
    (12.60,  9.10),
    (13.10,  9.20),
    # Reaches Cameroon border ~13.7°E
    (13.70,  8.80),
]

boundary_line = LineString(SHARIA_BOUNDARY_COORDS)

SHARIA_STATES = {
    'Zamfara', 'Kano', 'Sokoto', 'Katsina', 'Bauchi', 'Niger',
    'Jigawa', 'Kebbi', 'Yobe', 'Borno', 'Kaduna', 'Gombe'
}

# ── 2.  Signed distance function ─────────────────────────────────────────────

def signed_dist_km(lat_arr, lon_arr, state_arr, boundary, sharia_states,
                   earth_r=6371.0):
    """
    Returns signed distance (km) to boundary LineString.
    Positive  = respondent is in a Sharia state.
    Negative  = respondent is in a non-Sharia state.
    """
    n = len(lat_arr)
    dist = np.full(n, np.nan)
    for i in range(n):
        p = Point(lon_arr[i], lat_arr[i])
        # approximate degree-to-km at this latitude
        lat_rad = np.radians(lat_arr[i])
        deg_lat_km = np.pi * earth_r / 180.0
        deg_lon_km = deg_lat_km * np.cos(lat_rad)
        # shapely distance in degrees; convert to km using average degree size
        deg_dist = boundary.distance(p)
        km_est   = deg_dist * np.sqrt(deg_lat_km**2 + deg_lon_km**2) / np.sqrt(2)
        sign = 1.0 if state_arr[i] in sharia_states else -1.0
        dist[i] = sign * km_est
    return dist


# ── 3.  Load survey data ─────────────────────────────────────────────────────

def load_round(rnd):
    df = pd.read_excel(f'NIG_r{rnd}.csv.xlsx')
    df.columns = df.columns.str.lower().str.strip()
    lat_col = next(c for c in df.columns if 'lat' in c)
    lon_col = next(c for c in df.columns if 'lon' in c)
    df = df.rename(columns={lat_col: 'lat', lon_col: 'lon'})
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    return df.dropna(subset=['lat', 'lon'])

# Load ACLED to get state centroids (same approach as DiD)
acled = pd.read_csv('nigeria_acled.csv')
acled['event_date'] = pd.to_datetime(acled['event_date'], errors='coerce')
state_cents = (acled.groupby('admin1')[['latitude', 'longitude']]
               .mean().reset_index())
state_cents.columns = ['state', 'lat', 'lon']

def assign_state(lat_arr, lon_arr):
    cent_rad = np.radians(state_cents[['lat', 'lon']].values)
    tree = BallTree(cent_rad, metric='haversine')
    _, idx = tree.query(np.radians(np.column_stack([lat_arr, lon_arr])), k=1)
    return state_cents['state'].values[idx.flatten()]

# Load rounds 2-6 (R1 is pre-Sharia / messy timing)
rounds = {}
for rnd in [2, 3, 4, 5, 6]:
    try:
        df = load_round(rnd)
        df['round'] = rnd
        df['state'] = assign_state(df['lat'].values, df['lon'].values)
        df['dist_km'] = signed_dist_km(
            df['lat'].values, df['lon'].values,
            df['state'].values, boundary_line, SHARIA_STATES
        )
        df['sharia'] = (df['state'].isin(SHARIA_STATES)).astype(float)
        rounds[rnd] = df
        n_near = (df['dist_km'].abs() <= 150).sum()
        print(f"  R{rnd}: N={len(df)}, N within 150 km of boundary={n_near}, "
              f"Sharia={df['sharia'].mean():.2f}")
    except Exception as e:
        print(f"  R{rnd}: FAIL — {e}")

# ── 4.  Identify outcome columns present in each round ─────────────────────
# Trust variables: R2 uses q43-series; R3-R6 may differ.
# We search for trust-related q-numbers empirically.
# Also check for freedom-of-religion, women's rights items.

print("\n── Column search: 'trust' or key q-numbers ──")
for rnd, df in rounds.items():
    trust_cols = [c for c in df.columns if c.startswith('q43') or c.startswith('q52')
                  or c.startswith('q41') or 'trust' in c.lower()]
    print(f"R{rnd}: {sorted(trust_cols)[:12]}")

# ── 5.  R2 trust variables known from codebook / prior session ──────────────
# R2: q43a-q43l are trust items (0=not at all, 3=a lot)
# q43i = trust police, q43j = trust courts, q43e = trust traditional leaders
# q43d = trust elected local government
TRUST_MAP_R2 = {
    'trust_president':     'q43a',
    'trust_parliament':    'q43b',
    'trust_courts':        'q43j',
    'trust_police':        'q43i',
    'trust_trad_leaders':  'q43e',
    'trust_local_govt':    'q43d',
    'trust_religious':     'q43g',
}

# R3-R6: trust items shift columns; check what's there
# From prior DiD session: q52-series in R5, q55-series in R6 for trust
for rnd in [3, 4, 5, 6]:
    df = rounds.get(rnd)
    if df is None: continue
    for prefix in ['q43', 'q52', 'q55', 'q56']:
        cols = sorted([c for c in df.columns if c.startswith(prefix)])
        if cols:
            print(f"R{rnd} {prefix}*: {cols[:8]}")

# ── 6.  RD estimation function ───────────────────────────────────────────────

def run_rd(df, outcome_col, bandwidth=150, poly=1):
    """Local-linear RD at distance=0."""
    d = df.copy()
    if outcome_col not in d.columns:
        return None
    d[outcome_col] = pd.to_numeric(d[outcome_col], errors='coerce')
    d.loc[d[outcome_col] < 0, outcome_col] = np.nan
    d.loc[d[outcome_col] >= 7, outcome_col] = np.nan
    d = d.dropna(subset=[outcome_col, 'dist_km', 'sharia'])
    d = d[d['dist_km'].abs() <= bandwidth]
    if len(d) < 50 or d[outcome_col].nunique() < 2:
        return None
    d['y'] = d[outcome_col]
    d['x'] = d['dist_km']
    # Polynomial terms interacted with treatment
    if poly == 1:
        formula = 'y ~ sharia + x + x:sharia'
    else:
        formula = 'y ~ sharia + x + x:sharia + I(x**2) + I(x**2):sharia'
    urb = next((c for c in d.columns if 'urban' in c.lower()), None)
    if urb:
        d['urb'] = pd.to_numeric(d[urb], errors='coerce').fillna(0)
        formula += ' + urb'
    try:
        m = smf.ols(formula, data=d).fit(cov_type='HC3')
    except Exception:
        return None
    coef = m.params.get('sharia', np.nan)
    se   = m.bse.get('sharia', np.nan)
    pval = m.pvalues.get('sharia', np.nan)
    n_left  = (d['dist_km'] < 0).sum()
    n_right = (d['dist_km'] > 0).sum()
    if abs(coef) > 10 or se > 10: return None
    return {'outcome': outcome_col, 'round': df['round'].iloc[0],
            'n': len(d), 'n_nonsharia': n_left, 'n_sharia': n_right,
            'rd_coef': coef, 'rd_se': se, 'rd_pval': pval,
            'mean_nonsharia': d.loc[d['sharia']==0, 'y'].mean(),
            'mean_sharia':    d.loc[d['sharia']==1, 'y'].mean()}

# ── 7.  Run RD for R2 trust variables ───────────────────────────────────────

def stars(p):
    if pd.isna(p): return ''
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''

results = []
print("\n\n" + "="*90)
print("SHARIA BOUNDARY SPATIAL RD  (local-linear, bandwidth=150 km, HC3 SE)")
print("Treatment = in Sharia state (positive signed distance from boundary)")
print("="*90)

# R2 analysis
r2 = rounds.get(2)
if r2 is not None:
    print(f"\n── Round 2 (N={len(r2)}, N≤150km={( r2['dist_km'].abs()<=150).sum()}) ──")
    print(f"  {'Outcome':<28} {'N':>5} {'β_RD':>8} {'SE':>7} {'p':>6}  {'μ_NS':>6} {'μ_S':>6}")
    print("  " + "-"*70)
    for label, col in TRUST_MAP_R2.items():
        r = run_rd(r2, col, bandwidth=150)
        if r:
            results.append(r)
            r['label'] = label
            print(f"  {label:<28} {r['n']:>5} {r['rd_coef']:>8.4f} {r['rd_se']:>7.4f} "
                  f"{r['rd_pval']:>6.3f}{stars(r['rd_pval']):>3}  "
                  f"{r['mean_nonsharia']:>6.3f} {r['mean_sharia']:>6.3f}")
        else:
            print(f"  {label:<28}  (skip — col not found or too few obs)")

# ── 8.  Bandwidth sensitivity (R2 trust_police) ─────────────────────────────
print("\n── Bandwidth sensitivity (trust_police, R2) ──")
print(f"  {'BW (km)':>8} {'N':>5} {'β_RD':>8} {'SE':>7} {'p':>6}")
for bw in [50, 75, 100, 150, 200]:
    r = run_rd(r2, 'q43i', bandwidth=bw)
    if r:
        print(f"  {bw:>8} {r['n']:>5} {r['rd_coef']:>8.4f} {r['rd_se']:>7.4f} {r['rd_pval']:>6.3f}{stars(r['rd_pval']):>3}")

# ── 9.  R3-R6: find trust columns automatically and run ─────────────────────
# Trust item naming varies; search for columns with distributions consistent
# with 0-3 scale in states on both sides of the boundary.
print("\n── All rounds: auto-detected trust columns (q43* / q52* prefix) ──")
for rnd in [3, 4, 5, 6]:
    df = rounds.get(rnd)
    if df is None: continue
    sub = df[df['dist_km'].abs() <= 150]
    if len(sub) < 50: continue
    # Try both column families
    for prefix in ['q43', 'q52', 'q55']:
        for suffix in list('abcdefghijklmn'):
            col = prefix + suffix
            if col not in df.columns: continue
            r = run_rd(df, col, bandwidth=150)
            if r and r['rd_pval'] < 0.10:
                r['label'] = col
                results.append(r)
                print(f"  R{rnd} {col:<8} N={r['n']:>4}  β={r['rd_coef']:>8.4f}  "
                      f"SE={r['rd_se']:>6.4f}  p={r['rd_pval']:>6.3f}{stars(r['rd_pval']):>3}  "
                      f"μ_NS={r['mean_nonsharia']:.3f}  μ_S={r['mean_sharia']:.3f}")

# ── 10.  Sorting/placebo check: distribution of covariates at boundary ───────
print("\n── Covariate balance at boundary (R2, BW=150km) ──")
if r2 is not None:
    near = r2[r2['dist_km'].abs() <= 150].copy()
    urb_col = next((c for c in near.columns if 'urban' in c.lower()), None)
    age_col = next((c for c in near.columns if c in ['q1','age','respondent_age']), None)
    edu_col = next((c for c in near.columns if c in ['q97','education','edu']), None)
    for label, col in [('urban', urb_col), ('age', age_col), ('education', edu_col)]:
        if col is None: continue
        near[col] = pd.to_numeric(near[col], errors='coerce')
        g = near.groupby('sharia')[col].mean()
        print(f"  {label}: non-Sharia={g.get(0, np.nan):.3f}  Sharia={g.get(1, np.nan):.3f}")

# ── 11.  Save ────────────────────────────────────────────────────────────────
if results:
    rd_df = pd.DataFrame(results)
    rd_df.to_csv('sharia_rd_results.csv', index=False)
    print(f"\nSaved {len(rd_df)} RD results → sharia_rd_results.csv")

# ── 12.  Descriptive: raw means by distance bin ─────────────────────────────
print("\n── Raw means of trust_police (q43i) by 50-km bins (R2) ──")
if r2 is not None and 'q43i' in r2.columns:
    r2c = r2.copy()
    r2c['q43i'] = pd.to_numeric(r2c['q43i'], errors='coerce')
    r2c.loc[r2c['q43i'] >= 7, 'q43i'] = np.nan
    r2c['bin'] = pd.cut(r2c['dist_km'], bins=np.arange(-300, 351, 50))
    tbl = r2c.groupby('bin')['q43i'].agg(['mean','count']).round(3)
    print(tbl.to_string())
