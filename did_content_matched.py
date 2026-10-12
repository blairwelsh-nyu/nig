"""
Content-matched R5-R6 DiD: uses cross-round question mapping so variables
with different numbers but identical question wording can be compared.

Cross-round pairs verified by matching value-scale distributions in actual
data files (NIG_r5.csv.xlsx / NIG_r6.csv.xlsx) against expected scales.
"""
import os, re
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.neighbors import BallTree
import warnings; warnings.filterwarnings('ignore')

os.chdir('/home/user/nig')

BW_KM = 50; RADIUS_KM = 150; EARTH_R = 6371.0
KIDNAP_RE = re.compile(r'kidnap|abduct|ransom|hostage|forced disappear', re.I)
DELTA_STATES = {'Abia','Rivers','Edo','Delta','Bayelsa','Cross River','Akwa Ibom','Imo','Anambra'}

# ── Content-matched outcome mapping: (r5_col, r6_col, label, note)
# R5 q-number often shifts by 2–6 relative to R6 for same question wording.
# Verified: value-scale distributions match across rounds for every pair.
# DK codes: R5: 9/998/-1; R6: 9/98/-1. Strip all >=7 in run_did.
MATCHED_OUTCOMES = [
    # Direction of country (1=wrong, 2=right; binary)
    ('q7',  'q3',  'direction_of_country',       '1-2 binary; R5 q7 = R6 q3'),
    # Group membership (0=not member, 3=official leader; 0-3)
    ('q25a','q19a','member_religious_group',      '0-3 membership; R5 q25a = R6 q19a'),
    ('q25b','q19b','member_voluntary_assoc',      '0-3 membership; R5 q25b = R6 q19b'),
    # Civic participation (0=never would, 4=often; 0-4)
    ('q26a','q20a','civic_attend_meeting',        '0-4 participation; R5 q26a = R6 q20a'),
    ('q26b','q20b','civic_raise_issue',           '0-4 participation; R5 q26b = R6 q20b'),
    ('q26d','q27e','civic_demonstration',         '0-4 participation; R5 q26d = R6 q27e'),
    # Democracy attitudes (1-3 or 1-5 scale)
    ('q32', 'q30', 'support_elections',          '1-3; R5 q32 = R6 q30'),
    ('q33', 'q31', 'multiparty_support',          '1-5; R5 q33 = R6 q31'),
    ('q35', 'q33', 'oppos_parties_examine',       '1-5; R5 q35 = R6 q33'),
    ('q37', 'q35', 'parliament_makes_laws',       '1-5; R5 q37 = R6 q35'),
    ('q38', 'q36', 'media_checks_govt',           '1-5; R5 q38 = R6 q36'),
    ('q41', 'q39', 'pres_term_limits',            '1-5; R5 q41 = R6 q39'),
    ('q43', 'q41', 'satisfaction_democracy',      '0-4; R5 q43 = R6 q41'),
    ('q49', 'q43', 'obey_govt_always',            '1-5; R5 q49 = R6 q43'),
    # Fear of political violence (0=A lot, 3=Not at all; REVERSED scale)
    ('q54', 'q49', 'fear_political_violence',     '0-3 reversed; R5 q54 = R6 q49'),
    # Bribe experience (0=never, 3=often; 7=no experience with this service)
    ('q61a','q55f','bribe_document_permit',       '0-3 (7=no exp); R5 q61a = R6 q55f'),
    ('q61b','q55h','bribe_water_services',        '0-3 (7=no exp); R5 q61b = R6 q55h'),
]


def kernel_exp(rlat, rlon, elat, elon):
    n = len(rlat)
    exp_arr = np.zeros(n)
    if len(elat) == 0: return exp_arr
    resp_rad = np.radians(np.column_stack([rlat, rlon]))
    evt_rad  = np.radians(np.column_stack([elat, elon]))
    tree = BallTree(evt_rad, metric='haversine')
    r_thresh = RADIUS_KM / EARTH_R
    idx_list, dist_list = tree.query_radius(resp_rad, r=r_thresh, return_distance=True)
    for i in range(n):
        if len(idx_list[i]):
            d_km = dist_list[i] * EARTH_R
            exp_arr[i] = np.exp(-0.5*(d_km/BW_KM)**2).sum()
    return exp_arr


def assign_state(lat_arr, lon_arr, state_cents):
    cent_rad = np.radians(state_cents[['lat','lon']].values)
    tree = BallTree(cent_rad, metric='haversine')
    _, idx = tree.query(np.radians(np.column_stack([lat_arr, lon_arr])), k=1)
    return state_cents['state'].values[idx.flatten()]


# Load ACLED
acled = pd.read_csv('nigeria_acled.csv')
acled['event_date'] = pd.to_datetime(acled['event_date'], errors='coerce')
acled = acled.dropna(subset=['latitude','longitude','event_date'])

def load_round(rnd):
    df = pd.read_excel(f'NIG_r{rnd}.csv.xlsx')
    df.columns = df.columns.str.lower().str.strip()
    lat_col = next(c for c in df.columns if 'lat' in c)
    lon_col = next(c for c in df.columns if 'lon' in c)
    df = df.rename(columns={lat_col:'lat', lon_col:'lon'})
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    return df.dropna(subset=['lat','lon'])

r5 = load_round(5); r6 = load_round(6)

state_cents = acled.groupby('admin1')[['latitude','longitude']].mean().reset_index()
state_cents.columns = ['state','lat','lon']
r5['state'] = assign_state(r5['lat'].values, r5['lon'].values, state_cents)
r6['state'] = assign_state(r6['lat'].values, r6['lon'].values, state_cents)

for df, mid, start in [
    (r5, '2012-11-04', '2011-11-05'),
    (r6, '2015-08-15', '2014-08-15'),
]:
    win = acled[(acled['event_date']>=start) & (acled['event_date']<=mid)]
    knap = win[win['notes'].str.contains(KIDNAP_RE, na=False) |
               win['sub_event_type'].str.contains('Abduction', na=False)]
    viol = win[win['event_type'].isin(['Violence against civilians','Battles',
                                        'Explosions/Remote violence'])]
    df['knap_exp'] = kernel_exp(df['lat'].values, df['lon'].values,
                                 knap['latitude'].values, knap['longitude'].values)
    df['viol_exp'] = kernel_exp(df['lat'].values, df['lon'].values,
                                 viol['latitude'].values, viol['longitude'].values)
    df['delta'] = df['state'].isin(DELTA_STATES).astype(float)

# Build R6 frame with content-matched columns renamed to R5 names.
# Two-step rename via temp names avoids chain conflicts
# (e.g. R6: q41→q43, q43→q49, q49→q54 would clash if done sequentially).
r6_work = r6.copy()
TMP = '__tmp_'

# Step 1: rename all source columns to temp names first
tmp_rename = {}
for r5_col, r6_col, label, note in MATCHED_OUTCOMES:
    if r5_col != r6_col and r6_col in r6_work.columns:
        tmp_rename[r6_col] = TMP + r5_col
r6_work = r6_work.rename(columns=tmp_rename)

# Step 2: drop incumbent R6 columns that share the R5 target name
for r5_col, r6_col, label, note in MATCHED_OUTCOMES:
    if r5_col != r6_col and r5_col in r6_work.columns:
        r6_work = r6_work.drop(columns=[r5_col])

# Step 3: rename temp columns to final R5 names
final_rename = {TMP + r5_col: r5_col
                for r5_col, r6_col, label, note in MATCHED_OUTCOMES
                if r5_col != r6_col and (TMP + r5_col) in r6_work.columns}
r6_renamed = r6_work.rename(columns=final_rename)

# Pool
r5['post'] = 0.0; r6_renamed['post'] = 1.0
pooled = pd.concat([r5, r6_renamed], ignore_index=True, sort=False)
pooled['log_knap'] = np.log1p(pooled['knap_exp'])
pooled['log_viol'] = np.log1p(pooled['viol_exp'])

urb_col = next((c for c in pooled.columns if 'urban' in c.lower()), None)
if urb_col:
    pooled['urb'] = pd.to_numeric(pooled[urb_col], errors='coerce').fillna(0)


def run_did(pooled, var, treat_var):
    df = pooled.copy()
    df[var] = pd.to_numeric(df[var], errors='coerce')
    df.loc[df[var] < 0, var] = np.nan
    if 'bribe' in var or var in ['q61a','q61b']:
        df.loc[df[var] == 7, var] = np.nan
    else:
        df.loc[df[var] >= 7, var] = np.nan
    df = df.dropna(subset=[var, treat_var, 'post', 'state'])
    if len(df) < 300 or df[var].nunique() < 2:
        return None
    y_min, y_max = df[var].min(), df[var].max()
    if y_max == y_min: return None
    df['y'] = (df[var] - y_min) / (y_max - y_min)
    t_std = df[treat_var].std()
    if t_std == 0: return None
    df['t'] = (df[treat_var] - df[treat_var].mean()) / t_std
    ctrl = '+ urb' if 'urb' in df.columns else ''

    # delta is state-level: state FE would absorb all its variation (perfect
    # collinearity). Use HC3 robust SE without state FE for delta specification.
    if treat_var == 'delta':
        formula = f'y ~ t * post + post {ctrl}'
        cov_type = 'HC3'
        cov_kwds = {}
    else:
        formula = f'y ~ t * post + post {ctrl} + C(state)'
        cov_type = 'cluster'
        cov_kwds = {'groups': df['state']}

    try:
        m = smf.ols(formula, data=df).fit(cov_type=cov_type, cov_kwds=cov_kwds)
    except Exception:
        return None

    coef   = m.params.get('t:post', np.nan)
    se     = m.bse.get('t:post', np.nan)
    pval   = m.pvalues.get('t:post', np.nan)
    coef_t = m.params.get('t', np.nan)
    pval_t = m.pvalues.get('t', np.nan)
    return {'var': var, 'treat': treat_var,
            'n': len(df), 'n_r5': (df['post']==0).sum(), 'n_r6': (df['post']==1).sum(),
            'did_coef': coef, 'did_se': se, 'did_pval': pval,
            'main_treat_coef': coef_t, 'main_treat_pval': pval_t}


treat_vars = ['log_knap', 'log_viol', 'delta']

results = []
for r5_col, r6_col, label, note in MATCHED_OUTCOMES:
    if r5_col not in pooled.columns:
        print(f"  SKIP {r5_col} ({label}): not in pooled after rename")
        continue
    for tv in treat_vars:
        r = run_did(pooled, r5_col, tv)
        if r:
            r['label'] = label
            r['note'] = note
            results.append(r)

res_df = pd.DataFrame(results)


def stars(p):
    if pd.isna(p): return ''
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''


print("\n" + "="*110)
print("CONTENT-MATCHED DiD RESULTS (R5→R6 cross-number pairs, verified same wording)")
print("="*110)
print("Note: delta treatment uses HC3 robust SE (no state FE; state FE would be collinear)")
for tv in treat_vars:
    sub = res_df[res_df['treat'] == tv].sort_values('did_pval')
    print(f"\n--- Treatment: {tv} ---")
    print(f"{'Label':<32} {'N':>5} {'β_DiD':>8} {'SE':>7} {'p':>6} {'':>4} {'β_T':>8} {'p_T':>6}")
    print("-"*85)
    for _, row in sub.iterrows():
        print(f"{row['label'][:31]:<32} {int(row['n']):>5} "
              f"{row['did_coef']:>8.4f} {row['did_se']:>7.4f} "
              f"{row['did_pval']:>6.3f} {stars(row['did_pval']):>4} "
              f"{row['main_treat_coef']:>8.4f} {stars(row['main_treat_pval']):>6}")

res_df.to_csv('did_content_matched_results.csv', index=False)
print(f"\nSaved {len(res_df)} results to did_content_matched_results.csv")
