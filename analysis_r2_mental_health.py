"""
Nigeria Afrobarometer Round 2 (2003) × ACLED
Effect of local violence exposure on mental health (anxiety/worry)

Outcome: q93b — "Have you been so worried or anxious that you felt tired,
         worn out, or exhausted?" (0=Never, 1=Once/twice, 2=Many times, 3=Always)
Treatment: ACLED violence events within 50km in 12 months before survey (Oct 2002–Oct 2003)
Design: Cross-sectional OLS with state fixed effects, SE clustered at state
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

RADIUS_KM   = 50
EARTH_R_KM  = 6371
DATA_DIR    = '/home/user/nig'

# ── 1. LOAD AFROBAROMETER ROUND 2 ────────────────────────────────────────────
print("Loading Afrobarometer Round 2 ...")
df = pd.read_excel(f'{DATA_DIR}/NIG_r2.csv.xlsx')
df['dateintr'] = pd.to_datetime(df['dateintr'])

# ── 2. BUILD CLEAN ANALYSIS VARIABLES ────────────────────────────────────────
# Mental health outcome (q93b): 0=Never … 3=Always; 9=DK/missing
df['mhealth'] = pd.to_numeric(df['q93b'], errors='coerce').replace(9, np.nan)

# Physical health control (q93a): same scale
df['phealth_reduced'] = pd.to_numeric(df['q93a'], errors='coerce').replace(9, np.nan)

# Fear of crime (q11a): 0=Never, 1=Once/twice, 2=Several times, 3=Many times, 4=Always
df['fear_crime'] = pd.to_numeric(df['q11a'], errors='coerce').replace(9, np.nan)

# Victimised – theft (q11b), physical attack (q11c)
df['victim_theft']  = pd.to_numeric(df['q11b'], errors='coerce').replace(9, np.nan)
df['victim_attack'] = pd.to_numeric(df['q11c'], errors='coerce').replace(9, np.nan)

# Healthcare deprivation (q9c) — confirmed as "gone without medicines/medical treatment"
df['hlth_dep'] = pd.to_numeric(df['q9c'], errors='coerce').replace([7, 9], np.nan)

# Demographics
df['age']     = pd.to_numeric(df['q80'], errors='coerce').replace(999, np.nan)
df['female']  = (pd.to_numeric(df['q96'], errors='coerce') == 2).astype(float)   # q96: 1=M 2=F
df['urban']   = (pd.to_numeric(df['urbrur'], errors='coerce') == 1).astype(float)
df['educ']    = pd.to_numeric(df['q84'], errors='coerce').replace(9, np.nan)     # 0–8
# Employment: 5=self-employed, 6=formal employed → employed=1; rest=0
df['employed'] = pd.to_numeric(df['q86'], errors='coerce').isin([5, 6]).astype(float)

# State FE identifier
df['state'] = df['locationlevel1'].astype(str)

# Geo
df['lat'] = pd.to_numeric(df['latitude'],  errors='coerce')
df['lon'] = pd.to_numeric(df['longitude'], errors='coerce')

# Drop rows with missing outcome or coordinates
df_clean = df.dropna(subset=['mhealth', 'lat', 'lon']).copy()
print(f"  Analysis sample: {len(df_clean)} respondents, {df_clean['state'].nunique()} states")
print(f"  Survey period: {df_clean['dateintr'].min().date()} – {df_clean['dateintr'].max().date()}")

# ── 3. LOAD & FILTER ACLED ───────────────────────────────────────────────────
print("\nLoading ACLED ...")
acled = pd.read_csv(f'{DATA_DIR}/nigeria_acled.csv')
acled['event_date'] = pd.to_datetime(acled['event_date'])

# 12-month window ending on survey midpoint (Oct 18 2003)
survey_mid   = pd.Timestamp('2003-10-18')
window_start = survey_mid - pd.DateOffset(months=12)
window_end   = survey_mid

# Exclude Protests (non-violent) and Strategic developments
acled_viol = acled[
    (acled['event_date'] >= window_start) &
    (acled['event_date'] <= window_end) &
    (~acled['event_type'].isin(['Protests', 'Strategic developments']))
].copy()

print(f"  ACLED events in window ({window_start.date()} – {window_end.date()}): {len(acled_viol)}")
print(f"  Event types:\n{acled_viol['event_type'].value_counts().to_string()}")
print(f"  Total fatalities: {acled_viol['fatalities'].sum():,}")

# Subsets by event type for heterogeneity analysis
acled_vac     = acled_viol[acled_viol['event_type'] == 'Violence against civilians']
acled_battles = acled_viol[acled_viol['event_type'] == 'Battles']
acled_riots   = acled_viol[acled_viol['event_type'] == 'Riots']

# ── 4. SPATIAL EXPOSURE FUNCTION ─────────────────────────────────────────────
def compute_exposure(respondent_df, events_df, radius_km=RADIUS_KM):
    """
    For each respondent, count events and sum fatalities within radius_km.
    Returns arrays: n_events, n_fatalities, weighted_events (1/(1+d_km))
    """
    if len(events_df) == 0:
        z = np.zeros(len(respondent_df))
        return z, z, z

    resp_coords  = np.radians(respondent_df[['lat', 'lon']].values)
    event_coords = np.radians(events_df[['latitude', 'longitude']].values)
    fatalities   = events_df['fatalities'].values

    tree = BallTree(event_coords, metric='haversine')
    radius_rad = radius_km / EARTH_R_KM
    idx_list, dist_list = tree.query_radius(resp_coords, r=radius_rad, return_distance=True)

    n_ev, n_fat, wt_ev = [], [], []
    for idx, dist in zip(idx_list, dist_list):
        n_ev.append(len(idx))
        n_fat.append(float(fatalities[idx].sum()) if len(idx) > 0 else 0.0)
        if len(idx) > 0:
            dist_km = dist * EARTH_R_KM
            wt_ev.append(float((1 / (1 + dist_km)).sum()))
        else:
            wt_ev.append(0.0)

    return np.array(n_ev), np.array(n_fat), np.array(wt_ev)

print("\nComputing violence exposure measures ...")
# All violent events
n_all, fat_all, wt_all = compute_exposure(df_clean, acled_viol)
# Violence against civilians
n_vac, fat_vac, _      = compute_exposure(df_clean, acled_vac)
# Battles
n_bat, fat_bat, _      = compute_exposure(df_clean, acled_battles)

df_clean['n_events']    = n_all
df_clean['fatalities']  = fat_all
df_clean['wt_events']   = wt_all
df_clean['n_vac']       = n_vac
df_clean['fat_vac']     = fat_vac
df_clean['n_battles']   = n_bat
df_clean['fat_battles'] = fat_bat
df_clean['any_event']   = (df_clean['n_events'] > 0).astype(float)
df_clean['any_vac']     = (df_clean['n_vac'] > 0).astype(float)

# Log-transform count and fatality measures (add 1 to handle zeros)
df_clean['log_fat']     = np.log1p(df_clean['fatalities'])
df_clean['log_fat_vac'] = np.log1p(df_clean['fat_vac'])
df_clean['log_fat_bat'] = np.log1p(df_clean['fat_battles'])

print(f"  % respondents with any violent event within 50km: {df_clean['any_event'].mean()*100:.1f}%")
print(f"  % respondents with any VAC event within 50km:    {df_clean['any_vac'].mean()*100:.1f}%")
print(f"  Mean fatalities within 50km:   {df_clean['fatalities'].mean():.2f}")
print(f"  Median fatalities within 50km: {df_clean['fatalities'].median():.1f}")
print(f"  Max fatalities within 50km:    {df_clean['fatalities'].max():.0f}")

# ── 5. DESCRIPTIVE STATISTICS ────────────────────────────────────────────────
print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS")
print("="*60)
desc_vars = {
    'Mental health (0–3)':     'mhealth',
    'Fear of crime (0–4)':     'fear_crime',
    'Healthcare deprivation':  'hlth_dep',
    'Victim of attack (0–4)':  'victim_attack',
    'Any event within 50km':   'any_event',
    'Fatalities within 50km':  'fatalities',
    'Log fatalities':          'log_fat',
    'Age':                     'age',
    'Female':                  'female',
    'Urban':                   'urban',
    'Education (0–8)':         'educ',
    'Employed':                'employed',
}
rows = []
for label, col in desc_vars.items():
    s = df_clean[col].dropna()
    rows.append({'Variable': label, 'N': int(s.count()),
                 'Mean': round(s.mean(), 3), 'SD': round(s.std(), 3),
                 'Min': round(s.min(), 2),  'Max': round(s.max(), 2)})
print(pd.DataFrame(rows).to_string(index=False))

# Mental health outcome distribution
print("\n--- Mental health (q93b) distribution ---")
vc = df_clean['mhealth'].value_counts(normalize=True).sort_index() * 100
labels = {0: 'Never', 1: 'Once/twice', 2: 'Many times', 3: 'Always'}
for v, pct in vc.items():
    print(f"  {int(v)} ({labels.get(int(v),'')}): {pct:.1f}%")

# ── 6. REGRESSION ANALYSIS ───────────────────────────────────────────────────
print("\n" + "="*60)
print("REGRESSION RESULTS")
print("OLS | Outcome: mental health (q93b, 0–3)")
print("State FE + clustered SE at state level")
print("="*60)

controls = '+ age + female + urban + educ + employed'

def run_ols(formula, data, cluster_var):
    """OLS with cluster-robust SE."""
    m = smf.ols(formula, data=data.dropna(subset=formula.replace('~',' ').split()[:1] + [cluster_var])).fit(
        cov_type='cluster', cov_kwds={'groups': data.dropna()[cluster_var]}
    )
    return m

def display_result(result, label, var):
    coef = result.params.get(var, np.nan)
    se   = result.bse.get(var, np.nan)
    pval = result.pvalues.get(var, np.nan)
    stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
    n    = int(result.nobs)
    r2   = result.rsquared
    print(f"  {label:<42} β={coef:+.4f}  SE={se:.4f}  p={pval:.3f}{stars}  N={n}  R²={r2:.3f}")

print("\n[A] Binary treatment: any violent event within 50km")
formula = f'mhealth ~ any_event + C(state){controls}'
res_a = smf.ols(formula, data=df_clean.dropna(subset=['mhealth','any_event','state','age','female','urban','educ','employed'])).fit(
    cov_type='cluster', cov_kwds={'groups': df_clean.dropna(subset=['mhealth','any_event','state','age','female','urban','educ','employed'])['state']}
)
display_result(res_a, 'Any event within 50km', 'any_event')

print("\n[B] Log fatalities within 50km (all violent events)")
formula_b = f'mhealth ~ log_fat + C(state){controls}'
sub_b = df_clean.dropna(subset=['mhealth','log_fat','state','age','female','urban','educ','employed'])
res_b = smf.ols(formula_b, data=sub_b).fit(
    cov_type='cluster', cov_kwds={'groups': sub_b['state']}
)
display_result(res_b, 'Log(fatalities+1) within 50km', 'log_fat')

print("\n[C] Log fatalities — Violence Against Civilians only")
formula_c = f'mhealth ~ log_fat_vac + C(state){controls}'
sub_c = df_clean.dropna(subset=['mhealth','log_fat_vac','state','age','female','urban','educ','employed'])
res_c = smf.ols(formula_c, data=sub_c).fit(
    cov_type='cluster', cov_kwds={'groups': sub_c['state']}
)
display_result(res_c, 'Log(VAC fatalities+1) within 50km', 'log_fat_vac')

print("\n[D] Log fatalities — Battles only")
formula_d = f'mhealth ~ log_fat_bat + C(state){controls}'
sub_d = df_clean.dropna(subset=['mhealth','log_fat_bat','state','age','female','urban','educ','employed'])
res_d = smf.ols(formula_d, data=sub_d).fit(
    cov_type='cluster', cov_kwds={'groups': sub_d['state']}
)
display_result(res_d, 'Log(battle fatalities+1) within 50km', 'log_fat_bat')

# Full coefficient table for preferred specification (B)
print("\n" + "="*60)
print("FULL TABLE — Preferred spec [B]: Log fatalities")
print("="*60)
summary_b = res_b.summary2().tables[1][['Coef.','Std.Err.','t','P>|t|']]
# Show only non-FE rows
non_fe = [i for i in summary_b.index if not i.startswith('C(state)')]
print(summary_b.loc[non_fe].round(4).to_string())

# ── 7. HETEROGENEITY BY GENDER ───────────────────────────────────────────────
print("\n" + "="*60)
print("HETEROGENEITY BY GENDER")
print("="*60)
for sex, label in [(0.0, 'Women'), (1.0, 'Men')]:
    sub = df_clean[df_clean['female'] == sex].dropna(
        subset=['mhealth','log_fat','state','age','urban','educ','employed'])
    if len(sub) < 50:
        continue
    res = smf.ols(f'mhealth ~ log_fat + C(state) + age + urban + educ + employed',
                  data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['state']})
    display_result(res, f'{label} (n={len(sub)})', 'log_fat')

# ── 8. ROBUSTNESS: DIFFERENT RADII ───────────────────────────────────────────
print("\n" + "="*60)
print("ROBUSTNESS: DIFFERENT SPATIAL RADII")
print("="*60)
for r in [25, 50, 75, 100]:
    n_r, fat_r, _ = compute_exposure(df_clean, acled_viol, radius_km=r)
    df_clean[f'log_fat_{r}km'] = np.log1p(fat_r)
    sub_r = df_clean.dropna(subset=['mhealth',f'log_fat_{r}km','state','age','female','urban','educ','employed'])
    res_r = smf.ols(f'mhealth ~ log_fat_{r}km + C(state){controls}', data=sub_r).fit(
        cov_type='cluster', cov_kwds={'groups': sub_r['state']}
    )
    display_result(res_r, f'Log fatalities within {r}km', f'log_fat_{r}km')

# ── 9. SECONDARY OUTCOME: FEAR OF CRIME ──────────────────────────────────────
print("\n" + "="*60)
print("SECONDARY OUTCOME: FEAR OF CRIME (q11a, 0–4)")
print("="*60)
sub_fc = df_clean.dropna(subset=['fear_crime','log_fat','state','age','female','urban','educ','employed'])
res_fc = smf.ols(f'fear_crime ~ log_fat + C(state){controls}', data=sub_fc).fit(
    cov_type='cluster', cov_kwds={'groups': sub_fc['state']}
)
display_result(res_fc, 'Log(fatalities+1) → fear of crime', 'log_fat')

sub_fc2 = df_clean.dropna(subset=['fear_crime','log_fat_vac','state','age','female','urban','educ','employed'])
res_fc2 = smf.ols(f'fear_crime ~ log_fat_vac + C(state){controls}', data=sub_fc2).fit(
    cov_type='cluster', cov_kwds={'groups': sub_fc2['state']}
)
display_result(res_fc2, 'Log(VAC fatalities+1) → fear of crime', 'log_fat_vac')

print("\nDone.")
