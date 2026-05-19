"""
Nigeria Violence × Health: Spatial Analysis (Rounds 1–6)
=========================================================
Design:
  1. Kernel-weighted ACLED exposure (space-time decay) for each respondent
  2. Pooled OLS: all rounds, round + state FE
  3. Geographic RDD: distance-to-nearest-event as running variable
  4. Distance-decay plot: effect at varying radii
  5. Boko Haram zone boundary analysis (rounds 4–6)

Outcome: healthcare deprivation (goes without medicine/medical treatment)
  R1: povhth  (0–3)
  R2: q9c     (0–4)
  R3–R6: q8c  (0–4)  [R6 has -1 codes → treated as missing]

Treatment: ACLED violent events within 12 months before survey
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

EARTH_R  = 6371.0   # km
DATA_DIR = '/home/user/nig'

# ─────────────────────────────────────────────────────────────────────────────
# 1.  ROUND METADATA
# ─────────────────────────────────────────────────────────────────────────────
ROUNDS = {
    1: dict(file='NIG_r1.csv.xlsx', hvar='povhth',  hscale=3,
            survey_mid=pd.Timestamp('2000-11-15'), state_col='locationlevel1'),
    2: dict(file='NIG_r2.csv.xlsx', hvar='q9c',    hscale=4,
            survey_mid=None,  # use dateintr
            state_col='locationlevel1'),
    3: dict(file='NIG_r3.csv.xlsx', hvar='q8c',    hscale=4,
            survey_mid=None,  state_col='region'),
    4: dict(file='NIG_r4.csv.xlsx', hvar='q8c',    hscale=4,
            survey_mid=None,  state_col='locationlevel1'),
    5: dict(file='NIG_r5.csv.xlsx', hvar='q8c',    hscale=4,
            survey_mid=None,  state_col='locationlevel1'),
    6: dict(file='NIG_r6.csv.xlsx', hvar='q8c',    hscale=4,
            survey_mid=pd.Timestamp('2015-08-15'), state_col='locationlevel1'),
}

# ─────────────────────────────────────────────────────────────────────────────
# 2.  LOAD ACLED
# ─────────────────────────────────────────────────────────────────────────────
print("Loading ACLED ...")
acled = pd.read_csv(f'{DATA_DIR}/nigeria_acled.csv',
                    usecols=['event_date','event_type','actor1','latitude',
                             'longitude','fatalities','admin1'])
acled['event_date'] = pd.to_datetime(acled['event_date'])
acled = acled[~acled['event_type'].isin(['Protests','Strategic developments'])].copy()
acled['is_vac']   = (acled['event_type'] == 'Violence against civilians').astype(int)
acled['is_boko']  = acled['actor1'].str.contains('Boko Haram', na=False).astype(int)
acled['is_mend']  = acled['actor1'].str.contains('MEND|Niger Delta|Ijaw|Movement for', na=False).astype(int)
print(f"  {len(acled):,} violent events  |  "
      f"{acled['fatalities'].sum():,} fatalities  |  "
      f"{acled['event_date'].min().year}–{acled['event_date'].max().year}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  SPATIAL EXPOSURE FUNCTION  (kernel-weighted)
# ─────────────────────────────────────────────────────────────────────────────
def kernel_exposure(resp_df, events_df, bw_km=50):
    """
    For every respondent, compute Gaussian-kernel weighted fatalities
    using events within 3×bandwidth.  Returns array of length len(resp_df).
    Also returns distance to nearest event (km), and distance arrays for RDD.
    """
    if len(events_df) == 0:
        n = len(resp_df)
        return np.zeros(n), np.full(n, np.nan), np.zeros(n)

    resp_rad  = np.radians(resp_df[['lat','lon']].values)
    event_rad = np.radians(events_df[['latitude','longitude']].values)
    fat       = events_df['fatalities'].values.astype(float)

    tree      = BallTree(event_rad, metric='haversine')
    cutoff_r  = 3 * bw_km / EARTH_R          # 3-sigma cutoff
    idx_l, dist_l = tree.query_radius(resp_rad, r=cutoff_r, return_distance=True)

    # Nearest event distance
    near_dist, _ = tree.query(resp_rad, k=1)
    near_km = near_dist.flatten() * EARTH_R

    exposure  = np.zeros(len(resp_df))
    for i, (idx, dist) in enumerate(zip(idx_l, dist_l)):
        if len(idx) == 0:
            continue
        d_km = dist * EARTH_R
        w    = np.exp(-0.5 * (d_km / bw_km) ** 2)   # Gaussian kernel
        exposure[i] = (w * fat[idx]).sum()

    return exposure, near_km, np.array([len(i) for i in idx_l], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  LOAD & PREPARE EACH ROUND
# ─────────────────────────────────────────────────────────────────────────────
frames = []

for rnd, meta in ROUNDS.items():
    print(f"\nProcessing Round {rnd} ...")
    df = pd.read_excel(f'{DATA_DIR}/{meta["file"]}')

    # ── survey date ──
    if meta['survey_mid'] is not None:
        df['survey_date'] = meta['survey_mid']
    else:
        df['survey_date'] = pd.to_datetime(df['dateintr'], errors='coerce')
        median_date = df['survey_date'].dropna().median()
        df['survey_date'] = df['survey_date'].fillna(median_date)

    # ── outcome: healthcare deprivation ──
    hcol = meta['hvar']
    raw  = pd.to_numeric(df[hcol], errors='coerce')
    raw  = raw.replace([-1, 7, 8, 9], np.nan)
    df['hlth_dep_raw'] = raw
    # Normalise to 0–1
    df['hlth_dep'] = raw / meta['hscale']

    # ── binary outcome: any deprivation ──
    df['any_dep'] = (raw > 0).astype(float)
    df['any_dep'][raw.isna()] = np.nan

    # ── geography ──
    df['lat'] = pd.to_numeric(df['latitude'],  errors='coerce')
    df['lon'] = pd.to_numeric(df['longitude'], errors='coerce')

    # ── state FE ──
    sc = meta['state_col']
    df['state'] = (f'R{rnd}_' + df[sc].astype(str)) if sc in df.columns \
                  else f'R{rnd}_unknown'

    # ── urban/rural ──
    df['urban'] = (pd.to_numeric(df.get('urbrur', pd.Series(dtype=float)),
                                 errors='coerce') == 1).astype(float)

    # ── compute per-respondent ACLED window ──
    df_c = df.dropna(subset=['hlth_dep','lat','lon','survey_date']).copy()
    df_c = df_c[df_c['lat'].between(-90,90) & df_c['lon'].between(-180,180)]

    window_start = df_c['survey_date'] - pd.Timedelta(days=365)
    window_end   = df_c['survey_date']

    # Get the unique windows (survey dates cluster tightly)
    unique_dates = df_c['survey_date'].unique()
    exp_all  = np.zeros(len(df_c))
    exp_vac  = np.zeros(len(df_c))
    exp_boko = np.zeros(len(df_c))
    near_all = np.full(len(df_c), np.nan)
    n_events = np.zeros(len(df_c))

    for sd in unique_dates:
        mask = (df_c['survey_date'] == sd).values
        ws   = sd - pd.Timedelta(days=365)
        ev   = acled[(acled['event_date'] >= ws) & (acled['event_date'] <= sd)]
        ev_v = ev[ev['is_vac'] == 1]
        ev_b = ev[ev['is_boko'] == 1]
        sub  = df_c[mask]

        e, nd, nc        = kernel_exposure(sub, ev)
        exp_all[mask]    = e
        near_all[mask]   = nd
        n_events[mask]   = nc

        ev2, _, _        = kernel_exposure(sub, ev_v)
        exp_vac[mask]    = ev2
        eb, _, _         = kernel_exposure(sub, ev_b)
        exp_boko[mask]   = eb

    df_c['exp_all']    = exp_all
    df_c['exp_vac']    = exp_vac
    df_c['exp_boko']   = exp_boko
    df_c['near_km']    = near_all
    df_c['n_ev_50km']  = n_events
    df_c['log_exp']    = np.log1p(exp_all)
    df_c['log_exp_vac']= np.log1p(exp_vac)
    df_c['log_exp_boko']= np.log1p(exp_boko)
    df_c['round']      = rnd

    # Summarise
    acled_cnt = len(acled[(acled['event_date'] >=
                           df_c['survey_date'].min() - pd.Timedelta(days=365)) &
                           (acled['event_date'] <= df_c['survey_date'].max())])
    pct_any = (df_c['near_km'] < 50).mean() * 100
    print(f"  n={len(df_c):,}  |  ACLED events in window: ~{acled_cnt}  |  "
          f"%within 50km: {pct_any:.1f}%  |  "
          f"mean log-exposure: {df_c['log_exp'].mean():.3f}")

    frames.append(df_c[['round','state','urban','lat','lon','survey_date',
                         'hlth_dep','hlth_dep_raw','any_dep',
                         'exp_all','exp_vac','exp_boko',
                         'log_exp','log_exp_vac','log_exp_boko',
                         'near_km','n_ev_50km']])

pool = pd.concat(frames, ignore_index=True)
pool['round_str'] = 'R' + pool['round'].astype(str)
print(f"\nPooled dataset: {len(pool):,} respondents across {pool['round'].nunique()} rounds")
print(pool.groupby('round')[['hlth_dep','log_exp','near_km']].mean().round(3).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 5.  DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("DESCRIPTIVE STATISTICS BY ROUND")
print("="*65)
desc = pool.groupby('round').agg(
    N         = ('hlth_dep',  'count'),
    hlth_dep  = ('hlth_dep',  'mean'),
    any_dep   = ('any_dep',   'mean'),
    log_exp   = ('log_exp',   'mean'),
    near_km   = ('near_km',   'median'),
    pct_50km  = ('near_km',   lambda x: (x<50).mean()),
).round(3)
desc.columns = ['N','HealthDep(0-1)','%AnyDep','LogExp','MedianDist(km)','%Within50km']
print(desc.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 6.  POOLED OLS  (round FE + state FE, SE clustered at state)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("POOLED OLS — Healthcare deprivation (normalised 0–1)")
print("Round FE + State FE | SE clustered at state level")
print("="*65)

NEED = ['hlth_dep','log_exp','state','round_str','urban']

def pooled_ols(outcome, treatment, data, cluster='state'):
    sub = data.dropna(subset=[outcome, treatment, cluster, 'round_str', 'urban'])
    f   = f'{outcome} ~ {treatment} + C(round_str) + C({cluster}) + urban'
    res = smf.ols(f, data=sub).fit(cov_type='cluster',
                                   cov_kwds={'groups': sub[cluster]})
    return res, sub

def show(res, var, label, n_cl):
    c, se, p = res.params[var], res.bse[var], res.pvalues[var]
    stars = '***' if p<.01 else '**' if p<.05 else '*' if p<.1 else ''
    print(f"  {label:<35} β={c:+.5f}  SE={se:.5f}  p={p:.3f}{stars}"
          f"  N={int(res.nobs):,}  clust={n_cl}")

for tvar, label in [
    ('log_exp',      'Log kernel-exp (all events)'),
    ('log_exp_vac',  'Log kernel-exp (VAC only)'),
    ('log_exp_boko', 'Log kernel-exp (Boko Haram)'),
]:
    res, sub = pooled_ols('hlth_dep', tvar, pool)
    show(res, tvar, label, sub['state'].nunique())

# ─────────────────────────────────────────────────────────────────────────────
# 7.  BY-ROUND RESULTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("BY-ROUND: Log kernel-exposure (all events) → Healthcare dep.")
print("="*65)
for rnd in sorted(pool['round'].unique()):
    sub = pool[pool['round']==rnd].dropna(subset=['hlth_dep','log_exp','state','urban'])
    if sub['state'].nunique() < 3 or len(sub) < 100:
        continue
    res = smf.ols('hlth_dep ~ log_exp + C(state) + urban', data=sub).fit(
        cov_type='cluster', cov_kwds={'groups': sub['state']})
    show(res, 'log_exp', f'R{rnd}', sub['state'].nunique())

# ─────────────────────────────────────────────────────────────────────────────
# 8.  GEOGRAPHIC RDD
#     Running variable: log(distance to nearest event + 1)
#     Threshold: 50 km
#     Estimator: local linear regression with triangular kernel
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("GEOGRAPHIC RDD")
print("Running variable: distance to nearest ACLED event (km)")
print("Threshold: 50 km  |  Estimator: local linear (triangular kernel)")
print("="*65)

rdd_data = pool.dropna(subset=['hlth_dep','near_km','round_str','state','urban']).copy()
rdd_data = rdd_data[rdd_data['near_km'] < 500]   # drop implausible
rdd_data['Z'] = rdd_data['near_km'] - 50         # centred running variable
rdd_data['T'] = (rdd_data['near_km'] <= 50).astype(float)  # treated = within 50km

def local_linear_rdd(data, bw, outcome='hlth_dep', c_var='Z', treat='T'):
    """Triangular-kernel local linear RD estimator."""
    w = np.maximum(0, 1 - np.abs(data[c_var]) / bw)
    sub = data[w > 0].copy()
    sub['_w'] = w[w > 0]
    if len(sub) < 50 or sub[treat].nunique() < 2:
        return np.nan, np.nan, np.nan, 0
    f = f'{outcome} ~ {treat} + {c_var} + {treat}:{c_var} + C(round_str) + urban'
    try:
        res = smf.wls(f, data=sub, weights=sub['_w']).fit(
            cov_type='cluster', cov_kwds={'groups': sub['state']})
        c, se, p = res.params[treat], res.bse[treat], res.pvalues[treat]
        return c, se, p, int(res.nobs)
    except:
        return np.nan, np.nan, np.nan, 0

print("\n  Sensitivity to bandwidth (50-km threshold):")
for bw in [25, 40, 50, 75, 100]:
    c, se, p, n = local_linear_rdd(rdd_data, bw=bw)
    if np.isnan(c): continue
    stars = '***' if p<.01 else '**' if p<.05 else '*' if p<.1 else ''
    print(f"  BW={bw:>3}km  β={c:+.5f}  SE={se:.5f}  p={p:.3f}{stars}  N={n:,}")

# Different thresholds
print("\n  Sensitivity to distance threshold (BW=50km):")
for thresh in [25, 50, 75, 100]:
    rd = rdd_data.copy()
    rd['Z'] = rd['near_km'] - thresh
    rd['T'] = (rd['near_km'] <= thresh).astype(float)
    c, se, p, n = local_linear_rdd(rd, bw=50)
    if np.isnan(c): continue
    stars = '***' if p<.01 else '**' if p<.05 else '*' if p<.1 else ''
    print(f"  Thresh={thresh:>3}km  β={c:+.5f}  SE={se:.5f}  p={p:.3f}{stars}  N={n:,}")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  DISTANCE-DECAY PLOT
#     Estimate pooled OLS at 10 radii, plot coefficient + CI
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("DISTANCE-DECAY: Coefficient by radius (binary ≤r km treatment)")
print("="*65)

radii   = [10, 20, 30, 40, 50, 60, 75, 100, 150, 200]
coefs, ses, ps = [], [], []

for r in radii:
    pool[f'any_{r}'] = (pool['near_km'] <= r).astype(float)
    sub = pool.dropna(subset=['hlth_dep', f'any_{r}', 'state','round_str','urban'])
    if sub[f'any_{r}'].std() < 0.01:
        coefs.append(np.nan); ses.append(np.nan); ps.append(np.nan); continue
    try:
        res = smf.ols(f'hlth_dep ~ any_{r} + C(round_str) + C(state) + urban',
                      data=sub).fit(cov_type='cluster',
                                    cov_kwds={'groups': sub['state']})
        coefs.append(res.params[f'any_{r}'])
        ses.append(res.bse[f'any_{r}'])
        ps.append(res.pvalues[f'any_{r}'])
        stars = '***' if ps[-1]<.01 else '**' if ps[-1]<.05 else '*' if ps[-1]<.1 else ''
        print(f"  ≤{r:>3}km  β={coefs[-1]:+.5f}  SE={ses[-1]:.5f}  p={ps[-1]:.3f}{stars}")
    except:
        coefs.append(np.nan); ses.append(np.nan); ps.append(np.nan)

# ─────────────────────────────────────────────────────────────────────────────
# 10. BOKO HARAM ZONE ANALYSIS (Rounds 4–6)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("BOKO HARAM ZONE ANALYSIS — Rounds 4–6")
print("Treatment: Boko Haram kernel exposure (log)")
print("="*65)

boko_pool = pool[pool['round'].isin([4,5,6])].copy()
boko_pool = boko_pool.dropna(subset=['hlth_dep','log_exp_boko','state','urban'])

for rnd in [4,5,6]:
    sub = boko_pool[boko_pool['round']==rnd].copy()
    if sub['log_exp_boko'].std() < 0.01 or sub['state'].nunique() < 3:
        print(f"  R{rnd}: insufficient Boko Haram variation"); continue
    res = smf.ols('hlth_dep ~ log_exp_boko + C(state) + urban', data=sub).fit(
        cov_type='cluster', cov_kwds={'groups': sub['state']})
    c,se,p = res.params['log_exp_boko'], res.bse['log_exp_boko'], res.pvalues['log_exp_boko']
    stars = '***' if p<.01 else '**' if p<.05 else '*' if p<.1 else ''
    pct_exp = (sub['exp_boko']>0).mean()*100
    print(f"  R{rnd}: β={c:+.5f}  SE={se:.5f}  p={p:.3f}{stars}  "
          f"N={int(res.nobs):,}  (%exposed={pct_exp:.1f}%)")

# Pooled R4-R6 with round FE
res46, sub46 = pooled_ols('hlth_dep', 'log_exp_boko', boko_pool)
show(res46, 'log_exp_boko', 'Boko Haram (pooled R4–R6)', sub46['state'].nunique())

# ─────────────────────────────────────────────────────────────────────────────
# 11. FIGURES
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

# ── Panel A: Distance-decay coefficient plot ──────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
valid = [(r, c, s, p) for r, c, s, p in zip(radii, coefs, ses, ps)
         if not np.isnan(c)]
if valid:
    rv, cv, sv, pv = zip(*valid)
    rv, cv, sv = np.array(rv), np.array(cv), np.array(sv)
    ax1.axhline(0, color='grey', lw=0.8, ls='--')
    ax1.fill_between(rv, cv - 1.96*sv, cv + 1.96*sv, alpha=0.25, color='steelblue')
    ax1.plot(rv, cv, 'o-', color='steelblue', lw=2, ms=7)
    sig = np.array(pv) < 0.1
    ax1.scatter(np.array(rv)[sig], np.array(cv)[sig], color='steelblue', s=80, zorder=5)
    ax1.set_xlabel('Distance threshold (km)', fontsize=11)
    ax1.set_ylabel('β (binary treatment ≤ r km)', fontsize=11)
    ax1.set_title('Distance-decay: Effect of violence proximity on healthcare deprivation\n'
                  '(pooled R1–R6, round+state FE, 95% CI shaded)', fontsize=11)
    ax1.grid(True, alpha=0.3)

# ── Panel B: RDD plot — outcome vs distance to nearest event ──────────────
ax2 = fig.add_subplot(gs[1, 0])
rd_plot = rdd_data.copy()
rd_plot['dist_bin'] = pd.cut(rd_plot['near_km'], bins=np.arange(0, 301, 15))
means = rd_plot.groupby('dist_bin', observed=True)['hlth_dep'].mean()
mids  = [iv.mid for iv in means.index]
ax2.axvline(50, color='red', lw=1.5, ls='--', label='50km threshold')
ax2.scatter(mids, means.values, s=40, color='steelblue', zorder=5)
ax2.plot(mids, means.values, color='steelblue', lw=1.5, alpha=0.7)
ax2.set_xlabel('Distance to nearest ACLED event (km)', fontsize=10)
ax2.set_ylabel('Healthcare deprivation (mean, 0–1)', fontsize=10)
ax2.set_title('RDD: Outcome vs. distance to event\n(15-km bins, pooled R1–R6)', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# ── Panel C: Mean deprivation and log-exposure by round ───────────────────
ax3 = fig.add_subplot(gs[1, 1])
by_round = pool.groupby('round').agg(
    dep=('hlth_dep', 'mean'),
    exp=('log_exp', 'mean')
).reset_index()
ax3b = ax3.twinx()
ax3.bar(by_round['round'], by_round['dep'], color='steelblue', alpha=0.6,
        label='Healthcare dep.')
ax3b.plot(by_round['round'], by_round['exp'], 'o--', color='firebrick',
          lw=2, ms=8, label='Log violence exp.')
ax3.set_xlabel('Round', fontsize=10)
ax3.set_ylabel('Healthcare deprivation (mean)', fontsize=10, color='steelblue')
ax3b.set_ylabel('Log kernel-exposure (mean)', fontsize=10, color='firebrick')
ax3.set_title('Healthcare deprivation and violence\nexposure by round', fontsize=10)
ax3.set_xticks(by_round['round'])
lines1, labs1 = ax3.get_legend_handles_labels()
lines2, labs2 = ax3b.get_legend_handles_labels()
ax3.legend(lines1+lines2, labs1+labs2, fontsize=8, loc='upper left')
ax3.grid(True, alpha=0.3, axis='y')

plt.suptitle('Violence and Healthcare Deprivation in Nigeria (2000–2016)\n'
             'ACLED × Afrobarometer Spatial Analysis', fontsize=13, fontweight='bold')
plt.savefig(f'{DATA_DIR}/spatial_results.png', dpi=150, bbox_inches='tight')
print(f"\nFigure saved: {DATA_DIR}/spatial_results.png")
print("\nDone.")
