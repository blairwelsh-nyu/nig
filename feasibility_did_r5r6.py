"""
Feasibility assessment: R5–R6 DiD for kidnapping and general violence exposure.

Approach:
  Treatment 1: ACLED kidnapping (notes-based, lat/lon available)
  Treatment 2: ACLED general violence (all violent events, lat/lon available)
  Treatment 3: Niger Delta zone binary (Abia, Rivers, Edo, Delta, Bayelsa, Cross River, Akwa Ibom)

Outcomes tested: q8c (deprivation), plus all q-vars found in both rounds.

Design: Pool R5+R6. OLS: outcome ~ treat + treat*post + post + controls + state_FE
SE clustered at state level (via heteroskedasticity-robust / sandwich).
"""
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.neighbors import BallTree
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

os.chdir('/home/user/nig')

# ── Parameters ──────────────────────────────────────────────────────────────

BW_KM = 50          # Gaussian kernel bandwidth
RADIUS_KM = 150     # max search radius (3×BW)
EARTH_R = 6371.0

KIDNAP_RE = re.compile(
    r'kidnap|abduct|ransom|hostage|forced disappear', re.I
)

DELTA_STATES = {
    'Abia', 'Rivers', 'Edo', 'Delta', 'Bayelsa',
    'Cross River', 'Akwa Ibom', 'Imo', 'Anambra'
}

R5_SURVEY = (pd.Timestamp('2012-10-30'), pd.Timestamp('2013-01-19'))
R6_SURVEY = (pd.Timestamp('2014-06-01'), pd.Timestamp('2015-12-31'))  # wide window
LOOKBACK_DAYS = 365


# ── Helpers ──────────────────────────────────────────────────────────────────

def kernel_exp(resp_lat, resp_lon, evt_lat, evt_lon, bw_km):
    """Gaussian kernel-weighted exposure. Returns array length = len(resp)."""
    n = len(resp_lat)
    exp_arr = np.zeros(n)
    n_arr   = np.zeros(n, dtype=int)
    nearest = np.full(n, np.nan)

    resp_rad = np.radians(np.column_stack([resp_lat, resp_lon]))
    evt_rad  = np.radians(np.column_stack([evt_lat, evt_lon]))
    tree = BallTree(evt_rad, metric='haversine')

    r_thresh = (RADIUS_KM / EARTH_R)
    idx_list, dist_list = tree.query_radius(resp_rad, r=r_thresh,
                                             return_distance=True)
    for i in range(n):
        if len(idx_list[i]) == 0:
            continue
        d_km = dist_list[i] * EARTH_R
        w = np.exp(-0.5 * (d_km / bw_km) ** 2)
        exp_arr[i] = w.sum()
        n_arr[i] = len(d_km)
        nearest[i] = d_km.min()
    return exp_arr, n_arr, nearest


def assign_state(lat_arr, lon_arr, state_cents):
    cent_rad = np.radians(state_cents[['lat', 'lon']].values)
    tree = BallTree(cent_rad, metric='haversine')
    resp_rad = np.radians(np.column_stack([lat_arr, lon_arr]))
    _, idx = tree.query(resp_rad, k=1)
    return state_cents['state'].values[idx.flatten()]


# ── Load ACLED ────────────────────────────────────────────────────────────────

def load_acled():
    acled = pd.read_csv('nigeria_acled.csv')
    acled['event_date'] = pd.to_datetime(acled['event_date'], errors='coerce')
    acled = acled.dropna(subset=['latitude', 'longitude', 'event_date'])
    return acled


# ── Load survey round ─────────────────────────────────────────────────────────

def load_round(rnd):
    df = pd.read_excel(f'NIG_r{rnd}.csv.xlsx')
    df.columns = df.columns.str.lower().str.strip()

    # lat/lon
    lat_col = next((c for c in df.columns if 'lat' in c), None)
    lon_col = next((c for c in df.columns if 'lon' in c), None)
    if lat_col is None or lon_col is None:
        raise ValueError(f'No lat/lon in R{rnd}')
    df = df.rename(columns={lat_col: 'lat', lon_col: 'lon'})
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df = df.dropna(subset=['lat', 'lon'])

    # interview date
    if 'dateintr' in df.columns:
        df['idate'] = pd.to_datetime(df['dateintr'], errors='coerce')
        if rnd == 6:
            df['idate'] = pd.Timestamp('2015-08-15')
    else:
        df['idate'] = pd.Timestamp('2012-12-01') if rnd == 5 else pd.Timestamp('2015-08-15')

    df['round'] = rnd
    return df


# ── Build treatment exposures ─────────────────────────────────────────────────

def build_treatments(df, acled, rnd):
    # Events in look-back window before median survey date
    survey_mid = df['idate'].median()
    lookback_start = survey_mid - pd.Timedelta(days=LOOKBACK_DAYS)
    lookback_end = survey_mid

    acled_win = acled[
        (acled['event_date'] >= lookback_start) &
        (acled['event_date'] <= lookback_end)
    ]

    # Kidnapping events (notes-based)
    acled_knap = acled_win[
        acled_win['notes'].str.contains(KIDNAP_RE, na=False) |
        acled_win['sub_event_type'].str.contains('Abduction', na=False)
    ]

    # All violent events
    acled_viol = acled_win[acled_win['event_type'].isin([
        'Violence against civilians', 'Battles', 'Explosions/Remote violence'
    ])]

    lat = df['lat'].values
    lon = df['lon'].values

    if len(acled_knap) > 0:
        df['knap_exp'], df['knap_n'], df['knap_km'] = kernel_exp(
            lat, lon, acled_knap['latitude'].values, acled_knap['longitude'].values, BW_KM
        )
    else:
        df['knap_exp'] = 0.0
        df['knap_n'] = 0
        df['knap_km'] = np.nan

    if len(acled_viol) > 0:
        df['viol_exp'], df['viol_n'], df['viol_km'] = kernel_exp(
            lat, lon, acled_viol['latitude'].values, acled_viol['longitude'].values, BW_KM
        )
    else:
        df['viol_exp'] = 0.0
        df['viol_n'] = 0
        df['viol_km'] = np.nan

    print(f"\nR{rnd} treatment summary (survey mid={survey_mid.date()}, "
          f"lookback={lookback_start.date()} to {lookback_end.date()}):")
    print(f"  ACLED kidnap events in window: {len(acled_knap)}")
    print(f"  ACLED violent events in window: {len(acled_viol)}")
    print(f"  Resp with any knap_exp>0: {(df['knap_exp']>0).sum()} / {len(df)}")
    print(f"  Resp with any viol_exp>0: {(df['viol_exp']>0).sum()} / {len(df)}")
    print(f"  knap_exp p25/median/p75: "
          f"{df['knap_exp'].quantile(.25):.3f} / "
          f"{df['knap_exp'].quantile(.5):.3f} / "
          f"{df['knap_exp'].quantile(.75):.3f}")
    print(f"  viol_exp p25/median/p75: "
          f"{df['viol_exp'].quantile(.25):.3f} / "
          f"{df['viol_exp'].quantile(.5):.3f} / "
          f"{df['viol_exp'].quantile(.75):.3f}")

    return df


# ── Outcome audit ─────────────────────────────────────────────────────────────

CANDIDATE_OUTCOMES = {
    # ── CONFIRMED COMPARABLE: same question wording in R5 and R6 ─────────────
    # Deprivation: "How often gone without X" — 0=Never, 4=Always (0-4 scale both rounds)
    'q8a':  'depriv_food',
    'q8b':  'depriv_water',
    'q8c':  'depriv_medcare',       # healthcare access
    'q8d':  'depriv_cash',
    'q8e':  'depriv_fuel',
    # Crime/safety fear: same wording; R5 scale 0-3, R6 scale 0-4 (normalize to 0-1)
    'q10a': 'fear_crime_outside',   # "How often feared crime away from home"
    'q10b': 'fear_crime_home',      # "How often feared crime in your home"
    # Fear while walking: q88a same question stem in both (0-3; -1=N/A in R6, strip)
    'q88a': 'fear_unsafe_walking',  # "How often felt unsafe walking in neighborhood"
    # ── CONFIRMED COMPARABLE: q52a-e democratic accountability perceptions ─────
    # PDF-verified R5 question wording. Battery is standard longitudinal items
    # (R2/R5 sources), carried forward across rounds; scale 0-3 both rounds.
    # "In your opinion, how often, in this country: ..."
    'q52a': 'media_abuse_freedoms',      # "Does news media abuse its freedoms?" (R5)
    'q52b': 'party_competition_violent', # "Does party competition lead to violent conflict?" (R2)
    'q52c': 'pres_ignores_courts',       # "Does President ignore courts/laws?" (R2)
    'q52d': 'opp_parties_silenced',      # "Are opposition parties silenced by govt?" (R5)
    'q52e': 'pres_ignores_assembly',     # "Does President ignore national assembly?" (R5)
    # ── EXCLUDED (different questions across rounds): ─────────────────────────
    # q66a-e: R5=local govt perf, R6=national govt perf (different level+domain)
    # q67a-b: R5=service access difficulty, R6=local govt performance
    # q93a-b: R5=radio/TV listening freq, R6=civic engagement item
    # q41: R5=term limits opinion, R6=democracy satisfaction
    # q43: R5=democracy satisfaction, R6=civic duty to obey govt
    # q44: R5=democracy battery item, R6=tax payment norms
    # q65a-c: R5=national govt performance (1-4), R6=investment priority (categorical)
    # q59a-b: R5=trust in president/parliament, R6=MP/councilor listening frequency
    # q69a-c: R5=clinic access problems, R6=deprivation frequency items
    # q89a: R5=crime victim binary (0/1), R6=police response scale (1-5) INCOMPARABLE
}


def audit_outcomes(r5, r6):
    rows = []
    for var, label in CANDIDATE_OUTCOMES.items():
        r5_col = var if var in r5.columns else None
        r6_col = var if var in r6.columns else None
        r5_n = r5[r5_col].notna().sum() if r5_col else 0
        r6_n = r6[r6_col].notna().sum() if r6_col else 0
        r5_vals = sorted(r5[r5_col].dropna().unique())[:5] if r5_col else []
        r6_vals = sorted(r6[r6_col].dropna().unique())[:5] if r6_col else []

        # Check for negative/missing codes to exclude (strip < 0 and DK codes ≥ 7)
        def clean_range(col, df):
            if col is None:
                return (None, None)
            s = pd.to_numeric(df[col], errors='coerce')
            s = s[(s >= 0) & (s < 7)]
            return (s.min(), s.max())

        r5_range = clean_range(r5_col, r5)
        r6_range = clean_range(r6_col, r6)

        rows.append({
            'var': var, 'label': label,
            'in_r5': r5_col is not None,
            'in_r6': r6_col is not None,
            'r5_n': r5_n, 'r6_n': r6_n,
            'r5_min': r5_range[0], 'r5_max': r5_range[1],
            'r6_min': r6_range[0], 'r6_max': r6_range[1],
            'r5_vals': r5_vals,
            'r6_vals': r6_vals,
        })

    audit = pd.DataFrame(rows)
    both = audit[audit['in_r5'] & audit['in_r6'] & (audit['r5_n'] > 0) & (audit['r6_n'] > 0)]
    print(f"\n=== Outcome Audit: {len(both)} vars present in both rounds ===")
    print(both[['var','label','r5_n','r6_n','r5_min','r5_max','r6_min','r6_max']].to_string(index=False))
    return audit, both


# ── DiD regressions ───────────────────────────────────────────────────────────

def run_did(pooled, outcome_var, treat_var, label, state_col='state'):
    """Pool R5+R6. DiD: outcome ~ T*post + T + post + controls + state_FE."""
    df = pooled.copy()

    # Recode outcome: strip negative codes AND Afrobarometer DK/refused codes (≥7)
    df[outcome_var] = pd.to_numeric(df[outcome_var], errors='coerce')
    df.loc[df[outcome_var] < 0, outcome_var] = np.nan
    df.loc[df[outcome_var] >= 7, outcome_var] = np.nan

    df = df.dropna(subset=[outcome_var, treat_var, 'post', state_col])
    if len(df) < 100:
        return None

    # Normalize outcome to 0-1
    y_min = df[outcome_var].min()
    y_max = df[outcome_var].max()
    if y_max == y_min:
        return None
    df['y'] = (df[outcome_var] - y_min) / (y_max - y_min)

    # Standardize treatment
    t_std = df[treat_var].std()
    if t_std == 0:
        return None
    df['t'] = (df[treat_var] - df[treat_var].mean()) / t_std

    # Controls: urban
    ctrl = ''
    if 'urban' in df.columns:
        df['urb'] = pd.to_numeric(df['urban'], errors='coerce').fillna(0)
        ctrl = '+ urb'

    formula = f'y ~ t * post + post {ctrl} + C({state_col})'
    try:
        m = smf.ols(formula, data=df).fit(
            cov_type='cluster', cov_kwds={'groups': df[state_col]}
        )
    except Exception as e:
        return None

    coef = m.params.get('t:post', np.nan)
    se   = m.bse.get('t:post', np.nan)
    pval = m.pvalues.get('t:post', np.nan)
    ci_lo = coef - 1.96 * se
    ci_hi = coef + 1.96 * se

    coef_t = m.params.get('t', np.nan)
    pval_t = m.pvalues.get('t', np.nan)

    return {
        'outcome': outcome_var, 'label': label,
        'treat': treat_var,
        'n': len(df),
        'n_r5': (df['post'] == 0).sum(),
        'n_r6': (df['post'] == 1).sum(),
        'n_states': df[state_col].nunique(),
        'treat_mean': df[treat_var].mean(),
        'treat_pct_pos': (df[treat_var] > 0).mean() * 100,
        'did_coef': coef, 'did_se': se, 'did_pval': pval,
        'did_ci_lo': ci_lo, 'did_ci_hi': ci_hi,
        'main_treat_coef': coef_t, 'main_treat_pval': pval_t,
        'r2': m.rsquared,
    }


def stars(p):
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    acled = load_acled()
    r5 = load_round(5)
    r6 = load_round(6)

    # State assignment (via ACLED centroids)
    state_cents = acled.groupby('admin1')[['latitude', 'longitude']].mean()
    state_cents = state_cents.reset_index().rename(
        columns={'admin1': 'state', 'latitude': 'lat', 'longitude': 'lon'}
    )
    r5['state'] = assign_state(r5['lat'].values, r5['lon'].values, state_cents)
    r6['state'] = assign_state(r6['lat'].values, r6['lon'].values, state_cents)

    # Delta zone
    r5['delta'] = r5['state'].isin(DELTA_STATES).astype(float)
    r6['delta'] = r6['state'].isin(DELTA_STATES).astype(float)

    # Build spatial treatments
    r5 = build_treatments(r5, acled, 5)
    r6 = build_treatments(r6, acled, 6)

    # Urban column
    for df in [r5, r6]:
        for c in df.columns:
            if 'urban' in c.lower():
                df['urban'] = df[c]
                break

    # Pool
    r5['post'] = 0.0
    r6['post'] = 1.0
    pooled = pd.concat([r5, r6], ignore_index=True, sort=False)

    # Log exposure
    pooled['log_knap'] = np.log1p(pooled['knap_exp'])
    pooled['log_viol'] = np.log1p(pooled['viol_exp'])

    # Audit outcomes
    audit, both_vars = audit_outcomes(r5, r6)

    print("\n=== Treatment distributions (pooled R5+R6) ===")
    for tv in ['knap_exp', 'log_knap', 'viol_exp', 'log_viol', 'delta']:
        pct_pos = (pooled[tv] > 0).mean() * 100 if tv != 'delta' else pooled[tv].mean() * 100
        print(f"  {tv}: mean={pooled[tv].mean():.4f}, sd={pooled[tv].std():.4f}, "
              f"%>0={pct_pos:.1f}%")

    print("\n=== Delta zone state representation ===")
    for rnd, df in [(5, r5), (6, r6)]:
        d_n = df['delta'].sum()
        print(f"  R{rnd}: {int(d_n)}/{len(df)} respondents in Delta zone "
              f"({100*d_n/len(df):.1f}%)")
        print(f"    Delta states: {sorted(df[df['delta']==1]['state'].unique())}")

    # Run DiD for each outcome × treatment combination
    treat_vars = [
        ('knap_exp',  'ACLED kidnap (raw kernel)'),
        ('log_knap',  'ACLED kidnap (log kernel)'),
        ('viol_exp',  'ACLED violence (raw kernel)'),
        ('log_viol',  'ACLED violence (log kernel)'),
        ('delta',     'Niger Delta zone (binary)'),
    ]

    results = []
    feasible_outcomes = both_vars['var'].tolist()

    print(f"\nRunning {len(feasible_outcomes)} outcomes × {len(treat_vars)} treatments...")

    for tv, tv_label in treat_vars:
        for var in feasible_outcomes:
            label = CANDIDATE_OUTCOMES.get(var, var)
            r = run_did(pooled, var, tv, label)
            if r is not None:
                r['treat_label'] = tv_label
                results.append(r)

    res_df = pd.DataFrame(results)

    # ── Print results table ───────────────────────────────────────────────────
    print("\n" + "="*100)
    print("DiD RESULTS: outcome ~ treat*post (R5=pre, R6=post) | clustered SE by state")
    print("="*100)

    for tv, tv_label in treat_vars:
        sub = res_df[res_df['treat'] == tv].sort_values('did_pval')
        print(f"\n--- Treatment: {tv_label} ---")
        print(f"{'Outcome':<30} {'Label':<25} {'N':>6} {'β_DiD':>8} {'SE':>7} "
              f"{'p':>6} {'Stars':>5} {'β_T':>8} {'p_T':>6}")
        print("-" * 105)
        for _, row in sub.iterrows():
            print(f"{row['outcome']:<30} {row['label'][:24]:<25} {int(row['n']):>6} "
                  f"{row['did_coef']:>8.4f} {row['did_se']:>7.4f} "
                  f"{row['did_pval']:>6.3f} {stars(row['did_pval']):>5} "
                  f"{row['main_treat_coef']:>8.4f} {stars(row['main_treat_pval']):>6}")

    # ── Save results ──────────────────────────────────────────────────────────
    res_df.to_csv('did_r5r6_results.csv', index=False)
    print(f"\nSaved {len(res_df)} regression results to did_r5r6_results.csv")

    # ── Feasibility figure ────────────────────────────────────────────────────
    make_figure(res_df, pooled, r5, r6)

    return res_df, pooled


def make_figure(res_df, pooled, r5, r6):
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 3, hspace=0.45, wspace=0.35)
    fig.suptitle('R5–R6 DiD Feasibility: Kidnapping & Violence Exposure',
                 fontsize=13, fontweight='bold')

    # Panel A: treatment distribution R5 vs R6
    ax = fig.add_subplot(gs[0, 0])
    for lbl, df, c in [('R5', r5, '#2196F3'), ('R6', r6, '#F44336')]:
        knap = df['knap_exp']
        ax.hist(np.log1p(knap[knap > 0]), bins=20, alpha=0.6, color=c, label=lbl)
    ax.set_title('A. log(kidnap exposure) dist.\n(treated obs only)')
    ax.set_xlabel('log(1+kernel exposure)')
    ax.legend(fontsize=8)

    # Panel B: violence exposure distribution
    ax = fig.add_subplot(gs[0, 1])
    for lbl, df, c in [('R5', r5, '#2196F3'), ('R6', r6, '#F44336')]:
        viol = df['viol_exp']
        ax.hist(np.log1p(viol[viol > 0]), bins=20, alpha=0.6, color=c, label=lbl)
    ax.set_title('B. log(violence exposure) dist.\n(treated obs only)')
    ax.set_xlabel('log(1+kernel exposure)')
    ax.legend(fontsize=8)

    # Panel C: Delta zone composition
    ax = fig.add_subplot(gs[0, 2])
    delta_r5 = r5.groupby('state')['delta'].first().reset_index()
    delta_r5['rnd'] = 'R5'
    delta_r6 = r6.groupby('state')['delta'].first().reset_index()
    delta_r6['rnd'] = 'R6'
    for rnd_df, c, lbl in [(r5, '#2196F3', 'R5'), (r6, '#F44336', 'R6')]:
        state_n = rnd_df.groupby('state').size().reset_index(name='n')
        state_d = rnd_df.groupby('state')['delta'].first().reset_index()
        state_n = state_n.merge(state_d, on='state')
        delta_n = state_n[state_n['delta'] == 1]['n']
        other_n = state_n[state_n['delta'] == 0]['n']
        ax.bar([lbl + '\nDelta', lbl + '\nOther'],
               [delta_n.sum(), other_n.sum()], color=c, alpha=0.7)
    ax.set_title('C. Delta vs other states\n(respondents per round)')
    ax.set_ylabel('N respondents')

    # Panel D: DiD coefficients for kidnap treatment (log_knap)
    ax = fig.add_subplot(gs[1, :])
    sub = res_df[res_df['treat'] == 'log_knap'].sort_values('did_coef')
    if len(sub) > 0:
        y = range(len(sub))
        ax.barh(y, sub['did_coef'], xerr=1.96*sub['did_se'],
                color=['#F44336' if p < 0.1 else '#9E9E9E'
                       for p in sub['did_pval']],
                alpha=0.8, capsize=3)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_yticks(list(y))
        ax.set_yticklabels([f"{r['outcome']} ({r['label'][:20]})"
                            for _, r in sub.iterrows()], fontsize=7)
        ax.set_title('D. DiD β (log kidnap × post): red = p<0.10')
        ax.set_xlabel('DiD coefficient (normalized outcome scale)')

    # Panel E: DiD coefficients for violence treatment
    ax = fig.add_subplot(gs[2, :])
    sub2 = res_df[res_df['treat'] == 'log_viol'].sort_values('did_coef')
    if len(sub2) > 0:
        y = range(len(sub2))
        ax.barh(y, sub2['did_coef'], xerr=1.96*sub2['did_se'],
                color=['#F44336' if p < 0.1 else '#9E9E9E'
                       for p in sub2['did_pval']],
                alpha=0.8, capsize=3)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_yticks(list(y))
        ax.set_yticklabels([f"{r['outcome']} ({r['label'][:20]})"
                            for _, r in sub2.iterrows()], fontsize=7)
        ax.set_title('E. DiD β (log violence × post): red = p<0.10')
        ax.set_xlabel('DiD coefficient (normalized outcome scale)')

    plt.savefig('did_r5r6_feasibility.png', dpi=150, bbox_inches='tight')
    print("Saved did_r5r6_feasibility.png")
    plt.close()


if __name__ == '__main__':
    res_df, pooled = main()
