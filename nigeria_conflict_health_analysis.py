"""
Nigeria Conflict-Health Analysis Pipeline
==========================================
Uses DHS Birth Recodes (2008, 2013, 2018), ACLED conflict data, and CHIRPS
rainfall rasters to investigate causal links between farmer-herder conflict
and child mortality via a rainfall IV design.

Design:
  - Unit: individual birth × cluster × birth-year
  - Instrument: CHIRPS growing-season rainfall anomaly at cluster in birth year
  - Endogenous: ACLED farmer-herder events within 50km of cluster in birth year
  - Outcome: infant mortality (IMR) / under-5 mortality
  - FEs: cluster + birth-year
  - Cluster-robust SEs

Requires (in scratchpad):
  cluster_panel.csv     — prebuilt from chirps_iv_pipeline.py
  br_*/NGBR*FL.DTA      — DHS Birth Recode files
"""

import warnings, numpy as np, pandas as pd, pyreadstat
from numpy.linalg import lstsq
from scipy.stats import t as tdist

warnings.filterwarnings('ignore')

SCRATCH = '/tmp/claude-0/-home-user-nig/61b5f4c8-5e77-5920-95de-a0a04a66cb77/scratchpad'

MB_STATES = ['Benue','Plateau','Kaduna','Nasarawa','Kogi','Taraba',
             'Niger','Kwara','Borno','Adamawa','Bauchi','Gombe']
NE_STATES = ['Borno','Yobe','Adamawa','Bauchi','Gombe','Taraba']


def within_demean(arr, cl_arr):
    df = pd.DataFrame({'v': arr.astype(float), 'cl': cl_arr})
    return arr.astype(float) - df.groupby('cl')['v'].transform('mean').values


def ols_cr(y_dm, X_dm, cl_arr, label=''):
    b = lstsq(X_dm, y_dm, rcond=None)[0]
    resid = y_dm - X_dm @ b
    G = len(np.unique(cl_arr))
    k = X_dm.shape[1]
    meat = np.zeros((k, k))
    for c in np.unique(cl_arr):
        m = cl_arr == c
        meat += X_dm[m].T @ np.outer(resid[m], resid[m]) @ X_dm[m]
    bread = np.linalg.inv(X_dm.T @ X_dm + np.eye(k) * 1e-10)
    V = bread @ meat @ bread * (G/(G-1)) * (len(y_dm)-1)/(len(y_dm)-k)
    se = np.sqrt(np.diag(V))
    t = b / se
    p = 2 * tdist.sf(np.abs(t), df=G-1)
    if label:
        print(f"\n  {label}")
        print(f"  N={len(y_dm)}, G={G}, b={b[0]:.5f}, SE={se[0]:.5f}, t={t[0]:.2f}, p={p[0]:.4f}, F={t[0]**2:.1f}")
    return b, se, t, p


def run_fe(df, y_col, x_col, yr_cols, cl_col, label=''):
    cl = df[cl_col].values
    y  = within_demean(df[y_col].astype(float).values, cl)
    x  = within_demean(df[x_col].values, cl)
    if yr_cols:
        yr = np.column_stack([within_demean(df[c].astype(float).values, cl) for c in yr_cols])
        X  = np.column_stack([x, yr])
    else:
        X  = x.reshape(-1, 1)
    return ols_cr(y, X, cl, label=label)


def run_tsls(df, y_col, x_col, z_col, yr_cols, cl_col, label=''):
    cl = df[cl_col].values
    y  = within_demean(df[y_col].astype(float).values, cl)
    x  = within_demean(df[x_col].values, cl)
    z  = within_demean(df[z_col].values, cl)
    if yr_cols:
        yr = np.column_stack([within_demean(df[c].astype(float).values, cl) for c in yr_cols])
        X1 = np.column_stack([z, yr])
    else:
        X1 = z.reshape(-1, 1)
    b1 = lstsq(X1, x, rcond=None)[0]
    x_hat = X1 @ b1
    _, se1, t1, p1 = ols_cr(x, X1, cl)
    X2 = np.column_stack([x_hat, yr]) if yr_cols else x_hat.reshape(-1, 1)
    b2, se2, t2, p2 = ols_cr(y, X2, cl)
    if label:
        print(f"\n  {label}")
        print(f"  N={len(df)}, G={len(np.unique(cl))}")
        print(f"  FS: b={b1[0]:.5f} SE={se1[0]:.5f} t={t1[0]:.2f} p={p1[0]:.4f} F={t1[0]**2:.1f}")
        print(f"  SS: b={b2[0]:.5f} SE={se2[0]:.5f} t={t2[0]:.2f} p={p2[0]:.4f}")
    return b1, b2, t1[0]**2


def add_yrfe(df):
    yd = pd.get_dummies(df['birth_year'], prefix='y', drop_first=True)
    for c in yd.columns:
        df[c] = yd[c].astype(float).values
    return df, list(yd.columns)


def load_br(path, syr):
    want = ['v001', 'b3', 'b5', 'b7', 'b8', 'b0']
    avail = pyreadstat.read_dta(path, metadataonly=True)[1].column_names
    df, _ = pyreadstat.read_dta(path, usecols=[c for c in want if c in avail])
    df['survey_yr'] = syr
    df['birth_year'] = ((df['b3'] - 1) // 12 + 1900).astype(int)
    if 'b0' in df:
        df = df[df['b0'].fillna(0) == 0]
    df['b7'] = df['b7'].fillna(999)
    df['b8'] = df['b8'].fillna(0)
    df['imr'] = ((df['b5'] == 0) & (df['b7'] < 12)).astype(float)
    df['u5']  = ((df['b5'] == 0) & (df['b7'] < 60)).astype(float)
    return df


def main():
    print("Loading cluster panel...")
    cpanel = pd.read_csv(f'{SCRATCH}/cluster_panel.csv')
    cpanel = cpanel[cpanel['rain_grow'] > -100]
    rain = cpanel[['v001','year','z_grow','fh_events','bh_events','fh_any','bh_any']].rename(
        columns={'year': 'birth_year', 'z_grow': 'z_grow_birth',
                 'fh_events': 'fh_birth', 'bh_events': 'bh_birth',
                 'fh_any': 'fh_any_birth', 'bh_any': 'bh_any_birth'})
    si = cpanel[['v001', 'state']].drop_duplicates('v001')

    print("Loading Birth Recode files...")
    br = pd.concat([
        load_br(f'{SCRATCH}/br_2008/NGBR53DT/NGBR53FL.DTA', 2008),
        load_br(f'{SCRATCH}/br_2013/NGBR6ADT/NGBR6AFL.DTA', 2013),
        load_br(f'{SCRATCH}/br_2018/NGBR7BDT/NGBR7BFL.DTA', 2018),
    ], ignore_index=True)

    # Censoring cutoff: birth_year <= survey_year - 2 for IMR, - 6 for U5
    br_imr = br[br['birth_year'].between(2005, 2018) &
                (br['birth_year'] <= br['survey_yr'] - 2)].copy()
    br_u5  = br[br['birth_year'].between(2005, 2018) &
                (br['birth_year'] <= br['survey_yr'] - 6)].copy()

    for bdf in [br_imr, br_u5]:
        bdf.sort_values(['v001', 'birth_year'], inplace=True)

    br_imr = br_imr.merge(rain, on=['v001', 'birth_year'], how='left').merge(si, on='v001', how='left')
    br_u5  = br_u5.merge(rain,  on=['v001', 'birth_year'], how='left').merge(si, on='v001', how='left')

    def sub(df, states):
        return df[df['state'].isin(states) & df['z_grow_birth'].notna()].copy()

    imr_mb = sub(br_imr, MB_STATES)
    imr_ne = sub(br_imr, NE_STATES)
    u5_mb  = sub(br_u5,  MB_STATES)

    for df in [imr_mb, imr_ne, u5_mb]:
        df['fh_log'] = np.log1p(df['fh_birth'])
        df['bh_log'] = np.log1p(df['bh_birth'])

    imr_mb, yr_imr_mb = add_yrfe(imr_mb)
    imr_ne, yr_imr_ne = add_yrfe(imr_ne)
    u5_mb,  yr_u5_mb  = add_yrfe(u5_mb)

    print(f"\nSamples: IMR MB={len(imr_mb):,}, IMR NE={len(imr_ne):,}, U5 MB={len(u5_mb):,}")

    print("\n" + "="*60 + "\nFIRST STAGE\n" + "="*60)
    run_fe(imr_mb, 'fh_log', 'z_grow_birth', [], 'v001',
           label='FH ~ z_grow (cluster FE only, MB IMR sample)')
    run_fe(imr_mb, 'fh_log', 'z_grow_birth', yr_imr_mb, 'v001',
           label='FH ~ z_grow (cluster+year FE, MB IMR sample)')
    run_fe(u5_mb,  'fh_log', 'z_grow_birth', yr_u5_mb, 'v001',
           label='FH ~ z_grow (cluster+year FE, MB U5 sample)')
    run_fe(imr_ne, 'bh_log', 'z_grow_birth', yr_imr_ne, 'v001',
           label='BH ~ z_grow (cluster+year FE, NE — falsification)')

    print("\n" + "="*60 + "\nREDUCED FORM\n" + "="*60)
    run_fe(imr_mb, 'imr', 'z_grow_birth', yr_imr_mb, 'v001',
           label='IMR ~ z_grow (cluster+year FE, MB)')
    run_fe(imr_ne, 'imr', 'z_grow_birth', yr_imr_ne, 'v001',
           label='IMR ~ z_grow (cluster+year FE, NE — falsification)')
    if len(u5_mb) > 500:
        run_fe(u5_mb, 'u5', 'z_grow_birth', yr_u5_mb, 'v001',
               label='U5 ~ z_grow (cluster+year FE, MB)')

    print("\n" + "="*60 + "\n2SLS\n" + "="*60)
    run_tsls(imr_mb, 'imr', 'fh_log', 'z_grow_birth', yr_imr_mb, 'v001',
             label='2SLS: FH → IMR (MB, FS F≈4.5)')
    if len(u5_mb) > 500:
        run_tsls(u5_mb, 'u5', 'fh_log', 'z_grow_birth', yr_u5_mb, 'v001',
                 label='2SLS: FH → U5 (MB, FS F≈12.5)')

    print("\n" + "="*60 + "\nOLS DESCRIPTIVE\n" + "="*60)
    run_fe(imr_mb, 'imr', 'fh_log', yr_imr_mb, 'v001',
           label='OLS: FH conflict → IMR (cluster+year FE, MB)')


if __name__ == '__main__':
    main()
