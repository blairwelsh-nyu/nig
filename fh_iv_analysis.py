"""
IV Analysis: instruments for farmer-herder conflict intensity
Two instruments constructed from existing data:
  Z1: Rainfall stability (1/CV_annual) × post
      -- stable-rainfall Guinea savanna zones (low CV) ↔ more FH conflict
      -- ecological basis: stable rainfall = productive mixed zone = farmers + herders
  Z2: Proximity to Boko Haram epicenter × post
      -- Maiduguri BH conflict → pushed Fulani pastoralists south/west into MB
      -- geographic basis: MB LGAs closer to NE received more displaced herders

Also: standard CHIRPS z_grow IV as comparison (known weak first stage).
"""
import warnings, numpy as np, pandas as pd, pyreadstat
from numpy.linalg import lstsq
from scipy.stats import t as tdist, f as fdist
warnings.filterwarnings('ignore')

S = '/tmp/claude-0/-home-user-nig/61b5f4c8-5e77-5920-95de-a0a04a66cb77/scratchpad'
MB = ['Benue','Plateau','Kaduna','Nasarawa','Kogi','Taraba','Niger','Kwara',
      'Borno','Adamawa','Bauchi','Gombe']

cp  = pd.read_csv(f'{S}/cluster_panel.csv'); cp = cp[cp['rain_grow'] > -100]
si  = cp[['v001','state']].drop_duplicates('v001')
lga = pd.read_csv(f'{S}/cluster_lga.csv')

def wdm(arr, cl):
    df = pd.DataFrame({'v': arr.astype(float), 'cl': cl})
    return arr.astype(float) - df.groupby('cl')['v'].transform('mean').values

def cr(y, X, cl):
    b = lstsq(X, y, rcond=None)[0]; resid = y - X @ b
    G = len(np.unique(cl)); k = X.shape[1]
    meat = np.zeros((k,k))
    for c in np.unique(cl):
        m = cl==c; meat += X[m].T @ np.outer(resid[m],resid[m]) @ X[m]
    bread = np.linalg.inv(X.T @ X + np.eye(k)*1e-10)
    V = bread @ meat @ bread * (G/(G-1)) * (len(y)-1)/(len(y)-k)
    se = np.sqrt(np.diag(V)); t = b/se; p = 2*tdist.sf(np.abs(t), df=G-1)
    return b, se, t, p

def st(p): return '***' if p<.001 else ('**' if p<.01 else ('*' if p<.05 else ('†' if p<.1 else '')))

# ── Construct LGA-level instruments ──────────────────────────────────────────
cp_lga = cp.merge(lga[['v001','lga_id']], on='v001', how='left')

# LGA centroids from cluster means
lga_centers = cp_lga.groupby('lga_id')[['lat','lon']].mean()

# Z1: Rainfall stability (1 / cv_annual)
lga_cv = (cp_lga.groupby('lga_id')[['mean_annual','sd_annual','mean_grow','sd_grow',
                                      'mean_dry','sd_dry','z_grow']]
           .mean())
lga_cv['cv_annual'] = lga_cv['sd_annual'] / (lga_cv['mean_annual'] + 1e-9)
# Keep sensible values (filter CHIRPS artefacts: mean_annual should be 200–4000mm)
valid = (lga_cv['mean_annual'] > 200) & (lga_cv['mean_annual'] < 4000) & \
        (lga_cv['cv_annual'] > 0.04) & (lga_cv['cv_annual'] < 0.30)
lga_cv = lga_cv[valid].copy()
lga_cv['stability'] = 1.0 / lga_cv['cv_annual']  # high stability = low CV = more conflict
lga_cv['z_stability'] = ((lga_cv['stability'] - lga_cv['stability'].mean()) /
                           lga_cv['stability'].std())  # standardise

# Z2: Proximity to Maiduguri (BH epicenter, 11.846°N 13.160°E)
MAID_LAT, MAID_LON = 11.846, 13.160
lga_centers['dist_deg'] = np.sqrt((lga_centers['lat'] - MAID_LAT)**2 +
                                   (lga_centers['lon'] - MAID_LON)**2)
lga_centers['dist_km']  = lga_centers['dist_deg'] * 111.0
lga_centers['prox_maid'] = 1.0 / (lga_centers['dist_km'] + 1.0)  # proximity (large = close)
lga_centers['z_prox'] = ((lga_centers['prox_maid'] - lga_centers['prox_maid'].mean()) /
                           lga_centers['prox_maid'].std())

# LGA post-conflict total (endogenous variable)
lga_post = cp_lga[cp_lga['year'].between(2009,2018)].groupby('lga_id')['fh_events'].sum()
thresh   = lga_post.quantile(0.67)
treat_df = pd.DataFrame({'lga_id': lga_post.index,
                         'treated': (lga_post >= thresh).astype(float),
                         'lga_fh':  lga_post.values}).reset_index(drop=True)

# ── Check first stages (cross-sectional: LGA-level) ──────────────────────────
print("="*68)
print("CROSS-SECTIONAL INSTRUMENT CORRELATIONS (LGA level, n=LGAs)")
print("="*68)
common = lga_post.index.intersection(lga_cv.index).intersection(lga_centers.index)
print(f"  Common LGAs: {len(common)}")
fh_c = lga_post.loc[common]
st_c = lga_cv.loc[common, 'z_stability']
px_c = lga_centers.loc[common, 'z_prox']

from scipy.stats import pearsonr
r1,p1 = pearsonr(st_c, fh_c);  print(f"  Stability × FH:   r={r1:+.3f}  p={p1:.4f} {st(p1)}")
r2,p2 = pearsonr(px_c, fh_c);  print(f"  Proximity × FH:   r={r2:+.3f}  p={p2:.4f} {st(p2)}")
r3,p3 = pearsonr(st_c, px_c);  print(f"  Stability × Prox: r={r3:+.3f}  p={p3:.4f}  (instrument correlation)")

# ── Build IR-level DiD dataset with instruments ───────────────────────────────
want_ir = ['v001','v133','v714','v312','v025','v190','v012',
           'm2a_1','m2b_1','m3a_1','m3b_1','m3n_1']

def load_ir(path, syr):
    avail = pyreadstat.read_dta(path, metadataonly=True)[1].column_names
    df, _ = pyreadstat.read_dta(path, apply_value_formats=False,
                                 usecols=[c for c in want_ir if c in avail])
    df['survey_yr'] = syr; return df

ir = pd.concat([
    load_ir(f'{S}/ir_2008/NGIR53DT/NGIR53FL.DTA', 2008),
    load_ir(f'{S}/ir_2013/NGIR6ADT/NGIR6AFL.DTA', 2013),
    load_ir(f'{S}/ir_2018/NGIR7BDT/NGIR7BFL.DTA', 2018),
], ignore_index=True)
ir = ir.merge(lga[['v001','lga_id']], on='v001', how='left').merge(si, on='v001', how='left')
ir = ir[ir['state'].isin(MB) & ir['lga_id'].notna()].copy()

ir = ir.merge(treat_df, on='lga_id', how='left')
ir['treated'] = ir['treated'].fillna(0)
ir['lga_fh']  = ir['lga_fh'].fillna(0)
ir = ir.merge(lga_cv[['z_stability']].reset_index(), on='lga_id', how='left')
ir = ir.merge(lga_centers[['z_prox']].reset_index(), on='lga_id', how='left')
ir['z_stability'] = ir['z_stability'].fillna(0)
ir['z_prox']      = ir['z_prox'].fillna(0)

ir['post']       = (ir['survey_yr'] >= 2013).astype(float)
ir['post_treat'] = ir['post'] * ir['treated']
ir['post_logfh'] = ir['post'] * np.log1p(ir['lga_fh'])  # endogenous: conflict intensity

# Instruments
ir['z1_post'] = ir['post'] * ir['z_stability']  # stability × post
ir['z2_post'] = ir['post'] * ir['z_prox']        # proximity × post

yr_dum = pd.get_dummies(ir['survey_yr'], prefix='yd', drop_first=True)
for c in yr_dum.columns: ir[c] = yr_dum[c].astype(float).values
yr_cols = list(yr_dum.columns)

def any_col(*cols, df):
    r = pd.Series(False, index=df.index)
    for c in cols:
        if c in df.columns: r = r | (df[c]==1)
    return r.astype(float)

ir['skilled_del']   = any_col('m3a_1','m3b_1', df=ir)
ir['any_anc']       = any_col('m2a_1','m2b_1', df=ir)
ir['modern_contra'] = ir['v312'].isin([1,2,3,5,6,7]).astype(float)
ir['no_attend']     = (ir['m3n_1']==1).astype(float) if 'm3n_1' in ir.columns else np.nan

print(f"\nIR sample: {len(ir):,} women  |  {ir['lga_id'].nunique()} LGAs")

# ── FIRST STAGES ──────────────────────────────────────────────────────────────
print("\n" + "="*68)
print("FIRST STAGE: post_logfh ~ instrument × post  (LGA FE + year FE)")
print("Endogenous: post × log(1 + LGA total FH events 2009-2018)")
print("="*68)

def first_stage(ir, z_col, label):
    cl = ir['lga_id'].values
    x = wdm(ir['post_logfh'].values, cl)          # endogenous
    z = wdm(ir[z_col].values, cl)                  # instrument
    yr = np.column_stack([wdm(ir[c].astype(float).values, cl) for c in yr_cols])
    Z = np.column_stack([z, yr])
    b, se, t, p = cr(x, Z, cl)
    F = t[0]**2
    print(f"  {label:<45}  b={b[0]:+.4f}  se={se[0]:.4f}  p={p[0]:.4f} {st(p[0])}  F={F:.1f}")
    return b[0], F

print()
b_z1, F_z1 = first_stage(ir, 'z1_post', 'Z1 (stability × post)  →  post_logfh')
b_z2, F_z2 = first_stage(ir, 'z2_post', 'Z2 (proximity × post)  →  post_logfh')

# Combined (two instruments)
sub_all = ir.dropna(subset=['skilled_del']).copy()
cl = sub_all['lga_id'].values
x_end = wdm(sub_all['post_logfh'].values, cl)
z1 = wdm(sub_all['z1_post'].values, cl)
z2 = wdm(sub_all['z2_post'].values, cl)
yr = np.column_stack([wdm(sub_all[c].astype(float).values, cl) for c in yr_cols])
Z_both = np.column_stack([z1, z2, yr])
b_fs, se_fs, t_fs, p_fs = cr(x_end, Z_both, cl)
# Joint F statistic (Kleibergen-Paap style approximation: F for first 2 instruments)
F_joint = np.mean([t_fs[0]**2, t_fs[1]**2])  # simple average (not exact KP)
print(f"\n  Combined Z1+Z2 first stage:  F(Z1)={t_fs[0]**2:.1f}  F(Z2)={t_fs[1]**2:.1f}  F_avg={F_joint:.1f}")
print(f"  {'STRONG (F>10)' if F_joint>10 else 'MODERATE (F 5-10)' if F_joint>5 else 'WEAK (F<5) — IV unreliable'}")

# ── 2SLS ──────────────────────────────────────────────────────────────────────
print("\n" + "="*68)
print("2SLS: conflict intensity → health outcomes")
print("Instrument: Z1 (stability × post)  |  LGA FE + year FE")
print("="*68)
print(f"\n  {'Outcome':<22}  {'OLS post_logfh':>20}  {'2SLS Z1':>20}")
print("  " + "-"*68)

ocs = [('skilled_del','Skilled delivery'), ('any_anc','Any ANC'),
       ('modern_contra','Modern contra'),  ('no_attend','No attendant')]

for oc, lab in ocs:
    sub = ir.dropna(subset=[oc]).copy(); cl = sub['lga_id'].values
    y  = wdm(sub[oc].astype(float).values, cl)
    xr = wdm(sub['post_logfh'].values, cl)   # endogenous
    z1 = wdm(sub['z1_post'].values, cl)      # instrument
    yr = np.column_stack([wdm(sub[c].astype(float).values, cl) for c in yr_cols])

    # OLS
    X_ols = np.column_stack([xr, yr])
    b_ols, se_ols, _, p_ols = cr(y, X_ols, cl)

    # 2SLS: first stage
    Z_fs = np.column_stack([z1, yr])
    b1 = lstsq(Z_fs, xr, rcond=None)[0]
    xr_hat = Z_fs @ b1

    # Second stage
    X_ss = np.column_stack([xr_hat, yr])
    b2, se2, t2, p2 = cr(y, X_ss, cl)

    # FS F
    _, _, t_fs_s, _ = cr(xr, Z_fs, cl)
    F_s = t_fs_s[0]**2

    print(f"  {lab:<22}  OLS b={b_ols[0]:+.4f}(p={p_ols[0]:.3f}){st(p_ols[0]):<3}"
          f"  2SLS b={b2[0]:+.4f}(p={p2[0]:.3f}){st(p2[0]):<3}  FS F={F_s:.1f}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*68)
print("INSTRUMENT VALIDITY SUMMARY")
print("="*68)
print(f"  Z1 Rainfall stability:  FS b={b_z1:+.4f}  F={F_z1:.1f}")
print(f"  Z2 BH proximity:        FS b={b_z2:+.4f}  F={F_z2:.1f}")
print(f"  Instrument correlation: r={r3:+.3f} ({st(p3)})")
print(f"  Overidentification: Sargan test not computed (need exact VCV)")
print()
print("  Notes:")
print("  - Z1 (stability) assumes: stable-rainfall LGAs are structurally farmer-herder")
print("    transition zones; they experience more conflict when national FH violence escalates")
print("  - Z2 (BH proximity) assumes: LGAs closer to NE BH epicenter received displaced")
print("    Fulani pastoralists from ~2012 onward, increasing FH pressure in MB")
print("  - Exclusion: both instruments are predetermined geography, should not directly")
print("    affect women's healthcare 2013-2018 except via FH conflict channel")
print("    (conditional on LGA FE + year FE)")
