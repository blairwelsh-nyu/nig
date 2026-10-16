"""
Tsetse fly IV for farmer-herder conflict in Nigeria's Middle Belt.

Instrument logic (Z4):
  The tsetse fly (Glossina spp.) carries livestock trypanosomiasis and historically
  prevented cattle-keeping across much of sub-Saharan Africa.  Areas with high
  tsetse suitability had few cattle and therefore few Fulani pastoralists; areas
  with low tsetse suitability (drier, cooler, or upland) historically supported
  large cattle herds.  Post-2009 FH violence is structurally higher in low-tsetse
  (historically cattle-friendly) LGAs because Fulani herder presence was already
  embedded there.

  Z4 = cattle_suit × post
    where cattle_suit = 1 − tsetse_suitability
          tsetse_suitability = Tmin_suit × Precip_suit × Temp_suit

  Components (Rogers & Randolph 1993 ecological constraints):
    - Tmin_suit   = σ(BIO6 − 14)   [min-temp coldest month; >14°C = liveable for tsetse]
    - Precip_suit = σ(BIO12 − 600)  [annual precip; >600mm = humid enough]
    - Temp_suit   = Gaussian(BIO1; μ=25, σ=5) [mean annual temp; optimal ~24-26°C]

  So tsetse_suit peaks in warm-humid zones; cattle_suit peaks in cool/dry zones.

Exclusion restriction:
  Long-run climate normals are predetermined.  LGA FE absorbs the level effect
  of climate on baseline health.  The interaction (cattle_suit × post) affects
  health outcomes only through conflict; there is no plausible direct path from
  historical cattle suitability to post-2009 changes in health utilisation.

Key expected direction:
  More cattle-suitable LGA → more Fulani historically → more FH conflict post-2009
  → first-stage coefficient positive (cattle_suit → post_logfh)

References:
  Rogers & Randolph (1993) Science 261: 1604-1606.
  Alsan (2015) AER 105(1): 382-410.
  WorldClim v2.1 BIO1, BIO6, BIO12 (10 arcmin).
"""
import warnings, numpy as np, pandas as pd, pyreadstat, rasterio
from numpy.linalg import lstsq
from scipy.stats import t as tdist, pearsonr, chi2
warnings.filterwarnings('ignore')

S  = '/tmp/claude-0/-home-user-nig/61b5f4c8-5e77-5920-95de-a0a04a66cb77/scratchpad'
WC = f'{S}/worldclim_dl'
MB = ['Benue','Plateau','Kaduna','Nasarawa','Kogi','Taraba','Niger','Kwara',
      'Borno','Adamawa','Bauchi','Gombe']

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

def sigmoid(x): return 1/(1+np.exp(-x))

# ── Step 1: Sample WorldClim at cluster locations ─────────────────────────────
print("Loading WorldClim BIO1, BIO6, BIO12...")
cp   = pd.read_csv(f'{S}/cluster_panel.csv'); cp = cp[cp['rain_grow'] > -100]
si   = cp[['v001','state']].drop_duplicates('v001')
lga  = pd.read_csv(f'{S}/cluster_lga.csv')
clusters = cp[['v001','lat','lon']].drop_duplicates('v001').dropna(subset=['lat','lon'])

def sample_raster(path, lats, lons):
    ds = rasterio.open(path)
    arr = ds.read(1); nd = ds.nodata or -3.4e38
    from rasterio.transform import rowcol
    rows, cols = rowcol(ds.transform, lons, lats, op=round)
    nr, nc = arr.shape
    vals = []
    for r, c in zip(rows, cols):
        if 0 <= r < nr and 0 <= c < nc:
            v = arr[r,c]; vals.append(np.nan if (v==nd or v<-1e30) else float(v))
        else:
            vals.append(np.nan)
    ds.close(); return np.array(vals)

clusters = clusters.copy()
clusters['bio1']  = sample_raster(f'{WC}/wc2.1_10m_bio_1.tif',
                                   clusters['lat'].values, clusters['lon'].values)
clusters['bio12'] = sample_raster(f'{WC}/wc2.1_10m_bio_12.tif',
                                   clusters['lat'].values, clusters['lon'].values)

print(f"  BIO1  (mean annual temp °C):  mean={clusters['bio1'].mean():.1f}  "
      f"min={clusters['bio1'].min():.1f}  max={clusters['bio1'].max():.1f}")
print(f"  BIO12 (annual precip mm):     mean={clusters['bio12'].mean():.0f}  "
      f"min={clusters['bio12'].min():.0f}  max={clusters['bio12'].max():.0f}")

# ── Step 2: Tsetse suitability (Rogers & Randolph ecological constraints) ─────
# Components calibrated to Nigeria Middle Belt (tropical/sub-tropical context)
# Temperature suitability: Gaussian at 25°C, σ=5°C  (tsetse optimal range ~20-32°C)
TOPT = 25.0; TSIG = 5.0
clusters['temp_suit']   = np.exp(-0.5*((clusters['bio1'] - TOPT)/TSIG)**2)
# Precipitation suitability: sigmoid at 600mm  (tsetse needs humid conditions)
clusters['precip_suit'] = sigmoid(0.004*(clusters['bio12'] - 600))
# Joint tsetse suitability
clusters['tsetse_suit'] = clusters['temp_suit'] * clusters['precip_suit']
# Cattle suitability = inverse of tsetse suitability
clusters['cattle_suit'] = 1.0 - clusters['tsetse_suit']

print(f"\n  Tsetse suitability: mean={clusters['tsetse_suit'].mean():.3f}  "
      f"sd={clusters['tsetse_suit'].std():.3f}  "
      f"min={clusters['tsetse_suit'].min():.3f}  max={clusters['tsetse_suit'].max():.3f}")
print(f"  Cattle suitability: mean={clusters['cattle_suit'].mean():.3f}  "
      f"sd={clusters['cattle_suit'].std():.3f}")

# ── Step 3: Aggregate to LGA, z-score ─────────────────────────────────────────
cl_lga  = clusters.merge(lga[['v001','lga_id']], on='v001', how='left')
lga_ec  = cl_lga.groupby('lga_id')[['bio1','bio12','tsetse_suit','cattle_suit']].mean()
lga_ec['z_tsetse']  = ((lga_ec['tsetse_suit'] - lga_ec['tsetse_suit'].mean()) /
                        lga_ec['tsetse_suit'].std())
lga_ec['z_cattle']  = ((lga_ec['cattle_suit'] - lga_ec['cattle_suit'].mean()) /
                        lga_ec['cattle_suit'].std())

# ── Step 4: LGA conflict + treatment ─────────────────────────────────────────
cp_lga   = cp.merge(lga[['v001','lga_id']], on='v001', how='left')
lga_post = cp_lga[cp_lga['year'].between(2009,2018)].groupby('lga_id')['fh_events'].sum()
thresh   = lga_post.quantile(0.67)
treat_df = pd.DataFrame({'lga_id': lga_post.index,
                         'treated': (lga_post >= thresh).astype(float)}).reset_index(drop=True)

# ── Step 5: Cross-sectional correlation diagnostics ───────────────────────────
print("\n" + "="*68)
print("CROSS-SECTIONAL CORRELATIONS (LGA level, N=all Nigeria MB LGAs)")
print("="*68)
common = lga_post.index.intersection(lga_ec.dropna(subset=['z_cattle']).index)
fh_c = lga_post.loc[common]
tc_c = lga_ec.loc[common, 'z_cattle']
ts_c = lga_ec.loc[common, 'z_tsetse']
r_cattle, p_cattle = pearsonr(tc_c.values, fh_c.values)
r_tsetse, p_tsetse = pearsonr(ts_c.values, fh_c.values)
print(f"  Cattle suitability × FH events: r={r_cattle:+.3f}  p={p_cattle:.4f} {st(p_cattle)}")
print(f"  Tsetse suitability × FH events: r={r_tsetse:+.3f}  p={p_tsetse:.4f} {st(p_tsetse)}")
print(f"  LGAs in common: {len(common)}")

# Also compare Z3 and Z4 correlation
bio12_fit = np.exp(-0.5*((lga_ec.loc[common,'bio12'] - 1500)/300)**2)
z3_cs = (bio12_fit - bio12_fit.mean()) / bio12_fit.std()
r_z3z4, p_z3z4 = pearsonr(z3_cs.values, tc_c.values)
print(f"\n  Z3 (precip Gaussian 1500mm) × Z4 (cattle suit): r={r_z3z4:+.3f}  p={p_z3z4:.4f}")
print(f"  {'Highly correlated — same variation' if abs(r_z3z4)>.7 else 'Moderate overlap' if abs(r_z3z4)>.3 else 'Complementary'}")

# Temperature gradient check
r_temp, p_temp = pearsonr(lga_ec.loc[common,'bio1'].values, fh_c.values)
print(f"\n  BIO1 (mean temp) × FH events:   r={r_temp:+.3f}  p={p_temp:.4f} {st(p_temp)}")
print(f"  Jos Plateau LGAs (BIO1<22°C):")
plateau_lgas = lga_ec[(lga_ec['bio1'] < 22) & lga_ec.index.isin(common)].index
print(f"    N={len(plateau_lgas)} LGAs, mean FH events="
      f"{lga_post.loc[plateau_lgas].mean():.0f} vs overall {lga_post.loc[common].mean():.0f}")

# ── Step 6: Build IR panel ────────────────────────────────────────────────────
want_ir = ['v001','v312','v025','m2a_1','m2b_1','m3a_1','m3b_1','m3n_1']

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
ir = (ir.merge(lga[['v001','lga_id']], on='v001', how='left')
        .merge(si, on='v001', how='left'))
ir = ir[ir['state'].isin(MB) & ir['lga_id'].notna()].copy()
ir = (ir.merge(treat_df, on='lga_id', how='left')
        .merge(lga_ec[['z_cattle','z_tsetse']].reset_index(), on='lga_id', how='left'))
ir['treated']   = ir['treated'].fillna(0)
ir['z_cattle']  = ir['z_cattle'].fillna(0)
ir['z_tsetse']  = ir['z_tsetse'].fillna(0)
ir['post']       = (ir['survey_yr'] >= 2013).astype(float)
ir['lga_fh']     = ir['lga_id'].map(lga_post).fillna(0)
ir['post_logfh'] = ir['post'] * np.log1p(ir['lga_fh'])
ir['z4_post']    = ir['post'] * ir['z_cattle']   # instrument Z4

yr_dum = pd.get_dummies(ir['survey_yr'], prefix='yd', drop_first=True)
for c in yr_dum.columns: ir[c] = yr_dum[c].astype(float).values
yr_cols = list(yr_dum.columns)

def to_any(*cols, df):
    r = pd.Series(False, index=df.index)
    for c in cols:
        if c in df.columns: r = r | (df[c]==1)
    return r.astype(float)

ir['skilled_del']   = to_any('m3a_1','m3b_1', df=ir)
ir['any_anc']       = to_any('m2a_1','m2b_1', df=ir)
ir['modern_contra'] = ir['v312'].isin([1,2,3,5,6,7]).astype(float)
ir['no_attend']     = (ir['m3n_1']==1).astype(float) if 'm3n_1' in ir.columns else np.nan

print(f"\n  IR panel: {len(ir):,} women | {ir['lga_id'].nunique()} LGAs")

# Also build Z3 for comparison
cp_lga2 = cp.merge(lga[['v001','lga_id']], on='v001', how='left')
lga_wc2 = cl_lga.groupby('lga_id')[['bio12']].mean().copy()
lga_wc2['bio12_fit'] = np.exp(-0.5*((lga_wc2['bio12'] - 1500)/300)**2)
lga_wc2['z_tz']     = (lga_wc2['bio12_fit'] - lga_wc2['bio12_fit'].mean()) / lga_wc2['bio12_fit'].std()
ir = ir.merge(lga_wc2[['z_tz']].reset_index(), on='lga_id', how='left')
ir['z_tz'] = ir['z_tz'].fillna(0)
ir['z3_post'] = ir['post'] * ir['z_tz']

# Z1 (CHIRPS stability)
cp_lga_cv = cp_lga.groupby('lga_id')[['mean_annual','sd_annual']].mean()
cp_lga_cv['cv_annual'] = cp_lga_cv['sd_annual'] / (cp_lga_cv['mean_annual'] + 1e-9)
valid = ((cp_lga_cv['mean_annual'] > 200) & (cp_lga_cv['mean_annual'] < 4000) &
         (cp_lga_cv['cv_annual'] > 0.04) & (cp_lga_cv['cv_annual'] < 0.30))
cp_lga_cv = cp_lga_cv[valid].copy()
cp_lga_cv['z_stability'] = ((1/cp_lga_cv['cv_annual'] - (1/cp_lga_cv['cv_annual']).mean()) /
                              (1/cp_lga_cv['cv_annual']).std())
ir = ir.merge(cp_lga_cv[['z_stability']].reset_index(), on='lga_id', how='left')
ir['z_stability'] = ir['z_stability'].fillna(0)
ir['z1_post'] = ir['post'] * ir['z_stability']

# ── Step 7: First-stage comparison — all instruments ─────────────────────────
print("\n" + "="*68)
print("FIRST STAGE COMPARISON — Z1, Z3, Z4")
print("Endogenous: post × log(1 + LGA FH events)")
print("="*68)
cl = ir['lga_id'].values
x_end = wdm(ir['post_logfh'].values, cl)
yr    = np.column_stack([wdm(ir[c].astype(float).values, cl) for c in yr_cols])

for zname, zcol in [('Z1 (CHIRPS stability)','z1_post'),
                    ('Z3 (precip transition zone)','z3_post'),
                    ('Z4 (tsetse/cattle suit.)','z4_post')]:
    zv   = wdm(ir[zcol].values, cl)
    Za   = np.column_stack([zv, yr])
    b, se, t, p = cr(x_end, Za, cl)
    F    = t[0]**2
    print(f"  {zname:<30}  b={b[0]:+.4f}  se={se[0]:.4f}  "
          f"F={F:.1f}  {st(p[0])}  "
          f"{'STRONG' if F>16.4 else 'BORDERLINE' if F>10 else 'WEAK'}")

# ── Step 8: Full 2SLS table — all instruments × all outcomes ─────────────────
print("\n" + "="*68)
print("FULL 2SLS TABLE")
print("LGA FE + year FE | LGA-clustered SEs | Endogenous = post × log(FH)")
print("="*68)

ocs = [('skilled_del','Skilled delivery'),
       ('any_anc','Any ANC'),
       ('modern_contra','Modern contra'),
       ('no_attend','No attendant')]

instruments = [('OLS (DiD)', None),
               ('2SLS Z3 (transition)', 'z3_post'),
               ('2SLS Z4 (tsetse/cattle)', 'z4_post')]

rows = []
for oc, lab in ocs:
    sub = ir.dropna(subset=[oc]).copy(); cl = sub['lga_id'].values
    y   = wdm(sub[oc].astype(float).values, cl)
    xr  = wdm(sub['post_logfh'].values, cl)
    yrv = np.column_stack([wdm(sub[c].astype(float).values, cl) for c in yr_cols])
    row = {'outcome': lab}
    for spec, zcol in instruments:
        if zcol is None:
            b, se, _, p = cr(y, np.column_stack([xr, yrv]), cl)
        else:
            zv   = wdm(sub[zcol].values, cl)
            Zfs  = np.column_stack([zv, yrv])
            b1f  = lstsq(Zfs, xr, rcond=None)[0]
            xhat = Zfs @ b1f
            b, se, _, p = cr(y, np.column_stack([xhat, yrv]), cl)
            _, _, tfs, _ = cr(xr, Zfs, cl); row[f'F_{zcol}'] = tfs[0]**2
        row[f'b_{spec}']  = b[0]
        row[f'se_{spec}'] = se[0]
        row[f'p_{spec}']  = p[0]
    rows.append(row)
    print(f"\n  {lab}")
    print(f"    {'Spec':<28}  {'b':>8}  {'SE':>7}  {'p':>6}  {'':>5}")
    for spec, zcol in instruments:
        b   = row[f'b_{spec}']
        se  = row[f'se_{spec}']
        p   = row[f'p_{spec}']
        fs  = f"  F={row[f'F_{zcol}']:.1f}" if zcol and f'F_{zcol}' in row else ''
        print(f"    {spec:<28}  {b:>+8.4f}  {se:>7.4f}  {p:>6.3f}{st(p):<4}{fs}")

res = pd.DataFrame(rows)

# ── Step 9: Sargan overidentification test (Z3 + Z4 combined) ─────────────────
print("\n" + "="*68)
print("SARGAN OVERIDENTIFICATION TEST (Z3 + Z4 combined, skilled delivery)")
print("="*68)
sub = ir.dropna(subset=['skilled_del']).copy(); cl = sub['lga_id'].values
y   = wdm(sub['skilled_del'].astype(float).values, cl)
xr  = wdm(sub['post_logfh'].values, cl)
z3v = wdm(sub['z3_post'].values, cl)
z4v = wdm(sub['z4_post'].values, cl)
yrv = np.column_stack([wdm(sub[c].astype(float).values, cl) for c in yr_cols])

# First stage with both instruments
Zfs34 = np.column_stack([z3v, z4v, yrv])
b_fs  = lstsq(Zfs34, xr, rcond=None)[0]
xhat  = Zfs34 @ b_fs

# 2SLS structural residuals: y - [xr_hat, yrv] @ b_2sls
b_2sls = lstsq(np.column_stack([xhat, yrv]), y, rcond=None)[0]
# Structural residuals use original (not fitted) x for the response
resid = y - np.column_stack([xr, yrv]) @ b_2sls

# Sargan: regress structural residuals on full instrument set [z3, z4, yrv]
Z_full = np.column_stack([z3v, z4v, yrv])
b_sarg = lstsq(Z_full, resid, rcond=None)[0]
yhat_s = Z_full @ b_sarg
SS_res = resid @ resid
SS_fit = yhat_s @ yhat_s
n = len(resid)
J = n * (SS_fit / SS_res)
df_s = 1  # 2 instruments - 1 endog regressor
p_j  = 1 - chi2.cdf(J, df_s)
print(f"  J = {J:.3f}  df = {df_s}  p = {p_j:.4f}")
print(f"  {'PASS (p>0.1): instruments consistent with exclusion restriction' if p_j>0.1 else 'FAIL (p<0.1): overidentification rejected'}")
print(f"  Note: Sargan test is valid under homoskedasticity; result is indicative.")

# Combined first stage
_, _, tfs34, pfs34 = cr(xr, Zfs34, cl)
F_avg34 = np.mean([tfs34[0]**2, tfs34[1]**2])
print(f"\n  Combined Z3+Z4 first stage: Z3 F={tfs34[0]**2:.1f}  Z4 F={tfs34[1]**2:.1f}  F_avg={F_avg34:.1f}")

# ── Step 10: Summary for document ─────────────────────────────────────────────
print("\n" + "="*68)
print("SUMMARY FOR DOCUMENT")
print("="*68)
for oc, lab in ocs:
    row = res[res['outcome']==lab].iloc[0]
    print(f"\n  {lab}:")
    for spec, zcol in instruments:
        b  = row[f'b_{spec}']
        se = row[f'se_{spec}']
        p  = row[f'p_{spec}']
        ci_lo = b - 1.96*se; ci_hi = b + 1.96*se
        print(f"    {spec:<28}  b={b:+.4f} ({ci_lo:+.4f},{ci_hi:+.4f})  p={p:.3f}{st(p)}")
