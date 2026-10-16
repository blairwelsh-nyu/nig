"""
Robustness checks for the farmer-herder conflict / healthcare paper.

Runs:
 1. Leave-one-state-out (jackknife by state, skilled delivery)
 2. Alternative treatment definitions (p50, p75, p90 cutoffs; continuous per capita)
 3. Controls for Boko Haram violence
 4. Randomization inference (500 permutations within state, skilled delivery)
 5. Conley spatial SEs (20 km, 50 km cutoffs, OLS)
 6. Ethnicity triple-DiD: Fulani vs farming communities (Tiv, Berom, Tarok, Igbo)
 7. Religion triple-DiD: Muslim vs Christian
 8. Barriers-to-care by conflict (v467a-d: permission, money, distance, alone)
 9. Delivery at public vs private facility
10. Placebo outcome: women's own education attainment
"""
import warnings, numpy as np, pandas as pd, pyreadstat
from numpy.linalg import lstsq
from scipy.stats import t as tdist
warnings.filterwarnings('ignore')

S  = '/tmp/claude-0/-home-user-nig/61b5f4c8-5e77-5920-95de-a0a04a66cb77/scratchpad'
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

# ── Load DHS + LGA data ───────────────────────────────────────────────────────
print("Loading data...")
want = ['v001','v005','v131','v130','v025','v149','v312',
        'v467a','v467b','v467c','v467d',
        'm2a_1','m2b_1','m3a_1','m3b_1','m3n_1','m15_1']

def load_ir(path, syr):
    avail = pyreadstat.read_dta(path, metadataonly=True)[1].column_names
    df, meta = pyreadstat.read_dta(path, apply_value_formats=False,
                                    usecols=[c for c in want if c in avail])
    df['survey_yr'] = syr
    return df, meta

ir08, meta08 = load_ir(f'{S}/ir_2008/NGIR53DT/NGIR53FL.DTA', 2008)
ir13, meta13 = load_ir(f'{S}/ir_2013/NGIR6ADT/NGIR6AFL.DTA', 2013)
ir18, meta18 = load_ir(f'{S}/ir_2018/NGIR7BDT/NGIR7BFL.DTA', 2018)
ir = pd.concat([ir08, ir13, ir18], ignore_index=True)

cp  = pd.read_csv(f'{S}/cluster_panel.csv'); cp = cp[cp['rain_grow'] > -100]
lga = pd.read_csv(f'{S}/cluster_lga.csv')
si  = cp[['v001','state']].drop_duplicates('v001')

ir = (ir.merge(lga[['v001','lga_id']], on='v001', how='left')
        .merge(si, on='v001', how='left'))
ir = ir[ir['state'].isin(MB) & ir['lga_id'].notna()].copy()

cp_lga = cp.merge(lga[['v001','lga_id']], on='v001', how='left')

# LGA conflict totals
lga_post = cp_lga[cp_lga['year'].between(2009,2018)].groupby('lga_id')['fh_events'].sum()
lga_bh   = cp_lga[cp_lga['year'].between(2009,2018)].groupby('lga_id')['bh_events'].sum()

# LGA population proxy (cluster count)
lga_n = cp_lga.groupby('lga_id')['v001'].nunique().rename('n_clusters')

# Build main panel
ir['post'] = (ir['survey_yr'] >= 2013).astype(float)
ir['lga_fh'] = ir['lga_id'].map(lga_post).fillna(0)
ir['lga_bh'] = ir['lga_id'].map(lga_bh).fillna(0)
ir['post_logfh'] = ir['post'] * np.log1p(ir['lga_fh'])
ir['post_logbh'] = ir['post'] * np.log1p(ir['lga_bh'])

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

# 2-wave binary treatment (p67 cutoff, 2008+2013+2018)
thresh67 = lga_post.quantile(0.67)
ir['treated'] = (ir['lga_id'].map(lga_post).fillna(0) >= thresh67).astype(float)
ir['post_treat'] = ir['post'] * ir['treated']

print(f"  Panel: {len(ir):,} women | {ir['lga_id'].nunique()} LGAs | {ir['state'].nunique()} states")

# Helper: OLS DiD with binary treatment
def did_b(sub, oc, cl_col='lga_id'):
    sub = sub.dropna(subset=[oc]).copy()
    cl = sub[cl_col].values
    y  = wdm(sub[oc].astype(float).values, cl)
    x  = wdm(sub['post_treat'].astype(float).values, cl)
    yr = np.column_stack([wdm(sub[c].astype(float).values, cl) for c in yr_cols])
    b, se, t, p = cr(y, np.column_stack([x, yr]), cl)
    return b[0], se[0], p[0]

# Helper: OLS DiD with continuous treatment
def did_c(sub, oc, xvar='post_logfh', cl_col='lga_id'):
    sub = sub.dropna(subset=[oc]).copy()
    cl = sub[cl_col].values
    y  = wdm(sub[oc].astype(float).values, cl)
    x  = wdm(sub[xvar].astype(float).values, cl)
    yr = np.column_stack([wdm(sub[c].astype(float).values, cl) for c in yr_cols])
    b, se, t, p = cr(y, np.column_stack([x, yr]), cl)
    return b[0], se[0], p[0]

print("  Outcome check:")
for oc in ['skilled_del','any_anc','modern_contra','no_attend']:
    b,se,p = did_b(ir, oc)
    print(f"    {oc:<16}  b={b:+.4f}  se={se:.4f}  p={p:.3f}{st(p)}")

# ── 1. LEAVE-ONE-STATE-OUT ────────────────────────────────────────────────────
print("\n" + "="*68)
print("1. LEAVE-ONE-STATE-OUT (skilled delivery, binary treatment)")
print("="*68)
loo_results = []
for st_ in sorted(ir['state'].unique()):
    sub = ir[ir['state'] != st_]
    b, se, p = did_b(sub, 'skilled_del')
    loo_results.append({'state': st_, 'b': b, 'se': se, 'p': p})
    print(f"  Drop {st_:<12}  N={len(sub):>6}  b={b:+.4f}  se={se:.4f}  {st(p)}")
loo_df = pd.DataFrame(loo_results)
print(f"\n  Range: [{loo_df['b'].min():+.4f}, {loo_df['b'].max():+.4f}]  "
      f"All significant: {(loo_df['p'] < 0.05).all()}")

# ── 2. ALTERNATIVE TREATMENT DEFINITIONS ─────────────────────────────────────
print("\n" + "="*68)
print("2. ALTERNATIVE TREATMENT DEFINITIONS (skilled delivery)")
print("="*68)
print(f"  {'Definition':<40}  {'b':>8}  {'se':>7}  {'N-LGA':>6}  {'sig':>5}")
print("  " + "-"*70)

# Binary at different cutoffs
for pctile in [50, 60, 67, 75, 80, 90]:
    thr = lga_post.quantile(pctile/100)
    ir[f'tr{pctile}'] = (ir['lga_id'].map(lga_post).fillna(0) >= thr).astype(float)
    ir[f'pt{pctile}'] = ir['post'] * ir[f'tr{pctile}']
    sub = ir.dropna(subset=['skilled_del']).copy(); cl = sub['lga_id'].values
    y  = wdm(sub['skilled_del'].astype(float).values, cl)
    x  = wdm(sub[f'pt{pctile}'].astype(float).values, cl)
    yr = np.column_stack([wdm(sub[c].astype(float).values, cl) for c in yr_cols])
    b, se, t, p = cr(y, np.column_stack([x, yr]), cl)
    n_lga   = int(ir.groupby('lga_id')[f'tr{pctile}'].first().sum())
    print(f"  Binary p{pctile:>2} ({n_lga:>3} treated LGAs)        "
          f"  {b[0]:>+8.4f}  {se[0]:>7.4f}  {n_lga:>6}  {st(p[0])}")

# Continuous: post × log(events)
b, se, p = did_c(ir, 'skilled_del', 'post_logfh')
print(f"  Continuous post×log(FH events)            {b:>+8.4f}  {se:>7.4f}       —  {st(p)}")

# Continuous: post × log(events+1) per cluster
lga_n_cl = cp_lga.groupby('lga_id')['v001'].nunique()
ir['lga_fh_pc'] = ir['lga_id'].map(lga_post / (lga_n_cl + 1)).fillna(0)
ir['post_logfh_pc'] = ir['post'] * np.log1p(ir['lga_fh_pc'])
b, se, p = did_c(ir, 'skilled_del', 'post_logfh_pc')
print(f"  Continuous post×log(FH/cluster)           {b:>+8.4f}  {se:>7.4f}       —  {st(p)}")

# ── 3. BH VIOLENCE CONTROL ────────────────────────────────────────────────────
print("\n" + "="*68)
print("3. BOKO HARAM VIOLENCE CONTROL (continuous, skilled delivery)")
print("="*68)
for oc, lab in [('skilled_del','Skilled delivery'),('any_anc','Any ANC'),
                ('modern_contra','Modern contra'),('no_attend','No attendant')]:
    sub = ir.dropna(subset=[oc]).copy(); cl = sub['lga_id'].values
    y  = wdm(sub[oc].astype(float).values, cl)
    xf = wdm(sub['post_logfh'].values, cl)
    xb = wdm(sub['post_logbh'].values, cl)
    yr = np.column_stack([wdm(sub[c].astype(float).values, cl) for c in yr_cols])
    b, se, t, p = cr(y, np.column_stack([xf, xb, yr]), cl)
    print(f"  {lab:<20}  FH b={b[0]:+.4f}{st(p[0]):<3}  BH b={b[1]:+.4f}{st(p[1]):<3}")

# ── 4. RANDOMIZATION INFERENCE ────────────────────────────────────────────────
print("\n" + "="*68)
print("4. RANDOMIZATION INFERENCE (500 permutations within state, skilled del.)")
print("="*68)
np.random.seed(42)
N_PERM = 500

# Observed statistic (binary treatment, skilled delivery)
obs_b, _, obs_p = did_b(ir, 'skilled_del')
print(f"  Observed b = {obs_b:+.4f}")

# LGA-level treatment assignment (one row per LGA)
lga_treat = ir[['lga_id','treated','state']].drop_duplicates('lga_id').copy()

perm_betas = []
for _ in range(N_PERM):
    # Permute treatment within state
    perm = lga_treat.copy()
    for st_ in perm['state'].unique():
        mask = perm['state'] == st_
        perm.loc[mask, 'treated'] = perm.loc[mask, 'treated'].sample(frac=1, random_state=None).values
    # Merge back and compute DiD
    sub = ir.drop(columns=['treated','post_treat']).merge(
        perm[['lga_id','treated']], on='lga_id', how='left')
    sub['post_treat'] = sub['post'] * sub['treated'].fillna(0)
    b, _, _ = did_b(sub, 'skilled_del')
    perm_betas.append(b)

perm_betas = np.array(perm_betas)
ri_p = np.mean(np.abs(perm_betas) >= np.abs(obs_b))
print(f"  RI p-value (two-tailed) = {ri_p:.4f} {st(ri_p)}")
print(f"  Permutation distribution: mean={perm_betas.mean():+.4f}  "
      f"sd={perm_betas.std():.4f}  p5={np.percentile(perm_betas,5):+.4f}")
print(f"  Rank: {np.sum(np.abs(perm_betas) >= np.abs(obs_b))} / {N_PERM} permutations "
      f"as extreme as observed")

# ── 5. CONLEY SPATIAL SEs ─────────────────────────────────────────────────────
print("\n" + "="*68)
print("5. CONLEY SPATIAL SEs (LGA centroids, skilled delivery, binary treatment)")
print("="*68)

# Compute LGA centroids
lga_cents = (cp.merge(lga[['v001','lga_id']], on='v001', how='left')
               .groupby('lga_id')[['lat','lon']].mean())

# Haversine distance (km) between two sets of lat/lon
def haversine_matrix(lats, lons):
    R = 6371
    lats_r = np.radians(lats)
    lons_r = np.radians(lons)
    dlat = lats_r[:,None] - lats_r[None,:]
    dlon = lons_r[:,None] - lons_r[None,:]
    a = np.sin(dlat/2)**2 + np.cos(lats_r[:,None])*np.cos(lats_r[None,:])*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(a,0,1)))

def conley_se(y, X, lga_ids, lga_coords, cutoff_km):
    b = lstsq(X, y, rcond=None)[0]; resid = y - X @ b
    k = X.shape[1]
    # Unique LGAs
    ulgas = np.unique(lga_ids)
    lats  = lga_coords.loc[ulgas, 'lat'].values
    lons  = lga_coords.loc[ulgas, 'lon'].values
    D     = haversine_matrix(lats, lons)
    W     = np.maximum(1 - D/cutoff_km, 0)  # Bartlett kernel

    meat = np.zeros((k,k))
    for i, li in enumerate(ulgas):
        mi = lga_ids==li
        Xi = X[mi]; ri = resid[mi]
        si = Xi.T @ ri  # score for LGA i
        for j, lj in enumerate(ulgas):
            if W[i,j] == 0: continue
            mj = lga_ids==lj
            Xj = X[mj]; rj = resid[mj]
            sj = Xj.T @ rj
            meat += W[i,j] * np.outer(si, sj)

    bread = np.linalg.inv(X.T @ X + np.eye(k)*1e-10)
    V = bread @ meat @ bread
    return np.sqrt(np.diag(V))

sub = ir.dropna(subset=['skilled_del']).copy()
cl  = sub['lga_id'].values
y   = wdm(sub['skilled_del'].astype(float).values, cl)
x   = wdm(sub['post_treat'].astype(float).values, cl)
yr  = np.column_stack([wdm(sub[c].astype(float).values, cl) for c in yr_cols])
X   = np.column_stack([x, yr])
b_ols = lstsq(X, y, rcond=None)[0]
b_cluster, se_cluster, _, p_cluster = cr(y, X, cl)

# Cluster SE (baseline)
t_cl = b_cluster[0]/se_cluster[0]; p_cl = 2*tdist.sf(abs(t_cl), df=len(np.unique(cl))-1)
print(f"  Cluster SE (baseline):    b={b_cluster[0]:+.4f}  se={se_cluster[0]:.4f}  p={p_cl:.3f}{st(p_cl)}")

lga_coords_sub = lga_cents.loc[lga_cents.index.isin(cl)].dropna()
for cutoff in [20, 50, 100]:
    try:
        se_c = conley_se(y, X, cl, lga_coords_sub, cutoff)
        t_c  = b_ols[0]/se_c[0]; p_c = 2*tdist.sf(abs(t_c), df=len(np.unique(cl))-1)
        print(f"  Conley {cutoff:>3} km:             b={b_ols[0]:+.4f}  se={se_c[0]:.4f}  p={p_c:.3f}{st(p_c)}")
    except Exception as e:
        print(f"  Conley {cutoff} km: Error — {e}")

# ── 6. ETHNICITY TRIPLE-DID ───────────────────────────────────────────────────
print("\n" + "="*68)
print("6. ETHNICITY TRIPLE-DiD: Fulani vs. farming communities")
print("="*68)

# Check ethnicity codes
print("\n  v131 value counts (top 20):")
vc = ir['v131'].value_counts().head(20)
print(vc.to_string())
# Check DHS metadata
try:
    vl18 = meta18.variable_value_labels.get('v131',{})
    if vl18:
        print("\n  v131 labels (2018):")
        for k,v in sorted(vl18.items())[:30]: print(f"    {k}: {v}")
except: pass

# Identify Fulani (code 4 in most Nigeria DHS; verify from metadata)
# Also identify farming communities: Tiv (11), Berom (15), Igbo (3)
fulani_codes   = [4]   # will refine after seeing labels
farming_codes  = [3, 11, 15, 16, 17]  # Igbo, Tiv, Berom, Tarok, Jukun
print(f"\n  Checking groups in Middle Belt sample:")
for code, name in [(4,'Fulani'),(3,'Igbo'),(11,'Tiv'),(15,'Berom'),(16,'Tarok')]:
    n = (ir['v131']==code).sum()
    print(f"    {name} (v131={code}): N={n}")

ir['is_fulani']  = ir['v131'].isin(fulani_codes).astype(float)
ir['is_farming'] = ir['v131'].isin(farming_codes).astype(float)
ir['is_muslim']  = (ir['v130'] == 2).astype(float)  # 2=Islam in DHS

print(f"\n  Fulani women in MB sample: N={(ir['is_fulani']==1).sum()}")
print(f"  Farming women in MB sample: N={(ir['is_farming']==1).sum()}")
print(f"  Muslim women: N={(ir['is_muslim']==1).sum()}")

# Triple DiD: post × treat × group (restrict to Fulani + farming)
eth_sub = ir[ir['is_fulani']+ir['is_farming'] > 0].copy()
print(f"\n  Ethnicity subsample: {len(eth_sub):,} women")

eth_sub['post_treat_farm'] = eth_sub['post_treat'] * eth_sub['is_farming']
eth_sub['post_farm']       = eth_sub['post'] * eth_sub['is_farming']
eth_sub['treat_farm']      = eth_sub['treated'] * eth_sub['is_farming']

# Add year dummies for farming interaction
for c in yr_cols:
    eth_sub[f'{c}_farm'] = eth_sub[c] * eth_sub['is_farming']
farm_yr = [f'{c}_farm' for c in yr_cols]

for oc, lab in [('skilled_del','Skilled delivery'), ('no_attend','No attendant'),
                ('any_anc','Any ANC')]:
    sub2 = eth_sub.dropna(subset=[oc]).copy(); cl2 = sub2['lga_id'].values
    y2   = wdm(sub2[oc].astype(float).values, cl2)
    xvars = (['post_treat','post_farm','treat_farm','post_treat_farm','is_farming'] +
             yr_cols + farm_yr)
    Xm = np.column_stack([wdm(sub2[v].astype(float).values, cl2) for v in xvars])
    b2, se2, t2, p2 = cr(y2, Xm, cl2)
    # post_treat = effect for Fulani (index 0)
    # post_treat_farm = differential for farming vs Fulani (index 3)
    b_ful = b2[0]; se_ful = se2[0]; p_ful = p2[0]
    b_dif = b2[3]; se_dif = se2[3]; p_dif = p2[3]
    b_far = b_ful + b_dif
    se_far = np.sqrt(se_ful**2 + se_dif**2)  # approximate (ignores covariance)
    p_far  = 2*tdist.sf(abs(b_far/se_far), df=len(np.unique(cl2))-1)
    print(f"\n  {lab}:")
    print(f"    Fulani effect:      b={b_ful:+.4f}  se={se_ful:.4f}  p={p_ful:.3f}{st(p_ful)}")
    print(f"    Farming - Fulani:   b={b_dif:+.4f}  se={se_dif:.4f}  p={p_dif:.3f}{st(p_dif)}")
    print(f"    Farming net effect: b={b_far:+.4f}  se≈{se_far:.4f}  p≈{p_far:.3f}{st(p_far)}")

# ── 7. RELIGION TRIPLE-DID ────────────────────────────────────────────────────
print("\n" + "="*68)
print("7. RELIGION TRIPLE-DiD: Muslim vs. Christian (skilled delivery)")
print("="*68)
print(f"  v130 value counts: {ir['v130'].value_counts().to_dict()}")
ir['post_treat_chr'] = ir['post_treat'] * (1 - ir['is_muslim'])
ir['post_chr']       = ir['post'] * (1 - ir['is_muslim'])
ir['treat_chr']      = ir['treated'] * (1 - ir['is_muslim'])
ir['is_chr']         = 1 - ir['is_muslim']
for c in yr_cols: ir[f'{c}_chr'] = ir[c] * ir['is_chr']
chr_yr = [f'{c}_chr' for c in yr_cols]

for oc, lab in [('skilled_del','Skilled delivery'), ('any_anc','Any ANC')]:
    sub2 = ir.dropna(subset=[oc]).copy(); cl2 = sub2['lga_id'].values
    y2   = wdm(sub2[oc].astype(float).values, cl2)
    xvars = (['post_treat','post_chr','treat_chr','post_treat_chr','is_chr'] +
             yr_cols + chr_yr)
    Xm   = np.column_stack([wdm(sub2[v].astype(float).values, cl2) for v in xvars])
    b2, se2, t2, p2 = cr(y2, Xm, cl2)
    b_mus = b2[0]; se_mus = se2[0]; p_mus = p2[0]
    b_dif = b2[3]; se_dif = se2[3]; p_dif = p2[3]
    b_chr = b_mus + b_dif; se_chr = np.sqrt(se_mus**2+se_dif**2)
    p_chr = 2*tdist.sf(abs(b_chr/se_chr), df=len(np.unique(cl2))-1)
    print(f"  {lab}:")
    print(f"    Muslim effect:      b={b_mus:+.4f}  se={se_mus:.4f}  p={p_mus:.3f}{st(p_mus)}")
    print(f"    Christian - Muslim: b={b_dif:+.4f}  se={se_dif:.4f}  p={p_dif:.3f}{st(p_dif)}")
    print(f"    Christian net:      b={b_chr:+.4f}  se≈{se_chr:.4f}  p≈{p_chr:.3f}{st(p_chr)}")

# ── 8. BARRIERS TO CARE ───────────────────────────────────────────────────────
print("\n" + "="*68)
print("8. BARRIERS TO CARE (v467a-d: permission, money, distance, alone)")
print("   Conflict effect on reporting each barrier as 'big problem'")
print("="*68)
barrier_map = {'v467a':'Permission needed','v467b':'Getting money',
               'v467c':'Distance to facility','v467d':'Not wanting to go alone'}
print(f"\n  Barrier variable value counts (2018 sample):")
for bv in barrier_map:
    if bv in ir.columns:
        vc2 = ir[bv].value_counts().to_dict()
        print(f"    {bv} {barrier_map[bv]}: {vc2}")

for bv, blabel in barrier_map.items():
    if bv not in ir.columns: continue
    # 1 = big problem (in most DHS coding)
    ir[f'{bv}_big'] = (ir[bv] == 1).astype(float)
    b, se, p = did_b(ir, f'{bv}_big')
    print(f"  {blabel:<26}  b={b:+.4f}  se={se:.4f}  p={p:.3f}{st(p)}")

# ── 9. PUBLIC vs PRIVATE FACILITY ────────────────────────────────────────────
print("\n" + "="*68)
print("9. PUBLIC vs PRIVATE DELIVERY FACILITY (m15_1)")
print("="*68)
if 'm15_1' in ir.columns:
    print(f"  m15_1 value counts (top 15): {ir['m15_1'].value_counts().head(15).to_dict()}")
    # Public facilities: typically codes 20-29 in DHS; private: 30-39; home: 11-13
    # Nigeria DHS coding may vary; check
    ir['delivery_public']  = ir['m15_1'].between(20,29).astype(float)
    ir['delivery_private'] = ir['m15_1'].between(30,39).astype(float)
    ir['delivery_home']    = ir['m15_1'].between(11,19).astype(float)
    for oc, lab in [('delivery_public','Public facility'),
                    ('delivery_private','Private facility'),
                    ('delivery_home','Home birth')]:
        b, se, p = did_b(ir, oc)
        mean_oc = ir[oc].mean()
        print(f"  {lab:<20}  mean={mean_oc:.3f}  b={b:+.4f}  se={se:.4f}  p={p:.3f}{st(p)}")

# ── 10. PLACEBO OUTCOME: EDUCATION ────────────────────────────────────────────
print("\n" + "="*68)
print("10. PLACEBO: Women's own education (v149, completed before treatment)")
print("    Should be unaffected — measures pre-2009 schooling decisions")
print("="*68)
if 'v149' in ir.columns:
    print(f"  v149 codes: {ir['v149'].value_counts().to_dict()}")
    # v149: 0=no educ, 1=incomplete primary, 2=complete primary, 3=incomplete secondary,
    #        4=complete secondary, 5=higher
    ir['educ_secondary'] = (ir['v149'] >= 3).astype(float)
    b, se, p = did_b(ir, 'educ_secondary')
    print(f"  Secondary+ education:  b={b:+.4f}  se={se:.4f}  p={p:.3f}{st(p)}")
    b, se, p = did_b(ir, 'educ_secondary')
    # For women likely educated before 2009: focus on older women (age proxy = survey_yr - birth cohort)
    # Use v149 directly for all women as placebo

print("\nDone.")
