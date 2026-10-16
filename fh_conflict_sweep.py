"""
Comprehensive conflict-health sweep: Nigeria DHS x ACLED
Covers: dry-season IV first stage, OLS outcome sweep, LGA DiD, mm16 sibling violence
"""
import warnings, numpy as np, pandas as pd, pyreadstat
from numpy.linalg import lstsq
from scipy.stats import t as tdist

warnings.filterwarnings('ignore')
SCRATCH = '/tmp/claude-0/-home-user-nig/61b5f4c8-5e77-5920-95de-a0a04a66cb77/scratchpad'
MB = ['Benue','Plateau','Kaduna','Nasarawa','Kogi','Taraba','Niger','Kwara','Borno','Adamawa','Bauchi','Gombe']
NE = ['Borno','Yobe','Adamawa','Bauchi','Gombe','Taraba']

def within_demean(arr, cl_arr):
    df = pd.DataFrame({'v': arr.astype(float), 'cl': cl_arr})
    return arr.astype(float) - df.groupby('cl')['v'].transform('mean').values

def ols_cr(y_dm, X_dm, cl_arr, label='', sig=''):
    b = lstsq(X_dm, y_dm, rcond=None)[0]
    resid = y_dm - X_dm @ b
    G = len(np.unique(cl_arr)); k = X_dm.shape[1]
    meat = np.zeros((k, k))
    for c in np.unique(cl_arr):
        m = cl_arr == c
        meat += X_dm[m].T @ np.outer(resid[m], resid[m]) @ X_dm[m]
    bread = np.linalg.inv(X_dm.T @ X_dm + np.eye(k)*1e-10)
    V = bread @ meat @ bread * (G/(G-1)) * (len(y_dm)-1)/(len(y_dm)-k)
    se = np.sqrt(np.diag(V)); t = b/se; p = 2*tdist.sf(np.abs(t), df=G-1)
    if label:
        star = '***' if p[0]<0.01 else ('**' if p[0]<0.05 else ('*' if p[0]<0.10 else ''))
        print(f"  {label}")
        print(f"    N={len(y_dm)} G={G} b={b[0]:.5f} SE={se[0]:.5f} t={t[0]:.2f} p={p[0]:.4f} F={t[0]**2:.1f} {star}")
    return b, se, t, p

def run_fe(df, y_col, x_col, yr_cols, cl_col, label=''):
    cl = df[cl_col].values
    y  = within_demean(df[y_col].astype(float).values, cl)
    x  = within_demean(df[x_col].values, cl)
    if yr_cols:
        yr = np.column_stack([within_demean(df[c].astype(float).values, cl) for c in yr_cols])
        X  = np.column_stack([x, yr])
    else:
        X  = x.reshape(-1,1)
    return ols_cr(y, X, cl, label=label)

def add_yrfe(df, col='birth_year'):
    yd = pd.get_dummies(df[col], prefix='y', drop_first=True)
    for c in yd.columns: df[c] = yd[c].astype(float).values
    return df, list(yd.columns)

# ── Load cluster panel ────────────────────────────────────────────────────────
print("Loading cluster panel...")
cp = pd.read_csv(f'{SCRATCH}/cluster_panel.csv')
cp = cp[cp['rain_grow'] > -100]

# Compute z_dry properly
cp['z_dry'] = cp.groupby('v001')['rain_dry'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))

rain = cp[['v001','year','z_grow','z_dry','fh_events','bh_events','fh_any','bh_any']].rename(
    columns={'year':'birth_year','z_grow':'z_grow_birth','z_dry':'z_dry_birth',
             'fh_events':'fh_birth','bh_events':'bh_birth',
             'fh_any':'fh_any_birth','bh_any':'bh_any_birth'})
si = cp[['v001','state']].drop_duplicates('v001')

print(f"  Cluster-year rows: {len(cp)}, unique clusters: {cp['v001'].nunique()}")

# ── Load IR files ─────────────────────────────────────────────────────────────
print("\nLoading IR files...")

def load_ir(path, syr):
    want = ['v001','v005','v024','v133','v714','v743a','v744a','v744b','v744c',
            'v744d','v744e','v312','v850','v190','v025',
            'm2a_1','m2b_1','m3a_1','m3b_1','m3n_1']
    avail = pyreadstat.read_dta(path, metadataonly=True)[1].column_names
    df, _ = pyreadstat.read_dta(path, apply_value_formats=False,
                                 usecols=[c for c in want if c in avail])
    df['survey_yr'] = syr
    return df

ir08 = load_ir(f'{SCRATCH}/ir_2008/NGIR53DT/NGIR53FL.DTA', 2008)
ir13 = load_ir(f'{SCRATCH}/ir_2013/NGIR6ADT/NGIR6AFL.DTA', 2013)
ir18 = load_ir(f'{SCRATCH}/ir_2018/NGIR7BDT/NGIR7BFL.DTA', 2018)
ir = pd.concat([ir08, ir13, ir18], ignore_index=True)
print(f"  Pooled IR: {len(ir):,} women")

# Merge conflict at survey year (cross-section: match by cluster, survey_yr)
rain_xs = rain.rename(columns={'birth_year':'survey_yr'})
ir = ir.merge(rain_xs, on=['v001','survey_yr'], how='left').merge(si, on='v001', how='left')

# Build outcomes
def to_any(*cols, df):
    result = pd.Series(False, index=df.index)
    for c in cols:
        if c in df.columns:
            result = result | (df[c] == 1)
    return result.astype(float)

ir['skilled_del'] = to_any('m3a_1','m3b_1', df=ir)
ir['any_anc']     = to_any('m2a_1','m2b_1', df=ir)
ir['educ_yrs']    = ir['v133'].clip(0,25) if 'v133' in ir.columns else np.nan
ir['working']     = (ir['v714'] == 1).astype(float) if 'v714' in ir.columns else np.nan
ir['health_auto'] = (ir['v743a'] == 1).astype(float) if 'v743a' in ir.columns else np.nan
dv_cols = [c for c in ['v744a','v744b','v744c','v744d','v744e'] if c in ir.columns]
if dv_cols:
    ir['dv_justified'] = to_any(*dv_cols, df=ir)
ir['modern_contra'] = ir['v312'].isin([1,2,3,5,6,7]).astype(float) if 'v312' in ir.columns else np.nan
ir['no_attend']   = (ir['m3n_1'] == 1).astype(float) if 'm3n_1' in ir.columns else np.nan

ir['fh_log'] = np.log1p(ir['fh_birth'].fillna(0))
ir_mb = ir[ir['state'].isin(MB) & ir['z_grow_birth'].notna()].copy()
ir_mb, yr_ir = add_yrfe(ir_mb, 'survey_yr')

print(f"  MB IR sample: {len(ir_mb):,} women, {ir_mb['v001'].nunique()} clusters")

# ── PART 1: Dry-season IV first stage ─────────────────────────────────────────
print("\n" + "="*60)
print("PART 1: DRY-SEASON RAINFALL FIRST STAGE")
print("(z_dry = cluster-demeaned dry-season rainfall)")
print("="*60)

# Load BR for birth-year analysis
def load_br(path, syr):
    want = ['v001','b3','b5','b7','b8','b0']
    avail = pyreadstat.read_dta(path, metadataonly=True)[1].column_names
    df, _ = pyreadstat.read_dta(path, usecols=[c for c in want if c in avail])
    df['survey_yr'] = syr
    df['birth_year'] = ((df['b3']-1)//12+1900).astype(int)
    if 'b0' in df: df = df[df['b0'].fillna(0)==0]
    df['b7'] = df['b7'].fillna(999)
    df['imr'] = ((df['b5']==0) & (df['b7']<12)).astype(float)
    return df

br08 = load_br(f'{SCRATCH}/br_2008/NGBR53DT/NGBR53FL.DTA', 2008)
br13 = load_br(f'{SCRATCH}/br_2013/NGBR6ADT/NGBR6AFL.DTA', 2013)
br18 = load_br(f'{SCRATCH}/br_2018/NGBR7BDT/NGBR7BFL.DTA', 2018)
br = pd.concat([br08, br13, br18], ignore_index=True)
br_imr = br[br['birth_year'].between(2005,2018) & (br['birth_year'] <= br['survey_yr']-2)].copy()
br_imr = br_imr.merge(rain, on=['v001','birth_year'], how='left').merge(si, on='v001', how='left')
imr_mb = br_imr[br_imr['state'].isin(MB) & br_imr['z_grow_birth'].notna()].copy()
imr_mb['fh_log'] = np.log1p(imr_mb['fh_birth'])
imr_mb, yr_imr = add_yrfe(imr_mb)

print(f"\nIMR MB sample: {len(imr_mb):,} births, {imr_mb['v001'].nunique()} clusters")

run_fe(imr_mb, 'fh_log', 'z_grow_birth', yr_imr, 'v001',
       label='FH ~ z_grow (cluster+year FE, MB) [growing season]')
run_fe(imr_mb, 'fh_log', 'z_dry_birth', yr_imr, 'v001',
       label='FH ~ z_dry (cluster+year FE, MB) [dry season]')
run_fe(imr_mb, 'fh_log', 'z_dry_birth', [], 'v001',
       label='FH ~ z_dry (cluster FE only, MB) [dry season, no year FE]')

# ── PART 2: OLS outcome sweep ─────────────────────────────────────────────────
print("\n" + "="*60)
print("PART 2: OLS OUTCOME SWEEP (cluster+year FE, Middle Belt)")
print("Outcome ~ FH_log + year_FE + cluster_FE")
print("="*60)

outcomes = [
    ('skilled_del', 'Skilled delivery'),
    ('any_anc',     'Any ANC visit'),
    ('no_attend',   'No birth attendant'),
    ('educ_yrs',    'Years of education'),
    ('working',     'Currently working'),
    ('health_auto', 'Health decision autonomy'),
    ('dv_justified','DV justified (any)'),
    ('modern_contra','Modern contraception'),
]

results = []
for oc, label in outcomes:
    if oc not in ir_mb.columns: continue
    sub = ir_mb.dropna(subset=[oc,'fh_log']).copy()
    if len(sub) < 200: continue
    b, se, t, p = run_fe(sub, oc, 'fh_log', yr_ir, 'v001', label=f'{label} ({oc})')
    results.append({'outcome': label, 'b': b[0], 'se': se[0], 't': t[0], 'p': p[0], 'N': len(sub)})

# Also add IMR
b, se, t, p = run_fe(imr_mb, 'imr', 'fh_log', yr_imr, 'v001', label='IMR (birth-level)')
results.append({'outcome': 'IMR', 'b': b[0], 'se': se[0], 't': t[0], 'p': p[0], 'N': len(imr_mb)})

print("\n--- RANKED BY p-value ---")
df_res = pd.DataFrame(results).sort_values('p')
for _, r in df_res.iterrows():
    star = '***' if r['p']<0.01 else ('**' if r['p']<0.05 else ('*' if r['p']<0.10 else ''))
    print(f"  {r['outcome']:<30} b={r['b']:+.5f}  p={r['p']:.4f}  N={int(r['N']):,}  {star}")

# ── PART 3: LGA-level DiD ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("PART 3: LGA-LEVEL DID (High vs Low FH conflict LGAs)")
print("Treatment = top tercile FH events 2009-2018; Pre=2008, Post=2013+2018")
print("="*60)

lga = pd.read_csv(f'{SCRATCH}/cluster_lga.csv')
ir_lga = ir.merge(lga[['v001','lga_id']], on='v001', how='left')

# Build LGA-level conflict: sum FH events 2009-2018 within LGA
cp_lga = cp.merge(lga[['v001','lga_id']], on='v001', how='left')
lga_fh = cp_lga[cp_lga['year'].between(2009,2018)].groupby('lga_id')['fh_events'].sum().reset_index()
lga_fh.columns = ['lga_id','lga_fh_total']
lga_fh['treated'] = (lga_fh['lga_fh_total'] >= lga_fh['lga_fh_total'].quantile(0.67)).astype(float)

ir_lga = ir_lga.merge(lga_fh, on='lga_id', how='left')
ir_lga = ir_lga[ir_lga['state'].isin(MB) & ir_lga['lga_id'].notna()].copy()
ir_lga['post'] = (ir_lga['survey_yr'] >= 2013).astype(float)
ir_lga['post_treat'] = ir_lga['post'] * ir_lga['treated']

# For DiD we need LGA FE + year FE: demean by LGA rather than cluster
print(f"\nLGA DiD sample: {len(ir_lga):,} women, {ir_lga['lga_id'].nunique()} LGAs")
print(f"  Treated LGAs: {ir_lga[ir_lga['treated']==1]['lga_id'].nunique()}, Control: {ir_lga[ir_lga['treated']==0]['lga_id'].nunique()}")

# OLS with post × treat interaction (absorb LGA + year FE manually)
yr_did = pd.get_dummies(ir_lga['survey_yr'], prefix='yd', drop_first=True)
for c in yr_did.columns: ir_lga[c] = yr_did[c].astype(float).values
yr_did_cols = list(yr_did.columns)

print("\nDiD results (LGA-robust SEs):")
for oc, label in outcomes:
    if oc not in ir_lga.columns: continue
    sub = ir_lga.dropna(subset=[oc,'post_treat']).copy()
    if len(sub) < 200: continue
    cl_lga = sub['lga_id'].values
    y = within_demean(sub[oc].astype(float).values, cl_lga)
    # post_treat is the DiD term; also include post (absorbed if no treated variation across time)
    x = within_demean(sub['post_treat'].values, cl_lga)
    yr = np.column_stack([within_demean(sub[c].astype(float).values, cl_lga) for c in yr_did_cols])
    X = np.column_stack([x, yr])
    b, se, t, p = ols_cr(y, X, cl_lga)
    star = '***' if p[0]<0.01 else ('**' if p[0]<0.05 else ('*' if p[0]<0.10 else ''))
    print(f"  {label:<30} b={b[0]:+.5f}  SE={se[0]:.5f}  t={t[0]:.2f}  p={p[0]:.4f}  N={len(sub):,}  {star}")

# ── PART 4: mm16 Sibling death by violence ────────────────────────────────────
print("\n" + "="*60)
print("PART 4: MATERNAL MORTALITY MODULE — Sibling death by violence")
print("Direct conflict mortality via individual record links")
print("="*60)

def load_ir_mm(path, syr):
    # mm cols: mm1_* (sex), mm2_* (alive), mm8_* (age at death), mm15_* (year of death), mm16_* (violence/accident)
    avail = pyreadstat.read_dta(path, metadataonly=True)[1].column_names
    mm_cols = [c for c in avail if c.startswith('mm') and '_' in c]
    want = ['v001','v005'] + mm_cols[:80]  # limit to avoid huge read
    df, _ = pyreadstat.read_dta(path, apply_value_formats=False,
                                 usecols=[c for c in want if c in avail])
    df['survey_yr'] = syr
    return df

print("\nLoading IR files with MM module...")
try:
    ir08m = load_ir_mm(f'{SCRATCH}/ir_2008/NGIR53DT/NGIR53FL.DTA', 2008)
    ir13m = load_ir_mm(f'{SCRATCH}/ir_2013/NGIR6ADT/NGIR6AFL.DTA', 2013)
    ir18m = load_ir_mm(f'{SCRATCH}/ir_2018/NGIR7BDT/NGIR7BFL.DTA', 2018)

    # Reshape to sibling-level: wide → long for mm15 (year of death) and mm16 (cause)
    def reshape_mm(df, syr):
        records = []
        # Find number of sibling slots
        mm15_cols = sorted([c for c in df.columns if c.startswith('mm15_')])
        mm16_cols = sorted([c for c in df.columns if c.startswith('mm16_')])
        mm2_cols  = sorted([c for c in df.columns if c.startswith('mm2_')])
        for i, (c15, c16, c2) in enumerate(zip(mm15_cols, mm16_cols, mm2_cols)):
            sub = df[['v001','survey_yr', c15, c16, c2]].copy()
            sub.columns = ['v001','survey_yr','death_yr','cause_code','alive']
            sub = sub[sub['alive'] == 0]  # only deceased siblings
            sub = sub.dropna(subset=['death_yr'])
            sub['death_yr'] = sub['death_yr'].astype(int)
            sub['violence'] = (sub['cause_code'].isin([2,3])).astype(float)  # accident/violence
            records.append(sub)
        return pd.concat(records, ignore_index=True)

    sib08 = reshape_mm(ir08m, 2008)
    sib13 = reshape_mm(ir13m, 2013)
    sib18 = reshape_mm(ir18m, 2018)
    sibs  = pd.concat([sib08, sib13, sib18], ignore_index=True)
    sibs  = sibs[sibs['death_yr'].between(2005,2018)].copy()

    # Merge conflict exposure in year of sibling death
    rain_sib = rain.rename(columns={'birth_year':'death_yr'})
    sibs = sibs.merge(rain_sib, on=['v001','death_yr'], how='left').merge(si, on='v001', how='left')
    sibs_mb = sibs[sibs['state'].isin(MB) & sibs['z_grow_birth'].notna()].copy()
    sibs_mb['fh_log'] = np.log1p(sibs_mb['fh_birth'].fillna(0))
    sibs_mb, yr_sib = add_yrfe(sibs_mb, 'death_yr')

    print(f"\nSibling deaths (MB, 2005-2018): {len(sibs_mb):,}")
    print(f"  Violence deaths: {sibs_mb['violence'].sum():.0f} ({sibs_mb['violence'].mean():.3f})")
    if len(sibs_mb) > 200:
        run_fe(sibs_mb, 'violence', 'fh_log', yr_sib, 'v001',
               label='Sibling violence death ~ FH conflict (cluster+year FE, MB)')
        run_fe(sibs_mb, 'violence', 'z_grow_birth', yr_sib, 'v001',
               label='Sibling violence death ~ z_grow (RF, cluster+year FE, MB)')
except Exception as e:
    print(f"  MM module error: {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY OF KEY FINDINGS")
print("="*60)
print("\nMost significant effects (OLS, cluster+year FE, Middle Belt):")
if results:
    for _, r in df_res.head(5).iterrows():
        star = '***' if r['p']<0.01 else ('**' if r['p']<0.05 else ('*' if r['p']<0.10 else '  '))
        print(f"  {star} {r['outcome']:<28} b={r['b']:+.5f}  p={r['p']:.4f}")
print("\nDone.")
