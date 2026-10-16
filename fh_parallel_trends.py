"""
Parallel trends validation for LGA DiD
Since 2003 IR DTA is unavailable, we use three approaches:
  1. Event study: separate 2013 and 2018 post-treatment interactions
  2. Placebo treatment: redefine treatment using PRE-2009 FH conflict
     (should be null if DiD is not driven by pre-existing trends)
  3. State × year FE robustness: add state-year interactions
"""
import warnings, numpy as np, pandas as pd, pyreadstat
from numpy.linalg import lstsq
from scipy.stats import t as tdist

warnings.filterwarnings('ignore')
SCRATCH = '/tmp/claude-0/-home-user-nig/61b5f4c8-5e77-5920-95de-a0a04a66cb77/scratchpad'
MB = ['Benue','Plateau','Kaduna','Nasarawa','Kogi','Taraba','Niger','Kwara','Borno','Adamawa','Bauchi','Gombe']

def within_demean(arr, cl_arr):
    df = pd.DataFrame({'v': arr.astype(float), 'cl': cl_arr})
    return arr.astype(float) - df.groupby('cl')['v'].transform('mean').values

def ols_cr(y_dm, X_dm, cl_arr, label=''):
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
        print(f"  {label:<38} b={b[0]:+.4f}  SE={se[0]:.4f}  t={t[0]:.2f}  p={p[0]:.4f} {star}")
    return b, se, t, p

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
cp = pd.read_csv(f'{SCRATCH}/cluster_panel.csv')
cp = cp[cp['rain_grow'] > -100]
si = cp[['v001','state']].drop_duplicates('v001')
lga_map = pd.read_csv(f'{SCRATCH}/cluster_lga.csv')

def load_ir(path, syr):
    want = ['v001','v133','v714','v743a','v744a','v744b','v744c','v744d','v744e',
            'v312','v025','v190','v012','m2a_1','m2b_1','m3a_1','m3b_1','m3n_1']
    avail = pyreadstat.read_dta(path, metadataonly=True)[1].column_names
    df, _ = pyreadstat.read_dta(path, apply_value_formats=False,
                                 usecols=[c for c in want if c in avail])
    df['survey_yr'] = syr
    return df

ir08 = load_ir(f'{SCRATCH}/ir_2008/NGIR53DT/NGIR53FL.DTA', 2008)
ir13 = load_ir(f'{SCRATCH}/ir_2013/NGIR6ADT/NGIR6AFL.DTA', 2013)
ir18 = load_ir(f'{SCRATCH}/ir_2018/NGIR7BDT/NGIR7BFL.DTA', 2018)
ir = pd.concat([ir08, ir13, ir18], ignore_index=True)

ir = ir.merge(lga_map[['v001','lga_id']], on='v001', how='left')
ir = ir.merge(si, on='v001', how='left')
ir = ir[ir['state'].isin(MB) & ir['lga_id'].notna()].copy()

def to_any(*cols, df):
    r = pd.Series(False, index=df.index)
    for c in cols:
        if c in df.columns: r = r | (df[c]==1)
    return r.astype(float)

ir['skilled_del']  = to_any('m3a_1','m3b_1', df=ir)
ir['any_anc']      = to_any('m2a_1','m2b_1', df=ir)
ir['no_attend']    = (ir['m3n_1']==1).astype(float) if 'm3n_1' in ir.columns else np.nan
ir['educ_yrs']     = ir['v133'].clip(0,25) if 'v133' in ir.columns else np.nan
ir['working']      = (ir['v714']==1).astype(float) if 'v714' in ir.columns else np.nan
ir['modern_contra']= ir['v312'].isin([1,2,3,5,6,7]).astype(float) if 'v312' in ir.columns else np.nan
dv_cols = [c for c in ['v744a','v744b','v744c','v744d','v744e'] if c in ir.columns]
if dv_cols: ir['dv_justified'] = to_any(*dv_cols, df=ir)

# ── Build LGA conflict measures ────────────────────────────────────────────────
cp_lga = cp.merge(lga_map[['v001','lga_id']], on='v001', how='left')

# Actual treatment: top-tercile FH events 2009-2018
lga_post = cp_lga[cp_lga['year'].between(2009,2018)].groupby('lga_id')['fh_events'].sum()
lga_pre  = cp_lga[cp_lga['year'].between(2005,2008)].groupby('lga_id')['fh_events'].sum()
thresh_post = lga_post.quantile(0.67)
thresh_pre  = lga_pre.quantile(0.67)

lga_treat = pd.DataFrame({
    'lga_id':    lga_post.index,
    'treated':   (lga_post >= thresh_post).astype(float),
    'treated_placebo': (lga_pre.reindex(lga_post.index).fillna(0) >= thresh_pre).astype(float),
    'lga_fh_post': lga_post.values,
    'lga_fh_pre':  lga_pre.reindex(lga_post.index).fillna(0).values,
}).reset_index(drop=True)

ir = ir.merge(lga_treat, on='lga_id', how='left')
ir['post2013'] = (ir['survey_yr']==2013).astype(float)
ir['post2018'] = (ir['survey_yr']==2018).astype(float)
ir['post']     = (ir['survey_yr'] >= 2013).astype(float)
ir['pt']       = ir['post'] * ir['treated']         # baseline DiD term
ir['p13t']     = ir['post2013'] * ir['treated']     # event study: 2013
ir['p18t']     = ir['post2018'] * ir['treated']     # event study: 2018
ir['pt_plac']  = ir['post'] * ir['treated_placebo'] # placebo DiD term

# Year dummies for FE
yr_dum = pd.get_dummies(ir['survey_yr'], prefix='yd', drop_first=True)
for c in yr_dum.columns: ir[c] = yr_dum[c].astype(float).values
yr_cols = list(yr_dum.columns)

# State dummies for state × year interaction (state-year cells)
state_yr_dum = pd.get_dummies(ir['state'].astype(str) + '_' + ir['survey_yr'].astype(str),
                               prefix='sy', drop_first=True)
for c in state_yr_dum.columns: ir[c] = state_yr_dum[c].astype(float).values
sy_cols = list(state_yr_dum.columns)

print(f"  MB sample: {len(ir):,} women, {ir['lga_id'].nunique()} LGAs")
print(f"  Treated (2009-2018): {ir[ir['treated']==1]['lga_id'].nunique()} LGAs")
print(f"  Treated (2005-2008 placebo): {ir[ir['treated_placebo']==1]['lga_id'].nunique()} LGAs")

outcomes = [
    ('skilled_del',   'Skilled delivery'),
    ('any_anc',       'Any ANC'),
    ('no_attend',     'No attendant'),
    ('educ_yrs',      'Years educ'),
    ('modern_contra', 'Modern contra'),
    ('working',       'Currently working'),
]

# ── TEST 1: Event study (2013 × treated, 2018 × treated) ─────────────────────
print("\n" + "="*70)
print("TEST 1: EVENT STUDY — 2013 and 2018 post effects (LGA FE + year FE)")
print("Base period = 2008; LGA-clustered SEs")
print("="*70)
print(f"  {'Outcome':<22} {'2013×treated':>14} {'p':>7} {'2018×treated':>14} {'p':>7}")
print("  " + "-"*65)

for oc, lab in outcomes:
    sub = ir.dropna(subset=[oc]).copy()
    cl = sub['lga_id'].values
    y  = within_demean(sub[oc].astype(float).values, cl)
    x1 = within_demean(sub['p13t'].values, cl)
    x2 = within_demean(sub['p18t'].values, cl)
    yr = np.column_stack([within_demean(sub[c].astype(float).values, cl) for c in yr_cols])
    X  = np.column_stack([x1, x2, yr])
    b  = lstsq(X, y, rcond=None)[0]
    resid = y - X @ b
    G = len(np.unique(cl)); k = X.shape[1]
    meat = np.zeros((k,k))
    for c in np.unique(cl):
        m = cl==c
        meat += X[m].T @ np.outer(resid[m], resid[m]) @ X[m]
    bread = np.linalg.inv(X.T @ X + np.eye(k)*1e-10)
    V  = bread @ meat @ bread * (G/(G-1)) * (len(y)-1)/(len(y)-k)
    se = np.sqrt(np.diag(V)); t = b/se; p = 2*tdist.sf(np.abs(t), df=G-1)
    s1 = '***' if p[0]<0.01 else ('**' if p[0]<0.05 else ('*' if p[0]<0.10 else ''))
    s2 = '***' if p[1]<0.01 else ('**' if p[1]<0.05 else ('*' if p[1]<0.10 else ''))
    print(f"  {lab:<22} {b[0]:+.4f} (p={p[0]:.3f}){s1:3} {b[1]:+.4f} (p={p[1]:.3f}){s2:3}")

# ── TEST 2: Placebo treatment ─────────────────────────────────────────────────
print("\n" + "="*70)
print("TEST 2: PLACEBO TREATMENT — top-tercile 2005-2008 FH conflict")
print("If null: pre-existing conflict trends do not drive the DiD")
print("="*70)

for oc, lab in outcomes:
    sub = ir.dropna(subset=[oc]).copy()
    cl = sub['lga_id'].values
    y  = within_demean(sub[oc].astype(float).values, cl)
    x  = within_demean(sub['pt_plac'].values, cl)
    yr = np.column_stack([within_demean(sub[c].astype(float).values, cl) for c in yr_cols])
    X  = np.column_stack([x, yr])
    ols_cr(y, X, cl, label=f'Placebo: {lab}')

# ── TEST 3: State × year FE robustness ───────────────────────────────────────
print("\n" + "="*70)
print("TEST 3: STATE × YEAR FE — absorbs state-level time trends")
print("DiD term must survive within-state variation")
print("="*70)

for oc, lab in outcomes:
    sub = ir.dropna(subset=[oc]).copy()
    cl = sub['lga_id'].values
    y  = within_demean(sub[oc].astype(float).values, cl)
    x  = within_demean(sub['pt'].values, cl)
    # State × year FE (absorbs state trends)
    sy_avail = [c for c in sy_cols if c in sub.columns]
    sy = np.column_stack([within_demean(sub[c].astype(float).values, cl) for c in sy_avail])
    X  = np.column_stack([x, sy])
    ols_cr(y, X, cl, label=f'State×yr FE: {lab}')

print("\nDone.")
