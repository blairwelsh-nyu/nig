"""
Proper parallel trends test with 2003 DHS as genuine pre-treatment period.
Design: 4 waves: 2003 (pre), 2008 (pre), 2013 (post), 2018 (post)
Treatment onset: FH conflict escalation post-2009
Event study: 2003×treated and 2008×treated should be ~0 (pre-trends)
             2013×treated and 2018×treated should be negative (treatment effect)
"""
import warnings, numpy as np, pandas as pd, pyreadstat, geopandas as gpd
from numpy.linalg import lstsq
from scipy.stats import t as tdist

warnings.filterwarnings('ignore')
SCRATCH = '/tmp/claude-0/-home-user-nig/61b5f4c8-5e77-5920-95de-a0a04a66cb77/scratchpad'
MB = ['Benue','Plateau','Kaduna','Nasarawa','Kogi','Taraba','Niger','Kwara','Borno','Adamawa','Bauchi','Gombe']

def within_demean(arr, cl_arr):
    df = pd.DataFrame({'v': arr.astype(float), 'cl': cl_arr})
    return arr.astype(float) - df.groupby('cl')['v'].transform('mean').values

def ols_cr_full(y_dm, X_dm, cl_arr):
    b = lstsq(X_dm, y_dm, rcond=None)[0]
    resid = y_dm - X_dm @ b
    G = len(np.unique(cl_arr)); k = X_dm.shape[1]
    meat = np.zeros((k,k))
    for c in np.unique(cl_arr):
        m = cl_arr == c
        meat += X_dm[m].T @ np.outer(resid[m], resid[m]) @ X_dm[m]
    bread = np.linalg.inv(X_dm.T @ X_dm + np.eye(k)*1e-10)
    V = bread @ meat @ bread * (G/(G-1)) * (len(y_dm)-1)/(len(y_dm)-k)
    se = np.sqrt(np.diag(V)); t = b/se; p = 2*tdist.sf(np.abs(t), df=G-1)
    return b, se, t, p

# ── Step 1: Assign 2003 DHS clusters → LGAs via spatial join ─────────────────
print("Spatially joining 2003 clusters to LGAs...")
gps03 = gpd.read_file(f'{SCRATCH}/dhs_2003/NGGE4BFL.shp')[['DHSCLUST','LATNUM','LONGNUM','geometry']]
gps03 = gps03[gps03['LATNUM'] != 0].copy()  # drop missing coords
gps03 = gps03.to_crs('EPSG:32632')

lga_poly = gpd.read_file(f'{SCRATCH}/gadm2/gadm41_NGA_2.json')[['GID_2','NAME_1','NAME_2','geometry']]
lga_poly = lga_poly.rename(columns={'GID_2':'lga_id','NAME_1':'state','NAME_2':'lga_name'})
lga_poly = lga_poly.to_crs('EPSG:32632')

j = gpd.sjoin(gps03, lga_poly[['lga_id','state','geometry']], how='left', predicate='within')
# Nearest fallback for unmatched
unmatched = j[j['lga_id'].isna()].index
if len(unmatched):
    nm = gps03.loc[unmatched]
    lga_c = lga_poly.copy(); lga_c['geometry'] = lga_c.geometry.centroid
    for idx, row in nm.iterrows():
        dists = lga_c.geometry.distance(row.geometry)
        best = lga_c.iloc[dists.idxmin()]
        j.loc[idx, 'lga_id'] = best['lga_id']
        j.loc[idx, 'state']  = best['state']

cluster_lga_03 = j[['DHSCLUST','lga_id','state']].rename(columns={'DHSCLUST':'v001_03'}).drop_duplicates('v001_03')
print(f"  2003 clusters: {len(cluster_lga_03)}, matched to {cluster_lga_03['lga_id'].nunique()} LGAs")

# ── Step 2: Load 2003 IR ──────────────────────────────────────────────────────
print("Loading 2003 IR...")
want = ['v001','v133','v714','v743a','v744a','v744b','v744c','v744d','v744e',
        'v312','v025','v190','v012','m2a_1','m2b_1','m3a_1','m3b_1','m3n_1']
avail = pyreadstat.read_dta(f'{SCRATCH}/ir_2003/NGIR4BDT/NGIR4BFL.DTA', metadataonly=True)[1].column_names
ir03, _ = pyreadstat.read_dta(f'{SCRATCH}/ir_2003/NGIR4BDT/NGIR4BFL.DTA',
                               apply_value_formats=False,
                               usecols=[c for c in want if c in avail])
ir03['survey_yr'] = 2003
ir03 = ir03.merge(cluster_lga_03.rename(columns={'v001_03':'v001'}), on='v001', how='left')
print(f"  2003 women: {len(ir03):,}, LGA matched: {ir03['lga_id'].notna().mean():.1%}")

# ── Step 3: Load 2008-2018 IR ─────────────────────────────────────────────────
def load_ir(path, syr):
    avail = pyreadstat.read_dta(path, metadataonly=True)[1].column_names
    df, _ = pyreadstat.read_dta(path, apply_value_formats=False,
                                 usecols=[c for c in want if c in avail])
    df['survey_yr'] = syr
    return df

ir08 = load_ir(f'{SCRATCH}/ir_2008/NGIR53DT/NGIR53FL.DTA', 2008)
ir13 = load_ir(f'{SCRATCH}/ir_2013/NGIR6ADT/NGIR6AFL.DTA', 2013)
ir18 = load_ir(f'{SCRATCH}/ir_2018/NGIR7BDT/NGIR7BFL.DTA', 2018)

lga_map = pd.read_csv(f'{SCRATCH}/cluster_lga.csv')
for df in [ir08, ir13, ir18]:
    df = df.merge(lga_map[['v001','lga_id','state']], on='v001', how='left')

ir08 = ir08.merge(lga_map[['v001','lga_id','state']], on='v001', how='left')
ir13 = ir13.merge(lga_map[['v001','lga_id','state']], on='v001', how='left')
ir18 = ir18.merge(lga_map[['v001','lga_id','state']], on='v001', how='left')

# ── Step 4: Pool all 4 waves ──────────────────────────────────────────────────
ir = pd.concat([ir03, ir08, ir13, ir18], ignore_index=True)
ir = ir[ir['state'].isin(MB) & ir['lga_id'].notna()].copy()

def to_any(*cols, df):
    r = pd.Series(False, index=df.index)
    for c in cols:
        if c in df.columns: r = r | (df[c]==1)
    return r.astype(float)

ir['skilled_del']   = to_any('m3a_1','m3b_1', df=ir)
ir['any_anc']       = to_any('m2a_1','m2b_1', df=ir)
ir['no_attend']     = (ir['m3n_1']==1).astype(float) if 'm3n_1' in ir.columns else np.nan
ir['educ_yrs']      = ir['v133'].clip(0,25) if 'v133' in ir.columns else np.nan
ir['working']       = (ir['v714']==1).astype(float) if 'v714' in ir.columns else np.nan
ir['modern_contra'] = ir['v312'].isin([1,2,3,5,6,7]).astype(float) if 'v312' in ir.columns else np.nan
dv_cols = [c for c in ['v744a','v744b','v744c','v744d','v744e'] if c in ir.columns]
if dv_cols: ir['dv_justified'] = to_any(*dv_cols, df=ir)

print(f"\n4-wave MB sample: {len(ir):,} women")
print(ir.groupby('survey_yr').size().to_string())

# ── Step 5: Build treatment and event-study dummies ──────────────────────────
cp = pd.read_csv(f'{SCRATCH}/cluster_panel.csv')
cp = cp[cp['rain_grow'] > -100]
cp_lga = cp.merge(lga_map[['v001','lga_id']], on='v001', how='left')
lga_post = cp_lga[cp_lga['year'].between(2009,2018)].groupby('lga_id')['fh_events'].sum()
thresh = lga_post.quantile(0.67)
treat_df = pd.DataFrame({'lga_id': lga_post.index,
                          'treated': (lga_post >= thresh).astype(float)}).reset_index(drop=True)

ir = ir.merge(treat_df, on='lga_id', how='left')
ir['treated'] = ir['treated'].fillna(0)

# Event-study dummies (base = 2008)
for yr in [2003, 2013, 2018]:
    ir[f'd{yr}'] = (ir['survey_yr'] == yr).astype(float)
    ir[f'd{yr}t'] = ir[f'd{yr}'] * ir['treated']

# Year FE dummies
yr_dum = pd.get_dummies(ir['survey_yr'], prefix='yd', drop_first=True)
for c in yr_dum.columns: ir[c] = yr_dum[c].astype(float).values
yr_cols = list(yr_dum.columns)

print(f"\nTreated LGAs: {ir[ir['treated']==1]['lga_id'].nunique()}, Control: {ir[ir['treated']==0]['lga_id'].nunique()}")

# ── Step 6: Event study regression ───────────────────────────────────────────
print("\n" + "="*72)
print("EVENT STUDY: 4-wave DiD with 2003 pre-period")
print("Base year = 2008 | LGA-clustered SEs | LGA FE + year FE")
print("="*72)
print(f"\n  {'Outcome':<22} {'2003×T':>12} {'p':>7} {'2008 (base)':>12} {'2013×T':>12} {'p':>7} {'2018×T':>12} {'p':>7}")
print("  " + "-"*90)

outcomes = [
    ('skilled_del',   'Skilled delivery'),
    ('any_anc',       'Any ANC'),
    ('no_attend',     'No attendant'),
    ('educ_yrs',      'Years educ'),
    ('modern_contra', 'Modern contra'),
    ('working',       'Working'),
]

all_results = []
for oc, lab in outcomes:
    sub = ir.dropna(subset=[oc]).copy()
    cl  = sub['lga_id'].values
    y   = within_demean(sub[oc].astype(float).values, cl)
    x03 = within_demean(sub['d2003t'].values, cl)
    x13 = within_demean(sub['d2013t'].values, cl)
    x18 = within_demean(sub['d2018t'].values, cl)
    yr  = np.column_stack([within_demean(sub[c].astype(float).values, cl) for c in yr_cols])
    X   = np.column_stack([x03, x13, x18, yr])
    b, se, t, p = ols_cr_full(y, X, cl)

    def sf(p_): return '***' if p_<0.01 else ('**' if p_<0.05 else ('*' if p_<0.10 else '   '))
    print(f"  {lab:<22} {b[0]:+.4f}(p={p[0]:.3f}){sf(p[0])} {'[base]':>12} {b[1]:+.4f}(p={p[1]:.3f}){sf(p[1])} {b[2]:+.4f}(p={p[2]:.3f}){sf(p[2])}")
    all_results.append({'outcome': oc, 'label': lab,
                        'b03': b[0], 'p03': p[0], 'b13': b[1], 'p13': p[1], 'b18': b[2], 'p18': p[2],
                        'N': len(sub)})

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("PARALLEL TRENDS VERDICT")
print("="*72)
for r in all_results:
    pre_ok = r['p03'] > 0.10
    post_sig = r['p13'] < 0.10 or r['p18'] < 0.10
    verdict = "PASS" if pre_ok else "FAIL"
    print(f"  {r['label']:<22}  Pre-trend (2003): b={r['b03']:+.4f} p={r['p03']:.3f} [{verdict}]  "
          f"Post-2013: b={r['b13']:+.4f} p={r['p13']:.3f}  Post-2018: b={r['b18']:+.4f} p={r['p18']:.3f}")

print("\nDone.")
