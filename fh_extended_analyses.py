"""
Extended analyses for farmer-herder conflict × health paper
1.  Spatial spillovers        — neighbor-LGA conflict added to DiD
2.  Urban / rural             — triple DiD
3.  Parity                    — primiparous vs. multiparous interaction
4.  Wealth                    — poor vs. rich interaction
5.  Continuous dose-response  — log(fh) replacing binary treatment
6.  Within-mother sibling FE  — mother FE + birth-year FE on IMR
7.  Birth timing              — LGA-year birth counts ~ lagged conflict
"""
import warnings, numpy as np, pandas as pd, pyreadstat, geopandas as gpd
from numpy.linalg import lstsq
from scipy.stats import t as tdist
warnings.filterwarnings('ignore')

S = '/tmp/claude-0/-home-user-nig/61b5f4c8-5e77-5920-95de-a0a04a66cb77/scratchpad'
MB = ['Benue','Plateau','Kaduna','Nasarawa','Kogi','Taraba','Niger','Kwara',
      'Borno','Adamawa','Bauchi','Gombe']

# ── helpers ───────────────────────────────────────────────────────────────────
def wdm(arr, cl):
    df = pd.DataFrame({'v': arr.astype(float), 'cl': cl})
    return arr.astype(float) - df.groupby('cl')['v'].transform('mean').values

def cr(y, X, cl):
    b = lstsq(X, y, rcond=None)[0]; resid = y - X @ b
    G = len(np.unique(cl)); k = X.shape[1]
    meat = np.zeros((k, k))
    for c in np.unique(cl):
        m = cl == c; meat += X[m].T @ np.outer(resid[m], resid[m]) @ X[m]
    bread = np.linalg.inv(X.T @ X + np.eye(k)*1e-10)
    V = bread @ meat @ bread * (G/(G-1)) * (len(y)-1)/(len(y)-k)
    se = np.sqrt(np.diag(V)); t = b/se; p = 2*tdist.sf(np.abs(t), df=G-1)
    return b, se, t, p

def st(p): return '***' if p<.001 else ('**' if p<.01 else ('*' if p<.05 else ('†' if p<.1 else '')))

def reg_print(label, b, se, p, idx=0):
    print(f"  {label:<46} b={b[idx]:+.4f}  se={se[idx]:.4f}  p={p[idx]:.4f} {st(p[idx])}")

# ── load IR data ──────────────────────────────────────────────────────────────
print("Loading IR data...")
cp  = pd.read_csv(f'{S}/cluster_panel.csv'); cp = cp[cp['rain_grow'] > -100]
si  = cp[['v001','state']].drop_duplicates('v001')
lga = pd.read_csv(f'{S}/cluster_lga.csv')

want_ir = ['v001','v133','v714','v312','v025','v190','v012',
           'm2a_1','m2b_1','m3a_1','m3b_1','m3n_1','v201']

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

def any_col(*cols, df):
    r = pd.Series(False, index=df.index)
    for c in cols:
        if c in df.columns: r = r | (df[c] == 1)
    return r.astype(float)

ir['skilled_del']   = any_col('m3a_1','m3b_1', df=ir)
ir['any_anc']       = any_col('m2a_1','m2b_1', df=ir)
ir['modern_contra'] = ir['v312'].isin([1,2,3,5,6,7]).astype(float)
ir['no_attend']     = (ir['m3n_1'] == 1).astype(float) if 'm3n_1' in ir.columns else np.nan

# LGA treatment
cp_lga  = cp.merge(lga[['v001','lga_id']], on='v001', how='left')
lga_post = cp_lga[cp_lga['year'].between(2009,2018)].groupby('lga_id')['fh_events'].sum()
thresh   = lga_post.quantile(0.67)
treat_df = pd.DataFrame({'lga_id': lga_post.index,
                         'treated': (lga_post >= thresh).astype(float),
                         'lga_fh':  lga_post.values}).reset_index(drop=True)
ir = ir.merge(treat_df, on='lga_id', how='left')
ir['treated'] = ir['treated'].fillna(0)
ir['lga_fh']  = ir['lga_fh'].fillna(0)
ir['post']       = (ir['survey_yr'] >= 2013).astype(float)
ir['post_treat'] = ir['post'] * ir['treated']
ir['post_logfh'] = ir['post'] * np.log1p(ir['lga_fh'])

yr_dum = pd.get_dummies(ir['survey_yr'], prefix='yd', drop_first=True)
for c in yr_dum.columns: ir[c] = yr_dum[c].astype(float).values
yr_cols = list(yr_dum.columns)

ocs = [('skilled_del','Skilled delivery'), ('any_anc','Any ANC'),
       ('modern_contra','Modern contra'),  ('no_attend','No attendant')]

print(f"IR sample: {len(ir):,} women | {ir['lga_id'].nunique()} LGAs | treated: {ir[ir['treated']==1]['lga_id'].nunique()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. SPATIAL SPILLOVERS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("1. SPATIAL SPILLOVERS — neighbor-LGA conflict in DiD")
print("="*70)

lga_poly = gpd.read_file(f'{S}/gadm2/gadm41_NGA_2.json')[['GID_2','NAME_1','geometry']]
lga_poly = lga_poly.rename(columns={'GID_2':'lga_id','NAME_1':'state'}).to_crs('EPSG:32632')

buf = lga_poly[['lga_id','geometry']].copy()
buf['geometry'] = buf.geometry.buffer(50)          # 50 m catches shared edges
adj = gpd.sjoin(buf.rename(columns={'lga_id':'a'}),
                lga_poly[['lga_id','geometry']].rename(columns={'lga_id':'b'}),
                how='left', predicate='intersects')
adj = adj[adj['a'] != adj['b']][['a','b']].drop_duplicates()
print(f"  Adjacent LGA pairs: {len(adj):,}")

# neighbor conflict
nc = adj.merge(lga_post.reset_index().rename(columns={'lga_id':'b','fh_events':'fh_b'}),
               on='b', how='left')
nc_sum = nc.groupby('a')['fh_b'].sum().reset_index().rename(columns={'a':'lga_id','fh_b':'neigh_fh'})
thresh_n = nc_sum['neigh_fh'].quantile(0.67)
nc_sum['neigh_treated'] = (nc_sum['neigh_fh'] >= thresh_n).astype(float)
ir = ir.merge(nc_sum[['lga_id','neigh_treated']], on='lga_id', how='left')
ir['neigh_treated'] = ir['neigh_treated'].fillna(0)
ir['post_neigh']    = ir['post'] * ir['neigh_treated']

print(f"\n  {'Outcome':<22}  {'Own-LGA DiD':>20}  {'Neighbor spillover':>22}")
print("  " + "-"*68)
results_spill = []
for oc, lab in ocs:
    sub = ir.dropna(subset=[oc]).copy(); cl = sub['lga_id'].values
    y  = wdm(sub[oc].astype(float).values, cl)
    x1 = wdm(sub['post_treat'].values, cl)
    x2 = wdm(sub['post_neigh'].values, cl)
    yr = np.column_stack([wdm(sub[c].astype(float).values, cl) for c in yr_cols])
    X  = np.column_stack([x1, x2, yr])
    b, se, t, p = cr(y, X, cl)
    print(f"  {lab:<22}  b={b[0]:+.4f}(p={p[0]:.3f}){st(p[0]):<3}  "
          f"neigh_b={b[1]:+.4f}(p={p[1]:.3f}){st(p[1])}")
    results_spill.append({'outcome':oc,'b_own':b[0],'p_own':p[0],'b_neigh':b[1],'p_neigh':p[1]})

# ═══════════════════════════════════════════════════════════════════════════════
# 2. URBAN / RURAL HETEROGENEITY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("2. URBAN / RURAL HETEROGENEITY — triple DiD (post×treated×urban)")
print("="*70)

ir['urban_w']   = (ir['v025'] == 1).astype(float)
ir['pt_urban']  = ir['post_treat'] * ir['urban_w']
ir['post_urb']  = ir['post'] * ir['urban_w']

print(f"\n  {'Outcome':<22}  {'Rural effect':>20}  {'Urban extra (p×t×urban)':>26}  Urban net")
print("  " + "-"*78)
results_urb = []
for oc, lab in ocs:
    sub = ir.dropna(subset=[oc]).copy(); cl = sub['lga_id'].values
    y  = wdm(sub[oc].astype(float).values, cl)
    x1 = wdm(sub['post_treat'].values, cl)
    x2 = wdm(sub['pt_urban'].values, cl)
    x3 = wdm(sub['post_urb'].values, cl)
    yr = np.column_stack([wdm(sub[c].astype(float).values, cl) for c in yr_cols])
    X  = np.column_stack([x1, x2, x3, yr])
    b, se, t, p = cr(y, X, cl)
    net = b[0]+b[1]
    print(f"  {lab:<22}  b={b[0]:+.4f}(p={p[0]:.3f}){st(p[0]):<3}  "
          f"extra={b[1]:+.4f}(p={p[1]:.3f}){st(p[1]):<3}  net={net:+.4f}")
    results_urb.append({'outcome':oc,'b_rural':b[0],'p_rural':p[0],'b_extra':b[1],'p_extra':p[1],'net_urban':net})

# ═══════════════════════════════════════════════════════════════════════════════
# 3. PARITY HETEROGENEITY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("3. PARITY — first birth (v201==1) vs. multiparous")
print("="*70)

if 'v201' in ir.columns:
    ir['prim']     = (ir['v201'] == 1).astype(float)
    ir['pt_prim']  = ir['post_treat'] * ir['prim']
    ir['post_prim']= ir['post'] * ir['prim']
    for oc, lab in [('skilled_del','Skilled delivery'), ('any_anc','Any ANC')]:
        sub = ir.dropna(subset=[oc]).copy(); cl = sub['lga_id'].values
        y  = wdm(sub[oc].astype(float).values, cl)
        x1 = wdm(sub['post_treat'].values, cl)
        x2 = wdm(sub['pt_prim'].values, cl)
        x3 = wdm(sub['post_prim'].values, cl)
        yr = np.column_stack([wdm(sub[c].astype(float).values, cl) for c in yr_cols])
        X  = np.column_stack([x1, x2, x3, yr])
        b, se, t, p = cr(y, X, cl)
        n1 = sub['prim'].sum()
        print(f"  {lab:<22}  multi_b={b[0]:+.4f}(p={p[0]:.3f}){st(p[0]):<3}  "
              f"prim_extra={b[1]:+.4f}(p={p[1]:.3f}){st(p[1]):<3}  n_first={int(n1):,}")
else:
    print("  v201 not available — skipped")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. WEALTH HETEROGENEITY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("4. WEALTH HETEROGENEITY — poor (Q1–Q2) interaction")
print("="*70)

if 'v190' in ir.columns:
    ir['poor']     = (ir['v190'] <= 2).astype(float)
    ir['pt_poor']  = ir['post_treat'] * ir['poor']
    ir['post_poor']= ir['post'] * ir['poor']
    for oc, lab in ocs:
        sub = ir.dropna(subset=[oc]).copy(); cl = sub['lga_id'].values
        y  = wdm(sub[oc].astype(float).values, cl)
        x1 = wdm(sub['post_treat'].values, cl)
        x2 = wdm(sub['pt_poor'].values, cl)
        x3 = wdm(sub['post_poor'].values, cl)
        yr = np.column_stack([wdm(sub[c].astype(float).values, cl) for c in yr_cols])
        X  = np.column_stack([x1, x2, x3, yr])
        b, se, t, p = cr(y, X, cl)
        net_poor = b[0]+b[1]
        print(f"  {lab:<22}  rich_b={b[0]:+.4f}(p={p[0]:.3f}){st(p[0]):<3}  "
              f"poor_extra={b[1]:+.4f}(p={p[1]:.3f}){st(p[1]):<3}  poor_net={net_poor:+.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. CONTINUOUS DOSE-RESPONSE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("5. CONTINUOUS DOSE-RESPONSE — post × log(LGA FH conflict) replacing binary")
print("="*70)

for oc, lab in ocs:
    sub = ir.dropna(subset=[oc]).copy(); cl = sub['lga_id'].values
    y  = wdm(sub[oc].astype(float).values, cl)
    x  = wdm(sub['post_logfh'].values, cl)
    yr = np.column_stack([wdm(sub[c].astype(float).values, cl) for c in yr_cols])
    X  = np.column_stack([x, yr])
    b, se, t, p = cr(y, X, cl)
    print(f"  {lab:<22}  b={b[0]:+.5f}  se={se[0]:.5f}  p={p[0]:.4f} {st(p[0])}  "
          f"[10-event increase ≈ {b[0]*np.log1p(10)*100:.2f}pp]")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. WITHIN-MOTHER SIBLING FIXED EFFECTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("6. WITHIN-MOTHER SIBLING FIXED EFFECTS")
print("Mother FE + birth-year FE | outcome = IMR & U5 | X = log(fh+1) at v001×birth_year")
print("Identification: same mother, children born in different conflict years")
print("="*70)

br_want = ['caseid','v001','v002','v003','v025','v190','b3','b5','b7','b8','b0','b11']

def load_br(path, syr):
    avail = pyreadstat.read_dta(path, metadataonly=True)[1].column_names
    df, _ = pyreadstat.read_dta(path, apply_value_formats=False,
                                 usecols=[c for c in br_want if c in avail])
    df['survey_yr'] = syr
    df['birth_year']  = ((df['b3'] - 1) // 12 + 1900).astype(int)
    df['birth_month'] = ((df['b3'] - 1) % 12 + 1).astype(int)
    if 'b0' in df.columns: df = df[df['b0'].fillna(0) == 0]
    df['b7'] = df['b7'].fillna(999)
    df['imr'] = ((df['b5'] == 0) & (df['b7'] < 12)).astype(float)
    df['u5']  = ((df['b5'] == 0) & (df['b7'] < 60)).astype(float)
    # stable mom ID within this survey wave
    df['mom_id'] = df['caseid'].astype(str) + '_' + str(syr)
    return df

print("  Loading BR files...")
br = pd.concat([
    load_br(f'{S}/br_2008/NGBR53DT/NGBR53FL.DTA', 2008),
    load_br(f'{S}/br_2013/NGBR6ADT/NGBR6AFL.DTA', 2013),
    load_br(f'{S}/br_2018/NGBR7BDT/NGBR7BFL.DTA', 2018),
], ignore_index=True)

# Censoring: IMR needs ≥2 yrs before survey, U5 needs ≥6 yrs
br_imr = br[(br['birth_year'].between(2005, 2018)) &
            (br['birth_year'] <= br['survey_yr'] - 2)].copy()
br_u5  = br[(br['birth_year'].between(2005, 2018)) &
            (br['birth_year'] <= br['survey_yr'] - 6)].copy()

# Merge conflict + geography
rain_br = cp[['v001','year','fh_events']].rename(columns={'year':'birth_year'})
for bdf in [br_imr, br_u5]:
    bdf.drop(columns=[c for c in bdf.columns if c in ['fh_events','state','lga_id']], inplace=True, errors='ignore')
    bdf.update(bdf.merge(rain_br, on=['v001','birth_year'], how='left').set_index(bdf.index)[['fh_events']])

br_imr = br_imr.merge(rain_br, on=['v001','birth_year'], how='left', suffixes=('','_r'))
br_imr = br_imr.merge(si, on='v001', how='left').merge(lga[['v001','lga_id']], on='v001', how='left')
br_u5  = br_u5.merge(rain_br, on=['v001','birth_year'], how='left', suffixes=('','_r'))
br_u5  = br_u5.merge(si, on='v001', how='left').merge(lga[['v001','lga_id']], on='v001', how='left')

# Correct merge: use fh_events from merge if original was NaN
for bdf in [br_imr, br_u5]:
    if 'fh_events_r' in bdf.columns:
        bdf['fh_events'] = bdf['fh_events'].combine_first(bdf['fh_events_r'])
        bdf.drop(columns=['fh_events_r'], inplace=True)

br_imr_mb = br_imr[br_imr['state'].isin(MB) & br_imr['fh_events'].notna()].copy()
br_u5_mb  = br_u5[br_u5['state'].isin(MB)   & br_u5['fh_events'].notna()].copy()

br_imr_mb['fh_log'] = np.log1p(br_imr_mb['fh_events'])
br_u5_mb['fh_log']  = np.log1p(br_u5_mb['fh_events'])

for lab_df, oc_str, dfr in [('IMR','imr',br_imr_mb), ('U5','u5',br_u5_mb)]:
    # Keep only mothers with ≥2 births (needed for within FE to have variation)
    cnts = dfr.groupby('mom_id').size()
    multi = cnts[cnts >= 2].index
    sub   = dfr[dfr['mom_id'].isin(multi)].dropna(subset=[oc_str]).copy()

    yr_dum_br = pd.get_dummies(sub['birth_year'], prefix='by', drop_first=True)
    for c in yr_dum_br.columns: sub[c] = yr_dum_br[c].astype(float).values
    yr_br = list(yr_dum_br.columns)

    # Demean by mother FE
    cl_mom = sub['mom_id'].values
    cl_lga = sub['lga_id'].values
    y = wdm(sub[oc_str].astype(float).values, cl_mom)
    x = wdm(sub['fh_log'].values, cl_mom)
    yr_mat = np.column_stack([wdm(sub[c].astype(float).values, cl_mom) for c in yr_br])
    X = np.column_stack([x, yr_mat])
    b, se, t, p = cr(y, X, cl_lga)
    print(f"  {lab_df:<5}  b={b[0]:+.5f}  se={se[0]:.5f}  p={p[0]:.4f} {st(p[0])}"
          f"  [1-SD FH ≈ {b[0]*np.std(sub['fh_log'])*100:.2f}pp]"
          f"  N_births={len(sub):,}  N_moms={sub['mom_id'].nunique():,}  N_LGAs={len(np.unique(cl_lga)):,}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. BIRTH TIMING — does conflict suppress conceptions?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("7. BIRTH TIMING — LGA-year births ~ lagged LGA conflict")
print("log(births+1)[t] ~ log(fh+1)[t-1]  |  LGA FE + year FE, LGA-clustered SEs")
print("="*70)

# Aggregate BR births to LGA × birth_year
br_all = pd.concat([
    load_br(f'{S}/br_2008/NGBR53DT/NGBR53FL.DTA', 2008),
    load_br(f'{S}/br_2013/NGBR6ADT/NGBR6AFL.DTA', 2013),
    load_br(f'{S}/br_2018/NGBR7BDT/NGBR7BFL.DTA', 2018),
], ignore_index=True)
br_all = br_all.merge(si, on='v001', how='left').merge(lga[['v001','lga_id']], on='v001', how='left')
br_all = br_all[br_all['state'].isin(MB) & br_all['lga_id'].notna()].copy()
# Use only births reported within the relevant recall window (last 5 years)
br_all = br_all[br_all['birth_year'].between(2003,2018) &
                (br_all['birth_year'] >= br_all['survey_yr'] - 5)].copy()

births_lga = br_all.groupby(['lga_id','birth_year']).size().reset_index(name='n_births')
births_lga['log_births'] = np.log1p(births_lga['n_births'])

# LGA-level lagged conflict: aggregate cluster events → LGA year
lga_yr_conflict = (cp.merge(lga[['v001','lga_id']], on='v001', how='left')
                     .groupby(['lga_id','year'])['fh_events'].sum().reset_index())
lga_yr_conflict['birth_year'] = lga_yr_conflict['year'] + 1  # 1-year lag ≈ 9-month conception lag
lga_yr_conflict['fh_log_lag'] = np.log1p(lga_yr_conflict['fh_events'])

bp = births_lga.merge(lga_yr_conflict[['lga_id','birth_year','fh_log_lag']],
                      on=['lga_id','birth_year'], how='inner').dropna()
print(f"  LGA-year obs: {len(bp):,}  |  LGAs: {bp['lga_id'].nunique()}")

yr_bp = pd.get_dummies(bp['birth_year'], prefix='by', drop_first=True)
for c in yr_bp.columns: bp[c] = yr_bp[c].astype(float).values
yr_bp_cols = list(yr_bp.columns)

cl_lga_bp = bp['lga_id'].values
y  = wdm(bp['log_births'].values, cl_lga_bp)
x  = wdm(bp['fh_log_lag'].values, cl_lga_bp)
yr = np.column_stack([wdm(bp[c].astype(float).values, cl_lga_bp) for c in yr_bp_cols])
X  = np.column_stack([x, yr])
b, se, t, p = cr(y, X, cl_lga_bp)
print(f"  log(births)[t] ~ log(fh)[t-1]:  b={b[0]:+.5f}  se={se[0]:.5f}  p={p[0]:.4f} {st(p[0])}")
print(f"  Interpretation: {'NEGATIVE — conflict reduces conceptions ~9 months later' if b[0]<0 else 'positive / null'}")

# Also test contemporaneous (no lag) for comparison
lga_yr_conflict0 = lga_yr_conflict.copy()
lga_yr_conflict0['birth_year'] = lga_yr_conflict0['year']  # no lag
lga_yr_conflict0['fh_log_cont'] = np.log1p(lga_yr_conflict0['fh_events'])
bp2 = births_lga.merge(lga_yr_conflict0[['lga_id','birth_year','fh_log_cont']],
                       on=['lga_id','birth_year'], how='inner').dropna()
yr2 = pd.get_dummies(bp2['birth_year'], prefix='by', drop_first=True)
for c in yr2.columns: bp2[c] = yr2[c].astype(float).values
cl2 = bp2['lga_id'].values
yc = wdm(bp2['log_births'].values, cl2)
xc = wdm(bp2['fh_log_cont'].values, cl2)
yrc = np.column_stack([wdm(bp2[c].astype(float).values, cl2) for c in list(yr2.columns)])
Xc = np.column_stack([xc, yrc])
bc, sec, tc, pc = cr(yc, Xc, cl2)
print(f"  log(births)[t] ~ log(fh)[t]:    b={bc[0]:+.5f}  se={sec[0]:.5f}  p={pc[0]:.4f} {st(pc[0])}  (contemporaneous comparison)")

print("\n" + "="*70)
print("DONE — all analyses complete")
print("="*70)
