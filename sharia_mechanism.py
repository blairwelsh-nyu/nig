"""
Mechanism tests for the Sharia RD.
Four tests:
  1. Muslim heterogeneity  — does the effect differ by religion?
  2. Sharia salience       — stronger among respondents who named Sharia as top problem?
  3. Mediation (religious-leader trust) — formal causal mediation
  4. Extended outcomes     — broader state-legitimacy items (democracy satisfaction, etc.)
"""
import os, warnings
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from sklearn.neighbors import BallTree
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')

os.chdir('/home/user/nig')

# ── Boundary / constants ──────────────────────────────────────────────────────
SHARIA_BOUNDARY_COORDS = [
    (3.40, 11.85),(3.90, 11.20),(4.35, 10.95),(4.85, 10.65),(5.25, 10.40),
    (5.65, 10.10),(6.05,  9.85),(6.55,  9.55),(7.00,  9.35),(7.45,  9.15),
    (7.95,  9.40),(8.50,  9.60),(9.00,  9.80),(9.50,  9.75),(9.95,  9.65),
    (10.40, 9.35),(10.95, 9.10),(11.45, 9.00),(12.00, 9.00),(12.60, 9.10),
    (13.10, 9.20),(13.70, 8.80),
]
boundary_line = LineString(SHARIA_BOUNDARY_COORDS)
SHARIA_STATES = {'Zamfara','Kano','Sokoto','Katsina','Bauchi','Niger',
                 'Jigawa','Kebbi','Yobe','Borno','Kaduna','Gombe'}
BW = 150

# ── Load data ─────────────────────────────────────────────────────────────────
acled = pd.read_csv('nigeria_acled.csv')
state_cents = (acled.groupby('admin1')[['latitude','longitude']]
               .mean().reset_index())
state_cents.columns = ['state','lat','lon']

def assign_state(lat_arr, lon_arr):
    cent_rad = np.radians(state_cents[['lat','lon']].values)
    tree = BallTree(cent_rad, metric='haversine')
    _, idx = tree.query(np.radians(np.column_stack([lat_arr, lon_arr])), k=1)
    return state_cents['state'].values[idx.flatten()]

def signed_dist_km(lat_arr, lon_arr, state_arr, earth_r=6371.0):
    dist = np.full(len(lat_arr), np.nan)
    for i in range(len(lat_arr)):
        p = Point(lon_arr[i], lat_arr[i])
        lat_rad = np.radians(lat_arr[i])
        dlat = np.pi * earth_r / 180.0
        dlon = dlat * np.cos(lat_rad)
        km = boundary_line.distance(p) * np.sqrt(dlat**2 + dlon**2) / np.sqrt(2)
        dist[i] = km if state_arr[i] in SHARIA_STATES else -km
    return dist

r2_raw = pd.read_excel('NIG_r2.csv.xlsx')
r2_raw.columns = r2_raw.columns.str.lower().str.strip()
lat_col = next(c for c in r2_raw.columns if 'lat' in c and c != 'latitude' or c == 'latitude')
lon_col = next(c for c in r2_raw.columns if 'lon' in c and c != 'longitude' or c == 'longitude')
# Use the actual lat/lon columns
lat_col = next(c for c in r2_raw.columns if c in ('latitude', 'lat'))
lon_col = next(c for c in r2_raw.columns if c in ('longitude', 'lon'))
r2_raw = r2_raw.rename(columns={lat_col: 'lat', lon_col: 'lon'})
r2_raw['lat'] = pd.to_numeric(r2_raw['lat'], errors='coerce')
r2_raw['lon'] = pd.to_numeric(r2_raw['lon'], errors='coerce')
r2_raw = r2_raw.dropna(subset=['lat', 'lon']).reset_index(drop=True)

r2_raw['state']   = assign_state(r2_raw['lat'].values, r2_raw['lon'].values)
r2_raw['dist_km'] = signed_dist_km(r2_raw['lat'].values, r2_raw['lon'].values,
                                    r2_raw['state'].values)
r2_raw['sharia']  = r2_raw['state'].isin(SHARIA_STATES).astype(float)

# Police / courts trust (0–3 scale, 7+ → NaN)
for col in ['q43i', 'q43j']:
    r2_raw[col] = pd.to_numeric(r2_raw[col], errors='coerce')
    r2_raw.loc[r2_raw[col] >= 7, col] = np.nan

# ── Moderators ────────────────────────────────────────────────────────────────
# Muslim = q86 code 6
r2_raw['muslim'] = (r2_raw['q86'] == 6).astype(float)

# Sharia named as top problem in q44 (any part)
r2_raw['sharia_concern'] = (
    (r2_raw['q44pt1'] == 260) |
    (r2_raw['q44pt2'] == 260) |
    (r2_raw['q44pt3'] == 260)
).astype(float)

# Religious leader trust (mediator candidate) — 0–3 scale
r2_raw['rel_trust'] = pd.to_numeric(r2_raw['q43g'], errors='coerce')
r2_raw.loc[r2_raw['rel_trust'] >= 7, 'rel_trust'] = np.nan

# ── Extended outcomes (democracy / governance) ────────────────────────────────
# q47: 0=no democracy, 1=DK, 2=sometimes, 3=often, 4=always → recode to 0-3
r2_raw['dem_supply']  = pd.to_numeric(r2_raw['q47'], errors='coerce')
r2_raw.loc[r2_raw['dem_supply'].isin([0, 9]), 'dem_supply'] = np.nan
r2_raw['dem_supply'] -= 1   # 1-4 → 0-3

# q45a: government performance (1=very badly, 4=very well) → recode 0-3
r2_raw['govt_perf'] = pd.to_numeric(r2_raw['q45a'], errors='coerce')
r2_raw.loc[r2_raw['govt_perf'] >= 9, 'govt_perf'] = np.nan
r2_raw['govt_perf'] -= 1   # 1-4 → 0-3

# q43a: trust in president (0-3, 7+ → NaN)
r2_raw['trust_pres'] = pd.to_numeric(r2_raw['q43a'], errors='coerce')
r2_raw.loc[r2_raw['trust_pres'] >= 7, 'trust_pres'] = np.nan

# q43b: trust in parliament
r2_raw['trust_parl'] = pd.to_numeric(r2_raw['q43b'], errors='coerce')
r2_raw.loc[r2_raw['trust_parl'] >= 7, 'trust_parl'] = np.nan

# ── Core RD helper ────────────────────────────────────────────────────────────
def rd(df, outcome, bw=BW, extra_vars=None, formula_suffix=''):
    d = df.dropna(subset=[outcome, 'dist_km', 'sharia']).copy()
    d = d[d['dist_km'].abs() <= bw]
    if extra_vars:
        d = d.dropna(subset=extra_vars)
    if len(d) < 50 or d[outcome].nunique() < 2:
        return None
    fml = f'y ~ sharia + x + x:sharia{formula_suffix}'
    try:
        m = smf.ols(fml, data=d.assign(y=d[outcome], x=d['dist_km'],
                    **({v: d[v] for v in (extra_vars or [])}))).fit(cov_type='HC3')
    except Exception as e:
        return None
    out = {}
    for param in m.params.index:
        out[param] = {'coef': m.params[param], 'se': m.bse[param],
                      'pval': m.pvalues[param]}
    out['n'] = len(d)
    return out

def star(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.10:  return '†'
    return ''

# ══════════════════════════════════════════════════════════════════════════════
print('='*70)
print('TEST 1: MUSLIM HETEROGENEITY')
print('  y ~ sharia + x + x:sharia + muslim + sharia:muslim')
print('  Under substitution: sharia:muslim > 0 (larger NEGATIVE effect for Muslims)')
print('  Under backlash:     sharia:muslim < 0 (larger NEGATIVE effect for non-Muslims)')
print('='*70)
print()

for outcome, label in [('q43i','Trust: police'), ('q43j','Trust: courts')]:
    res = rd(r2_raw, outcome,
             extra_vars=['muslim'],
             formula_suffix=' + muslim + sharia:muslim')
    if res is None:
        print(f'{label}: insufficient data')
        continue
    main_coef = res.get('sharia', {})
    int_coef  = res.get('sharia:muslim', {})
    mus_coef  = res.get('muslim', {})
    print(f'{label} (N={res["n"]}):')
    c, s, p = main_coef.get('coef', np.nan), main_coef.get('se', np.nan), main_coef.get('pval', np.nan)
    print(f'  β(sharia)         = {c:+.3f}  SE={s:.3f}  p={p:.3f}{star(p)}')
    c, s, p = int_coef.get('coef', np.nan), int_coef.get('se', np.nan), int_coef.get('pval', np.nan)
    print(f'  β(sharia×muslim)  = {c:+.3f}  SE={s:.3f}  p={p:.3f}{star(p)}')
    c, s, p = mus_coef.get('coef', np.nan), mus_coef.get('se', np.nan), mus_coef.get('pval', np.nan)
    print(f'  β(muslim)         = {c:+.3f}  SE={s:.3f}  p={p:.3f}{star(p)}')

    # Marginal effects: effect for Muslims and non-Muslims separately
    d_bw = r2_raw.dropna(subset=[outcome, 'dist_km', 'sharia', 'muslim'])
    d_bw = d_bw[d_bw['dist_km'].abs() <= BW]
    for grp, grp_label in [(0, 'Non-Muslim'), (1, 'Muslim')]:
        dg = d_bw[d_bw['muslim'] == grp]
        rg = rd(dg, outcome, bw=BW)
        if rg and 'sharia' in rg:
            c, s, p = rg['sharia']['coef'], rg['sharia']['se'], rg['sharia']['pval']
            print(f'    [{grp_label} only, N={rg["n"]}]: β = {c:+.3f}  SE={s:.3f}  p={p:.3f}{star(p)}')
    print()

# ══════════════════════════════════════════════════════════════════════════════
print('='*70)
print('TEST 2: SHARIA SALIENCE MODERATION')
print('  y ~ sharia + x + x:sharia + sharia_concern + sharia:sharia_concern')
print('  Respondents who named Sharia as a top problem (q44 code 260)')
print('='*70)
print()

for outcome, label in [('q43i','Trust: police'), ('q43j','Trust: courts')]:
    res = rd(r2_raw, outcome,
             extra_vars=['sharia_concern'],
             formula_suffix=' + sharia_concern + sharia:sharia_concern')
    if res is None:
        print(f'{label}: insufficient data'); continue
    main_coef = res.get('sharia', {})
    int_coef  = res.get('sharia:sharia_concern', {})
    print(f'{label} (N={res["n"]}):')
    c, s, p = main_coef.get('coef', np.nan), main_coef.get('se', np.nan), main_coef.get('pval', np.nan)
    print(f'  β(sharia)                   = {c:+.3f}  SE={s:.3f}  p={p:.3f}{star(p)}')
    c, s, p = int_coef.get('coef', np.nan), int_coef.get('se', np.nan), int_coef.get('pval', np.nan)
    print(f'  β(sharia×sharia_concern)    = {c:+.3f}  SE={s:.3f}  p={p:.3f}{star(p)}')

    # Marginal: concern vs no concern
    d_bw = r2_raw.dropna(subset=[outcome, 'dist_km', 'sharia', 'sharia_concern'])
    d_bw = d_bw[d_bw['dist_km'].abs() <= BW]
    for grp, grp_label in [(0,'Not concerned'), (1,'Sharia as problem')]:
        dg = d_bw[d_bw['sharia_concern'] == grp]
        rg = rd(dg, outcome, bw=BW)
        if rg and 'sharia' in rg:
            c, s, p = rg['sharia']['coef'], rg['sharia']['se'], rg['sharia']['pval']
            print(f'    [{grp_label}, N={rg["n"]}]: β = {c:+.3f}  SE={s:.3f}  p={p:.3f}{star(p)}')
    print()

# ══════════════════════════════════════════════════════════════════════════════
print('='*70)
print('TEST 3: MEDIATION — does religious-leader trust mediate police/court decline?')
print('  Step 1: X → M  (Sharia → religious trust)')
print('  Step 2: X → Y  (Sharia → police/court trust, no mediator)')
print('  Step 3: X + M → Y  (does β_X shrink when controlling for M?)')
print('  Under substitution: β(sharia→M) > 0, inclusion of M shrinks β(sharia→Y)')
print('='*70)
print()

# Step 1: X → M
res_m = rd(r2_raw, 'rel_trust', bw=BW)
if res_m and 'sharia' in res_m:
    c, s, p = res_m['sharia']['coef'], res_m['sharia']['se'], res_m['sharia']['pval']
    print(f'Step 1 | X → religious trust:  β = {c:+.3f}  SE={s:.3f}  p={p:.3f}{star(p)}  N={res_m["n"]}')
else:
    print('Step 1 failed')
print()

for outcome, label in [('q43i','Trust: police'), ('q43j','Trust: courts')]:
    # Step 2: X → Y (no M)
    r2_m = rd(r2_raw, outcome, bw=BW)
    # Step 3: X + M → Y
    r3_m = rd(r2_raw, outcome, extra_vars=['rel_trust'],
              formula_suffix=' + rel_trust')
    if r2_m and r3_m and 'sharia' in r2_m and 'sharia' in r3_m:
        c2, p2 = r2_m['sharia']['coef'], r2_m['sharia']['pval']
        c3, p3 = r3_m['sharia']['coef'], r3_m['sharia']['pval']
        pct = (c2 - c3) / c2 * 100 if abs(c2) > 1e-8 else np.nan
        print(f'{label}:')
        print(f'  Step 2 | β(sharia, no M)   = {c2:+.3f}  p={p2:.3f}{star(p2)}  N={r2_m["n"]}')
        print(f'  Step 3 | β(sharia, with M)  = {c3:+.3f}  p={p3:.3f}{star(p3)}  N={r3_m["n"]}')
        print(f'  Mediated fraction: {pct:.1f}%  '
              f'({"→ mediation present" if abs(pct) > 5 else "→ no meaningful mediation"})')
        # Also report M coefficient
        if 'rel_trust' in r3_m:
            cm, pm = r3_m['rel_trust']['coef'], r3_m['rel_trust']['pval']
            print(f'  β(rel_trust in step 3)  = {cm:+.3f}  p={pm:.3f}{star(pm)}')
        print()

# ══════════════════════════════════════════════════════════════════════════════
print('='*70)
print('TEST 4: EXTENDED OUTCOMES — broader state legitimacy')
print('  Are democracy/governance attitudes also affected at the boundary?')
print('  If only police/courts affected → enforcement-specific, not general anti-state')
print('='*70)
print()

ext_outcomes = [
    ('q43i',      'Trust: police (main)'),
    ('q43j',      'Trust: courts (main)'),
    ('trust_pres','Trust: president'),
    ('trust_parl','Trust: parliament'),
    ('dem_supply','Democracy supply (q47)'),
    ('govt_perf', 'Govt performance (q45a)'),
    ('rel_trust', 'Trust: relig. leaders (mediator)'),
]
print(f'  {"Outcome":<32}  {"N":>5}  {"β_RD":>8}  {"SE":>7}  {"p":>8}')
print('  ' + '-'*65)
for col, label in ext_outcomes:
    res = rd(r2_raw, col, bw=BW)
    if res and 'sharia' in res:
        c, s, p = res['sharia']['coef'], res['sharia']['se'], res['sharia']['pval']
        print(f'  {label:<32}  {res["n"]:>5}  {c:>+8.3f}  {s:>7.3f}  {p:>7.3f}{star(p)}')
    else:
        print(f'  {label:<32}  {"—":>5}')
print()

# ══════════════════════════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════════════════════════
C_MAIN = '#3d7ab5'
C_PALE = '#9bbbd6'
C_RED  = '#c0392b'
C_GREY = '#888888'

fig = plt.figure(figsize=(13, 9))
gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.38,
                        left=0.07, right=0.97, top=0.93, bottom=0.07)

# ── Panel A: Muslim heterogeneity ─────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
groups = ['Non-Muslim\n(police)', 'Muslim\n(police)',
          'Non-Muslim\n(courts)', 'Muslim\n(courts)']
coefs, ses, pvals = [], [], []
for outcome in ['q43i', 'q43j']:
    d_bw = r2_raw.dropna(subset=[outcome, 'dist_km', 'sharia', 'muslim'])
    d_bw = d_bw[d_bw['dist_km'].abs() <= BW]
    for grp in [0, 1]:
        dg = d_bw[d_bw['muslim'] == grp]
        rg = rd(dg, outcome)
        if rg and 'sharia' in rg:
            coefs.append(rg['sharia']['coef'])
            ses.append(rg['sharia']['se'])
            pvals.append(rg['sharia']['pval'])
        else:
            coefs.append(np.nan); ses.append(np.nan); pvals.append(np.nan)

colors_a = [C_GREY, C_MAIN, C_GREY, C_MAIN]
for i, (c, s, p, lbl, col) in enumerate(zip(coefs, ses, pvals, groups, colors_a)):
    if not np.isnan(c):
        ci = 1.96 * s
        ax_a.barh(i, c, color=col, alpha=0.75, height=0.55)
        ax_a.errorbar(c, i, xerr=ci, fmt='none', color='#333333', lw=1.5, capsize=4)
        ax_a.text(c + (ci + 0.03) * np.sign(c), i,
                  f'β={c:+.2f}{star(p)}', va='center', ha='left' if c < 0 else 'right',
                  fontsize=8.5)
ax_a.axvline(0, color='#333333', lw=0.8, ls='--', alpha=0.6)
ax_a.set_yticks(range(len(groups)))
ax_a.set_yticklabels(groups, fontsize=9)
ax_a.set_xlabel('β (Sharia effect, 0–3 scale)', fontsize=9)
ax_a.set_title('A. Heterogeneity by Religion\n(Muslim vs. non-Muslim)', fontsize=10, fontweight='bold')
ax_a.spines[['top','right']].set_visible(False)

# ── Panel B: Sharia salience moderation ──────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
groups_b = ['Not concerned\n(police)', 'Sharia concern\n(police)',
            'Not concerned\n(courts)', 'Sharia concern\n(courts)']
coefs_b, ses_b, pvals_b = [], [], []
for outcome in ['q43i', 'q43j']:
    d_bw = r2_raw.dropna(subset=[outcome, 'dist_km', 'sharia', 'sharia_concern'])
    d_bw = d_bw[d_bw['dist_km'].abs() <= BW]
    for grp in [0, 1]:
        dg = d_bw[d_bw['sharia_concern'] == grp]
        rg = rd(dg, outcome)
        if rg and 'sharia' in rg:
            coefs_b.append(rg['sharia']['coef'])
            ses_b.append(rg['sharia']['se'])
            pvals_b.append(rg['sharia']['pval'])
        else:
            coefs_b.append(np.nan); ses_b.append(np.nan); pvals_b.append(np.nan)

colors_b = [C_GREY, C_RED, C_GREY, C_RED]
for i, (c, s, p, lbl, col) in enumerate(zip(coefs_b, ses_b, pvals_b, groups_b, colors_b)):
    if not np.isnan(c):
        ci = 1.96 * s
        ax_b.barh(i, c, color=col, alpha=0.75, height=0.55)
        ax_b.errorbar(c, i, xerr=ci, fmt='none', color='#333333', lw=1.5, capsize=4)
        ax_b.text(c + (ci + 0.03) * np.sign(c), i,
                  f'β={c:+.2f}{star(p)}', va='center', ha='left' if c < 0 else 'right',
                  fontsize=8.5)
ax_b.axvline(0, color='#333333', lw=0.8, ls='--', alpha=0.6)
ax_b.set_yticks(range(len(groups_b)))
ax_b.set_yticklabels(groups_b, fontsize=9)
ax_b.set_xlabel('β (Sharia effect, 0–3 scale)', fontsize=9)
ax_b.set_title('B. Sharia Salience Moderation\n(q44 = Sharia named as top problem)', fontsize=10, fontweight='bold')
ax_b.spines[['top','right']].set_visible(False)

# ── Panel C: Mediation path diagram style ────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])
med_labels = ['Religious trust\n(mediator)', 'Police trust\n(no mediator)',
              'Police trust\n(with relig. trust)', 'Courts trust\n(no mediator)',
              'Courts trust\n(with relig. trust)']
med_coefs, med_ses, med_pvals, med_colors = [], [], [], []

res_m = rd(r2_raw, 'rel_trust')
if res_m and 'sharia' in res_m:
    med_coefs.append(res_m['sharia']['coef'])
    med_ses.append(res_m['sharia']['se'])
    med_pvals.append(res_m['sharia']['pval'])
    med_colors.append('#999999')

for outcome in ['q43i', 'q43j']:
    r2_m = rd(r2_raw, outcome)
    r3_m = rd(r2_raw, outcome, extra_vars=['rel_trust'], formula_suffix=' + rel_trust')
    for res, col in [(r2_m, C_MAIN), (r3_m, '#5b9bd5')]:
        if res and 'sharia' in res:
            med_coefs.append(res['sharia']['coef'])
            med_ses.append(res['sharia']['se'])
            med_pvals.append(res['sharia']['pval'])
            med_colors.append(col)
        else:
            med_coefs.append(np.nan); med_ses.append(np.nan)
            med_pvals.append(np.nan); med_colors.append(col)

for i, (c, s, p, lbl, col) in enumerate(
        zip(med_coefs, med_ses, med_pvals, med_labels, med_colors)):
    if not np.isnan(c):
        ci = 1.96 * s
        ax_c.barh(i, c, color=col, alpha=0.75, height=0.55)
        ax_c.errorbar(c, i, xerr=ci, fmt='none', color='#333333', lw=1.5, capsize=4)
        ax_c.text(max(c + ci, 0) + 0.02, i,
                  f'β={c:+.2f}{star(p)}', va='center', ha='left', fontsize=8.5)
ax_c.axvline(0, color='#333333', lw=0.8, ls='--', alpha=0.6)
ax_c.set_yticks(range(len(med_labels)))
ax_c.set_yticklabels(med_labels, fontsize=9)
ax_c.set_xlabel('β (Sharia effect, 0–3 scale)', fontsize=9)
ax_c.set_title('C. Mediation: Religious Leader Trust\n(does M mediate X→Y?)', fontsize=10, fontweight='bold')
ax_c.spines[['top','right']].set_visible(False)

# ── Panel D: Extended outcomes ────────────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])
ext_cols = [
    ('q43i',      'Trust: police',        C_MAIN, True),
    ('q43j',      'Trust: courts',        C_MAIN, True),
    ('trust_pres','Trust: president',     C_PALE, False),
    ('trust_parl','Trust: parliament',    C_PALE, False),
    ('dem_supply','Democracy supply',     C_PALE, False),
    ('govt_perf', 'Govt performance',     C_PALE, False),
    ('rel_trust', 'Trust: relig. leaders',C_PALE, False),
]
ext_c, ext_s, ext_p, ext_l, ext_col, ext_main = [], [], [], [], [], []
for col, label, color, is_main in ext_cols:
    res = rd(r2_raw, col)
    if res and 'sharia' in res:
        ext_c.append(res['sharia']['coef'])
        ext_s.append(res['sharia']['se'])
        ext_p.append(res['sharia']['pval'])
        ext_l.append(label)
        ext_col.append(color)
        ext_main.append(is_main)
    else:
        ext_c.append(np.nan); ext_s.append(np.nan); ext_p.append(np.nan)
        ext_l.append(label); ext_col.append(color); ext_main.append(is_main)

for i, (c, s, p, lbl, col) in enumerate(zip(ext_c, ext_s, ext_p, ext_l, ext_col)):
    if not np.isnan(c):
        ci = 1.96 * s
        ax_d.barh(i, c, color=col, alpha=0.75, height=0.55)
        ax_d.errorbar(c, i, xerr=ci, fmt='none', color='#333333', lw=1.5, capsize=4)
        ax_d.text(c - 0.02 if c < 0 else c + ci + 0.02, i,
                  f'β={c:+.2f}{star(p)}', va='center',
                  ha='right' if c < 0 else 'left', fontsize=8.5)
ax_d.axvline(0, color='#333333', lw=0.8, ls='--', alpha=0.6)
ax_d.set_yticks(range(len(ext_l)))
ax_d.set_yticklabels(ext_l, fontsize=9)
ax_d.set_xlabel('β (Sharia effect, 0–3 scale)', fontsize=9)
ax_d.set_title('D. Extended Outcomes\n(enforcement-specific vs. general anti-state)', fontsize=10, fontweight='bold')
ax_d.spines[['top','right']].set_visible(False)

fig.suptitle('Mechanism Tests: Sharia Law and Institutional Trust\n'
             'Spatial RD, BW = 150 km, HC3 SE  |  *** p<0.001  ** p<0.01  * p<0.05  † p<0.10',
             fontsize=11, fontweight='bold')

plt.savefig('sharia_mechanism_fig.png', dpi=200, bbox_inches='tight', facecolor='white')
print('\nSaved sharia_mechanism_fig.png')
