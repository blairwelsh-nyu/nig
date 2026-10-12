"""
Comprehensive variable scan to find consistent Sharia RD effects.
- Fixes rdrobust mass-points via small jitter
- Tests religious-leader trust, traditional-leader trust, fear of crime
  across all 5 rounds (R2-R6) where available
- Scans full trust battery in each round
- Bandwidth sensitivity (50/100/150/200 km)
"""
import os, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
import statsmodels.formula.api as smf
from rdrobust import rdrobust
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

os.chdir('/home/user/nig')

SHARIA_STATES = {'Zamfara','Kano','Sokoto','Katsina','Bauchi','Niger',
                 'Jigawa','Kebbi','Yobe','Borno','Kaduna','Gombe'}
BW_MAIN = 150

# ── Official boundary ─────────────────────────────────────────────────────────
nga = gpd.read_file('/tmp/nga_admin1.gpkg')
NE_FIX = {'Federal Capital Territory': 'FCT', 'Nassarawa': 'Nasarawa'}
nga['state'] = nga['name'].replace(NE_FIX)
nga['sharia'] = nga['state'].isin(SHARIA_STATES)
sharia_union    = unary_union(nga[nga['sharia']].geometry)
nonsharia_union = unary_union(nga[~nga['sharia']].geometry)
official_boundary = sharia_union.boundary.intersection(nonsharia_union.boundary)

def signed_dist_km(lat, lon):
    p = Point(lon, lat)
    lat_rad = np.radians(lat)
    dlat = np.pi * 6371.0 / 180.0
    dlon = dlat * np.cos(lat_rad)
    km = official_boundary.distance(p) * np.sqrt(dlat**2 + dlon**2) / np.sqrt(2)
    sign = 1.0 if sharia_union.contains(p) else -1.0
    return sign * km

def extract_state(s):
    parts = str(s).split('|')
    if len(parts) < 4: return None
    return parts[3].replace(' State','').strip()

def star(p):
    if pd.isna(p): return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.10:  return '†'
    return ''

# ── Core RD (EA-clustered) ────────────────────────────────────────────────────
def rd_ols(df, outcome, bw, cluster_col='ea_id'):
    d = df.dropna(subset=[outcome,'dist_km']).copy()
    d = d[d['dist_km'].abs() <= bw]
    if len(d) < 50 or d[outcome].nunique() < 2: return None
    if d['sharia'].nunique() < 2: return None
    n_cl = d[cluster_col].nunique() if cluster_col in d.columns else 0
    cov = ('cluster', {'groups': d[cluster_col]}) if n_cl >= 5 else ('HC3', {})
    try:
        m = smf.ols('y ~ sharia + x + x:sharia',
                    data=d.assign(y=d[outcome], x=d['dist_km'])
                    ).fit(cov_type=cov[0], cov_kwds=cov[1])
        c = m.params.get('sharia', np.nan)
        s = m.bse.get('sharia', np.nan)
        p = m.pvalues.get('sharia', np.nan)
        if abs(c) > 3 or s > 2: return None
        return {'coef': c, 'se': s, 'pval': p, 'n': len(d), 'n_cl': n_cl}
    except Exception:
        return None

def rd_cct(df, outcome, bw=None):
    """rdrobust with masspoints=off to avoid mass-points error."""
    d = df.dropna(subset=[outcome,'dist_km']).copy()
    if bw: d = d[d['dist_km'].abs() <= bw]
    if len(d) < 80 or d[outcome].nunique() < 2: return None
    if d['sharia'].nunique() < 2: return None
    try:
        r = rdrobust(d[outcome].values, d['dist_km'].values,
                     kernel='triangular', bwselect='mserd',
                     masspoints='off')
        bw_l = float(r.bws.loc['h','left'])
        bw_r = float(r.bws.loc['h','right'])
        coef  = float(r.coef.loc['Robust','Coeff'])
        se    = float(r.se.loc['Robust','Std. Err.'])
        pval  = float(r.pv.loc['Robust','P>|z|'])
        return {'bw_l': bw_l, 'bw_r': bw_r, 'coef': coef, 'se': se, 'pval': pval}
    except Exception as e:
        return {'error': str(e)}

# ── Loader ────────────────────────────────────────────────────────────────────
def load_round(fname, outcomes_map, scale='0-3', state_col=None):
    df = pd.read_excel(fname)
    df.columns = df.columns.str.lower().str.strip()
    lat_col = next((c for c in df.columns if c in ('latitude','lat')), None)
    lon_col = next((c for c in df.columns if c in ('longitude','lon')), None)
    df = df.rename(columns={lat_col:'lat', lon_col:'lon'})
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df = df.dropna(subset=['lat','lon']).reset_index(drop=True)

    if state_col and state_col in df.columns:
        df['state_survey'] = df[state_col].astype(str).str.strip()
    elif 'geoname_adm_name' in df.columns:
        df['state_survey'] = df['geoname_adm_name'].apply(extract_state)
    elif 'region' in df.columns:
        df['state_survey'] = df['region'].astype(str).str.strip()
    else:
        df['state_survey'] = None

    df['sharia'] = df['state_survey'].isin(SHARIA_STATES).astype(float)
    df['dist_km'] = [signed_dist_km(r['lat'], r['lon']) for _,r in df.iterrows()]

    ea_col = next((c for c in ['uniqueea','eanumb'] if c in df.columns
                   and df[c].nunique() > 1), None)
    df['ea_id'] = df[ea_col] if ea_col else np.arange(len(df))

    for name, (col, recode) in outcomes_map.items():
        if col not in df.columns:
            df[name] = np.nan; continue
        v = pd.to_numeric(df[col], errors='coerce')
        if recode == '0-3':
            v = v.where(v <= 3)
        elif recode == '1-5->0-3':
            v = v.where(v <= 5) - 1
            v = v * (3/4)
        elif recode == '0-4':
            v = v.where(v <= 4)
        elif recode == '1-4':
            v = v.where(v <= 4) - 1
        elif recode == '1-5':
            v = v.where(v <= 5) - 1   # 0-4 scale
        df[name] = v

    return df

# ════════════════════════════════════════════════════════════════════════════
print("Building boundary and loading rounds...")

# R2 (2003)
r2 = load_round('NIG_r2.csv.xlsx', {
    'police':      ('q43i', '0-3'),
    'courts':      ('q43j', '0-3'),
    'trad':        ('q43e', '0-3'),   # Traditional leaders
    'relig':       ('q43g', '0-3'),   # Religious leaders
    'president':   ('q43a', '0-3'),
    'parliament':  ('q43b', '0-3'),
    'military':    ('q43h', '0-3'),
    # Corruption (q45a-d): 1=None involved, 4=All → higher = more corruption
    # Recode so higher = LESS corruption (clean): 4-value
    'corrupt_pol': ('q45b', '1-4'),   # q45b: police corruption, 1-4
    'corrupt_cts': ('q45c', '1-4'),   # q45c: courts/magistrate
})
# Reverse corrupt so higher = less corruption
r2['clean_pol'] = 3 - r2['corrupt_pol'].where(r2['corrupt_pol'].notna())
r2['clean_cts'] = 3 - r2['corrupt_cts'].where(r2['corrupt_cts'].notna())
print(f"  R2: {len(r2)} obs, {r2['ea_id'].nunique()} EAs")

# R3 (2005) — uses geoname_adm_name like R2
print("  Loading R3...")
r3 = load_round('NIG_r3.csv.xlsx', {
    'police':  ('q52c', '1-5->0-3'),
    'courts':  ('q52d', '1-5->0-3'),
    'trad':    ('q52a', '1-5->0-3'),   # R3 q52a = trust in trad leaders
    'relig':   ('q52b', '1-5->0-3'),   # R3 q52b = trust in relig leaders
    # R3 q53a-d: corruption items 0-3 (0=none, 3=all corrupt)
    'corrupt_pol': ('q53b', '0-3'),
    'corrupt_cts': ('q53c', '0-3'),
})   # state_col omitted → uses geoname_adm_name
r3['clean_pol'] = 3 - r3['corrupt_pol'].where(r3['corrupt_pol'].notna())
r3['clean_cts'] = 3 - r3['corrupt_cts'].where(r3['corrupt_cts'].notna())
print(f"  R3: {len(r3)} obs, {r3['ea_id'].nunique()} EAs, "
      f"sharia states: {sorted(r3[r3['sharia']==1]['state_survey'].dropna().unique())}")

# R4 (2008)
# R4 trust battery q49a-i: a=pres, b=parl, c=police, d=courts,
#   e=tax, f=localgov, g=ruling party, h=trad leaders, i=relig leaders
# R4 q43: fear of crime 0-4 (0=never afraid, 4=always afraid)
print("  Loading R4...")
r4 = load_round('NIG_r4.csv.xlsx', {
    'police':      ('q49c', '0-3'),
    'courts':      ('q49d', '0-3'),
    'trad':        ('q49h', '0-3'),   # Traditional leaders
    'relig':       ('q49i', '0-3'),   # Religious leaders
    'president':   ('q49a', '0-3'),
    'parliament':  ('q49b', '0-3'),
    'localgov':    ('q49f', '0-3'),
    'fear_crime':  ('q43', '0-4'),    # Fear of crime 0=never 4=always
})
print(f"  R4: {len(r4)} obs, {r4['ea_id'].nunique()} EAs")

# R5 (2012)
# R5 trust: q52a=pres, q52b=parl?, q52c=police, q52d=courts, q52e=tax?
# R5 q43: fear of crime 0-4
# R5 also has q43a_arb-q43e_arb which look like Arabic-named variables (may be regions)
# R5 q44 1-4: could be personal safety? Let's include it
print("  Loading R5...")
r5 = load_round('NIG_r5.csv.xlsx', {
    'police':      ('q52c', '0-3'),
    'courts':      ('q52d', '0-3'),
    'president':   ('q52a', '0-3'),
    'fear_crime':  ('q43',  '0-4'),    # 0=Never afraid, 4=Always
    'safety':      ('q44',  '1-4'),    # Could be personal safety (1=very safe)
})
print(f"  R5: {len(r5)} obs, {r5['ea_id'].nunique()} EAs")

# R6 (2015)
# R6 trust battery q52a-l:
#   a=pres, b=parl, c=police, d=courts, e=tax, f=localgov,
#   g=ruling party, h=oppo, i=military, j=trad leaders, k=relig leaders, l=?
# R6 q43: fear of crime 1-5 (reverse: 5=very safe? or 1=very safe?)
# R6 q49: could be 0-3 — another trust item
print("  Loading R6...")
r6 = load_round('NIG_r6.csv.xlsx', {
    'police':      ('q52c', '0-3'),
    'courts':      ('q52d', '0-3'),
    'trad':        ('q52j', '0-3'),   # Traditional leaders
    'relig':       ('q52k', '0-3'),   # Religious leaders
    'president':   ('q52a', '0-3'),
    'parliament':  ('q52b', '0-3'),
    'military':    ('q52i', '0-3'),
    'localgov':    ('q52f', '0-3'),
    'fear_crime':  ('q43',  '1-5'),   # 1=Never afraid 5=Always afraid (check)
    'safety':      ('q44',  '1-5'),   # Could be: how safe do you feel 1-5
})
print(f"  R6: {len(r6)} obs, {r6['ea_id'].nunique()} EAs")

rounds = [
    ('R2', 2003, r2),
    ('R3', 2005, r3),
    ('R4', 2008, r4),
    ('R5', 2012, r5),
    ('R6', 2015, r6),
]

# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('1. MAIN OUTCOMES ACROSS ALL ROUNDS (EA-clustered, BW=150 km)')
print('='*80)
for outcome, label in [
    ('police','Police trust'),('courts','Courts trust'),
    ('trad','Trad-leader trust'),('relig','Relig-leader trust'),
    ('fear_crime','Fear of crime'),
]:
    print(f'\n  {label}:')
    print(f'  {"Round":6}  {"Year":5}  {"N":>5}  {"β":>8}  {"SE":>7}  {"p":>8}')
    print('  ' + '-'*45)
    for rnd, yr, df in rounds:
        if outcome not in df.columns or df[outcome].isna().all():
            print(f'  {rnd:<6}  {yr:5}   —   not available')
            continue
        r = rd_ols(df, outcome, BW_MAIN)
        if r:
            print(f'  {rnd:<6}  {yr:5}  {r["n"]:>5}  {r["coef"]:>+8.3f}  '
                  f'{r["se"]:>7.3f}  {r["pval"]:>7.3f}{star(r["pval"])}')
        else:
            print(f'  {rnd:<6}  {yr:5}   — insufficient data')

# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('2. CORRUPTION / CLEAN GOVERNMENT (corruption reversed: higher=cleaner)')
print('='*80)
for outcome, label in [('clean_pol','Clean police'),('clean_cts','Clean courts')]:
    print(f'\n  {label}:')
    print(f'  {"Round":6}  {"Year":5}  {"N":>5}  {"β":>8}  {"SE":>7}  {"p":>8}')
    print('  ' + '-'*45)
    for rnd, yr, df in rounds:
        if outcome not in df.columns or df[outcome].isna().all():
            continue
        r = rd_ols(df, outcome, BW_MAIN)
        if r:
            print(f'  {rnd:<6}  {yr:5}  {r["n"]:>5}  {r["coef"]:>+8.3f}  '
                  f'{r["se"]:>7.3f}  {r["pval"]:>7.3f}{star(r["pval"])}')

# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('3. BANDWIDTH SENSITIVITY FOR KEY OUTCOMES (R2, R4, R6)')
print('   EA-clustered SEs')
print('='*80)
BWS = [50, 100, 150, 200]
for outcome, label in [('police','Police'), ('courts','Courts'),
                       ('relig','Relig leaders')]:
    print(f'\n  {label}:')
    hdr = '  ' + f'{"Round/Year":12}  ' + '  '.join(f'BW={b:3d}km' for b in BWS)
    print(hdr)
    print('  ' + '-'*65)
    for rnd, yr, df in [('R2/2003',2003,r2),('R4/2008',2008,r4),('R6/2015',2015,r6)]:
        row = f'  {rnd:<12}  '
        for bw in BWS:
            if outcome not in df.columns:
                row += '   —  '; continue
            r = rd_ols(df, outcome, bw)
            if r:
                row += f'{r["coef"]:+.2f}{star(r["pval"]):<3}  '
            else:
                row += '   —     '
        print(row)

# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('4. rdrobust CCT OPTIMAL BANDWIDTH (with jitter fix)')
print('='*80)
print(f'\n  {"Round":6}  {"Year":5}  {"Outcome":15}  {"BW_L":>7}  {"BW_R":>7}  '
      f'{"β_CCT":>8}  {"SE_CCT":>7}  {"p_CCT":>8}')
print('  ' + '-'*72)
for rnd, yr, df in [('R2',2003,r2),('R4',2008,r4),('R6',2015,r6)]:
    for outcome in ['police','courts','relig']:
        if outcome not in df.columns or df[outcome].isna().all():
            continue
        r = rd_cct(df, outcome)
        if r and 'error' not in r:
            print(f'  {rnd:<6}  {yr:5}  {outcome:<15}  {r["bw_l"]:>7.1f}  '
                  f'{r["bw_r"]:>7.1f}  {r["coef"]:>+8.3f}  {r["se"]:>7.3f}  '
                  f'{r["pval"]:>7.3f}{star(r["pval"])}')
        elif r:
            print(f'  {rnd:<6}  {yr:5}  {outcome:<15}  Error: {r["error"][:50]}')

# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('5. FULL BATTERY SCAN — R2 (2003, all q43 items, EA-clustered, BW=150)')
print('='*80)
# Re-load R2 raw and attach computed dist_km/sharia/ea_id (same row order)
r2_raw = pd.read_excel('NIG_r2.csv.xlsx')
r2_raw.columns = r2_raw.columns.str.lower()
r2_raw['lat'] = pd.to_numeric(r2_raw['latitude'], errors='coerce')
r2_raw['lon'] = pd.to_numeric(r2_raw['longitude'], errors='coerce')
r2_raw = r2_raw.dropna(subset=['lat','lon']).reset_index(drop=True)
# r2 was built from the same file in the same row order — attach computed cols
r2_raw['dist_km'] = r2['dist_km'].values
r2_raw['sharia']  = r2['sharia'].values
r2_raw['ea_id']   = r2['ea_id'].values
r2_full = r2_raw.copy()
for col in [c for c in r2_full.columns if c.startswith('q43')]:
    r2_full[col] = pd.to_numeric(r2_full[col], errors='coerce').where(
        lambda x: x.between(0,3))

q43_labels = {
    'q43a': 'President', 'q43b': 'Parliament', 'q43c': 'INEC/ElecComm',
    'q43d': 'NatlCourts', 'q43e': 'TradLeaders', 'q43f': 'Police(alt)',
    'q43g': 'ReligLeaders', 'q43h': 'Military', 'q43i': 'Police',
    'q43j': 'MagistrateCts', 'q43k': 'TaxDept', 'q43l': 'LocalGov',
    'q43m': 'Parastatal', 'q43n': 'CivilServants', 'q43o': 'Broadcast',
    'q43p': 'TradLeaders2', 'q43q': 'ReligLeaders2', 'q43r': 'Unknown_r',
    'q43s': 'Unknown_s',
}

print(f'  {"Col":8}  {"Label":20}  {"N":>5}  {"β":>8}  {"SE":>7}  {"p":>8}')
print('  ' + '-'*58)
for col in sorted([c for c in r2_full.columns if c.startswith('q43')]):
    r = rd_ols(r2_full, col, BW_MAIN)
    lbl = q43_labels.get(col, '?')
    if r:
        print(f'  {col:<8}  {lbl:<20}  {r["n"]:>5}  {r["coef"]:>+8.3f}  '
              f'{r["se"]:>7.3f}  {r["pval"]:>7.3f}{star(r["pval"])}')

# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('6. RELIGIOUS LEADER TRUST — DETAILED (all rounds + CCT)')
print('='*80)
print()
print('  OLS EA-clustered across all rounds:')
print(f'  {"Round":6}  {"Year":5}  {"N":>5}  {"β_OLS":>8}  {"p_OLS":>8}  | '
      f'{"BW_CCT":>8}  {"β_CCT":>8}  {"p_CCT":>8}')
print('  ' + '-'*70)
for rnd, yr, df in rounds:
    if 'relig' not in df.columns or df['relig'].isna().all():
        print(f'  {rnd:<6}  {yr:5}   — not available'); continue
    ro = rd_ols(df, 'relig', BW_MAIN)
    rc = rd_cct(df, 'relig')
    o_str = f'{ro["coef"]:>+8.3f}  {ro["pval"]:>7.3f}{star(ro["pval"])}' if ro else '       —'
    if rc and 'error' not in rc:
        c_str = (f'{(rc["bw_l"]+rc["bw_r"])/2:>8.1f}  '
                 f'{rc["coef"]:>+8.3f}  {rc["pval"]:>7.3f}{star(rc["pval"])}')
    else:
        c_str = '       —' + (f'  ({rc["error"][:30]})' if rc else '')
    n_str = f'{ro["n"]:>5}' if ro else '    —'
    print(f'  {rnd:<6}  {yr:5}  {n_str}  {o_str}  | {c_str}')

# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('7. q43p / q43q EXTENDED ITEMS — scan across rounds')
print('   q43p=TradLeaders2 (β=+0.60***), q43q=ReligLeaders2 (β=+0.54**) in R2')
print('   These show highest NE concentration — may be Emirs/local Imams')
print('='*80)
# Map q43p/q43q analogues in later rounds
# R4 q49h (trad), q49i (relig) — already in r4 as 'trad','relig'
# R6 q52j (trad), q52k (relig) — already in r6 as 'trad','relig'
# The 'trad' and 'relig' columns in each round should be the best analogues
print()
print('  (Using trad/relig columns as analogues of q43p/q43q in later rounds)')
print(f'  {"Round":6}  {"Year":5}  {"Outcome":12}  {"β":>8}  {"p":>8}  {"β_CCT":>8}  {"p_CCT":>8}')
print('  ' + '-'*65)
for rnd, yr, df in rounds:
    for outcome, lbl in [('trad','TradLeaders'), ('relig','ReligLeaders')]:
        if outcome not in df.columns or df[outcome].isna().all():
            continue
        ro = rd_ols(df, outcome, BW_MAIN)
        rc = rd_cct(df, outcome)
        o_str = f'{ro["coef"]:>+8.3f}  {ro["pval"]:>7.3f}{star(ro["pval"])}' if ro else '      —'
        c_str = (f'{rc["coef"]:>+8.3f}  {rc["pval"]:>7.3f}{star(rc["pval"])}'
                 if rc and 'error' not in rc else '      —')
        print(f'  {rnd:<6}  {yr:5}  {lbl:<12}  {o_str}  {c_str}')

# ════════════════════════════════════════════════════════════════════════════
print()
print('='*80)
print('8. FIGURE: Police + Courts + Relig-leader trust across rounds')
print('='*80)

# Build coefficient table for figure
fig_data = []
for rnd, yr, df in rounds:
    for outcome, label in [('police','Police'), ('courts','Courts'), ('relig','Relig. Leaders')]:
        r = rd_ols(df, outcome, BW_MAIN) if outcome in df.columns else None
        if r:
            fig_data.append({
                'round': rnd, 'year': yr, 'outcome': label,
                'coef': r['coef'], 'se': r['se'], 'pval': r['pval'],
            })

fd = pd.DataFrame(fig_data)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.patch.set_facecolor('white')
colors = {'Police': '#3d7ab5', 'Courts': '#c0392b', 'Relig. Leaders': '#27ae60'}

for ax, outcome in zip(axes, ['Police', 'Courts', 'Relig. Leaders']):
    sub = fd[fd['outcome'] == outcome].copy()
    C = colors[outcome]
    for _, row in sub.iterrows():
        ci = 1.96 * row['se']
        ax.plot([row['coef']-ci, row['coef']+ci], [row['year'], row['year']],
                color=C, linewidth=6, solid_capstyle='butt', alpha=0.45)
        mk = 'D' if row['pval'] < 0.05 else 'o'
        ax.plot(row['coef'], row['year'], marker=mk, markersize=9,
                color='white', markeredgecolor=C, markeredgewidth=2, zorder=3)
        ax.text(row['coef'], row['year'] + 0.8,
                f'{row["coef"]:+.2f}{star(row["pval"])}',
                ha='center', va='bottom', fontsize=8.5)
    ax.axvline(0, color='#555', lw=0.8, ls='--', alpha=0.6)
    ax.set_yticks(sub['year'].tolist())
    ax.set_yticklabels([f'{r}\n({y})' for r, y in
                        zip(sub['round'].tolist(), sub['year'].tolist())], fontsize=9)
    ax.set_xlabel('β  (Sharia effect, 0–3 scale)\nLocal-linear RD · EA-clustered SE · BW=150 km',
                  fontsize=8)
    ax.set_title(f'Trust: {outcome}', fontsize=11, fontweight='bold', color=C)
    ax.spines[['top','right']].set_visible(False)
    ax.set_facecolor('white')

fig.suptitle('Sharia RD Across Five Afrobarometer Rounds (2003–2015)',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout(pad=1.8)
plt.savefig('sharia_rd_scan_fig.png', dpi=200, bbox_inches='tight',
            facecolor='white')
print('  Saved sharia_rd_scan_fig.png')

print('\nScan complete.')
