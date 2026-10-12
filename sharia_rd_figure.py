"""Make a 2-panel figure for the Sharia spatial RD results."""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import LineString, Point
from sklearn.neighbors import BallTree
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')

os.chdir('/home/user/nig')

# ── Reproduce boundary and data from sharia_rd.py ──
SHARIA_BOUNDARY_COORDS = [
    (3.40, 11.85),(3.90, 11.20),(4.35, 10.95),(4.85, 10.65),(5.25, 10.40),
    (5.65, 10.10),(6.05, 9.85),(6.55, 9.55),(7.00, 9.35),(7.45, 9.15),
    (7.95, 9.40),(8.50, 9.60),(9.00, 9.80),(9.50, 9.75),(9.95, 9.65),
    (10.40, 9.35),(10.95, 9.10),(11.45, 9.00),(12.00, 9.00),(12.60, 9.10),
    (13.10, 9.20),(13.70, 8.80),
]
boundary_line = LineString(SHARIA_BOUNDARY_COORDS)
SHARIA_STATES = {'Zamfara','Kano','Sokoto','Katsina','Bauchi','Niger',
                 'Jigawa','Kebbi','Yobe','Borno','Kaduna','Gombe'}

acled = pd.read_csv('nigeria_acled.csv')
acled['event_date'] = pd.to_datetime(acled['event_date'], errors='coerce')
state_cents = acled.groupby('admin1')[['latitude','longitude']].mean().reset_index()
state_cents.columns = ['state','lat','lon']

def assign_state(lat_arr, lon_arr):
    cent_rad = np.radians(state_cents[['lat','lon']].values)
    tree = BallTree(cent_rad, metric='haversine')
    _, idx = tree.query(np.radians(np.column_stack([lat_arr, lon_arr])), k=1)
    return state_cents['state'].values[idx.flatten()]

def signed_dist_km(lat_arr, lon_arr, state_arr, boundary, sharia_states, earth_r=6371.0):
    n = len(lat_arr)
    dist = np.full(n, np.nan)
    for i in range(n):
        p = Point(lon_arr[i], lat_arr[i])
        lat_rad = np.radians(lat_arr[i])
        deg_lat_km = np.pi * earth_r / 180.0
        deg_lon_km = deg_lat_km * np.cos(lat_rad)
        deg_dist = boundary.distance(p)
        km_est = deg_dist * np.sqrt(deg_lat_km**2 + deg_lon_km**2) / np.sqrt(2)
        sign = 1.0 if state_arr[i] in sharia_states else -1.0
        dist[i] = sign * km_est
    return dist

r2 = pd.read_excel('NIG_r2.csv.xlsx')
r2.columns = r2.columns.str.lower().str.strip()
lat_col = next(c for c in r2.columns if 'lat' in c)
lon_col = next(c for c in r2.columns if 'lon' in c)
r2 = r2.rename(columns={lat_col:'lat', lon_col:'lon'})
r2['lat'] = pd.to_numeric(r2['lat'], errors='coerce')
r2['lon'] = pd.to_numeric(r2['lon'], errors='coerce')
r2 = r2.dropna(subset=['lat','lon'])
r2['state'] = assign_state(r2['lat'].values, r2['lon'].values)
r2['dist_km'] = signed_dist_km(r2['lat'].values, r2['lon'].values,
                                r2['state'].values, boundary_line, SHARIA_STATES)
r2['sharia'] = r2['state'].isin(SHARIA_STATES).astype(float)

# Clean trust_police
r2['q43i'] = pd.to_numeric(r2['q43i'], errors='coerce')
r2.loc[r2['q43i'] >= 7, 'q43i'] = np.nan

# ── Panel 1: binned scatter plot ──
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax1 = axes[0]
near = r2[r2['dist_km'].abs() <= 200].copy().dropna(subset=['q43i','dist_km'])
bins = np.arange(-200, 201, 25)
near['bin'] = pd.cut(near['dist_km'], bins=bins)
bdata = near.groupby('bin', observed=True).agg(
    y=('q43i', 'mean'),
    x=('dist_km', 'mean'),
    n=('q43i', 'count')
).dropna()
bdata = bdata[bdata['n'] >= 5]

col_ns = '#2166ac'; col_sh = '#d6604d'
left  = bdata[bdata['x'] < 0]
right = bdata[bdata['x'] > 0]
ax1.scatter(left['x'], left['y'],  s=left['n']*0.5,  color=col_ns, alpha=0.85,
            edgecolors='white', linewidths=0.4, zorder=3, label='Non-Sharia state')
ax1.scatter(right['x'], right['y'], s=right['n']*0.5, color=col_sh, alpha=0.85,
            edgecolors='white', linewidths=0.4, zorder=3, label='Sharia state')

# Fit lines on each side
for mask, color, side in [(bdata['x']<0, col_ns, 'left'), (bdata['x']>0, col_sh, 'right')]:
    sub = bdata[mask]
    if len(sub) >= 3:
        p = np.polyfit(sub['x'], sub['y'], 1)
        xs = np.linspace(sub['x'].min(), sub['x'].max(), 100)
        ax1.plot(xs, np.polyval(p, xs), color=color, linewidth=2, zorder=2)

ax1.axvline(0, color='black', linewidth=1.2, linestyle='--', alpha=0.7)
ax1.set_xlabel('Signed distance from Sharia boundary (km)', fontsize=11)
ax1.set_ylabel('Mean trust in police (0–3)', fontsize=11)
ax1.set_title('Trust in Police at the Sharia Boundary\n(Afrobarometer R2, 2003)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='upper left')
ax1.text(5, ax1.get_ylim()[0] + 0.01, 'Sharia→', fontsize=8, color=col_sh)
ax1.text(-10, ax1.get_ylim()[0] + 0.01, '←Non-Sharia', fontsize=8, color=col_ns, ha='right')
ax1.grid(axis='y', alpha=0.3)

# ── Panel 2: coefficient plot for R2 trust items ──
ax2 = axes[1]

def run_rd(df, outcome_col, bandwidth=150):
    d = df.copy()
    if outcome_col not in d.columns: return None
    d[outcome_col] = pd.to_numeric(d[outcome_col], errors='coerce')
    d.loc[d[outcome_col] < 0, outcome_col] = np.nan
    d.loc[d[outcome_col] >= 7, outcome_col] = np.nan
    d = d.dropna(subset=[outcome_col, 'dist_km', 'sharia'])
    d = d[d['dist_km'].abs() <= bandwidth]
    if len(d) < 50 or d[outcome_col].nunique() < 2: return None
    d['y'] = d[outcome_col]; d['x'] = d['dist_km']
    try:
        m = smf.ols('y ~ sharia + x + x:sharia', data=d).fit(cov_type='HC3')
    except: return None
    coef = m.params.get('sharia', np.nan)
    se   = m.bse.get('sharia', np.nan)
    pval = m.pvalues.get('sharia', np.nan)
    if abs(coef) > 10 or se > 10: return None
    return {'coef': coef, 'se': se, 'pval': pval, 'n': len(d)}

TRUST_MAP = {
    'Police':              'q43i',
    'Courts / Law':        'q43j',
    'President':           'q43a',
    'Parliament':          'q43b',
    'Local Govt':          'q43d',
    'Trad. Leaders':       'q43e',
    'Religious Leaders':   'q43g',
}

labels, coefs, ses, pvs = [], [], [], []
for label, col in TRUST_MAP.items():
    r = run_rd(r2, col, bandwidth=150)
    if r:
        labels.append(label)
        coefs.append(r['coef'])
        ses.append(r['se'])
        pvs.append(r['pval'])

y_pos = np.arange(len(labels))
coefs = np.array(coefs); ses = np.array(ses); pvs = np.array(pvs)
colors = ['#d6604d' if p < 0.05 else ('#fdae61' if p < 0.10 else '#aaaaaa') for p in pvs]

ax2.barh(y_pos, coefs, xerr=1.96*ses,
         color=colors, edgecolor='white', height=0.6,
         error_kw={'ecolor': '#333333', 'capsize': 4, 'linewidth': 1.2},
         zorder=3)
ax2.axvline(0, color='black', linewidth=1.0, linestyle='--', alpha=0.7)
ax2.set_yticks(y_pos); ax2.set_yticklabels(labels, fontsize=10)
ax2.set_xlabel('β (Sharia state effect on trust, 0–3 scale)', fontsize=11)
ax2.set_title('Institutional Trust Discontinuity at Sharia Boundary\n(R2, BW=150 km, HC3 SE)', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
# Legend
patches = [
    mpatches.Patch(color='#d6604d', label='p < 0.05'),
    mpatches.Patch(color='#fdae61', label='p < 0.10'),
    mpatches.Patch(color='#aaaaaa', label='n.s.'),
]
ax2.legend(handles=patches, fontsize=9, loc='lower right')

for i, (c, se, p) in enumerate(zip(coefs, ses, pvs)):
    stars = '***' if p<0.01 else ('**' if p<0.05 else ('*' if p<0.10 else ''))
    if stars:
        xoff = c + 1.96*se + 0.01 if c > 0 else c - 1.96*se - 0.01
        ha = 'left' if c > 0 else 'right'
        ax2.text(xoff, i, stars, va='center', ha=ha, fontsize=10, color='#333333')

plt.tight_layout(pad=2.0)
plt.savefig('sharia_rd_figure.png', dpi=150, bbox_inches='tight')
print("Saved sharia_rd_figure.png")
