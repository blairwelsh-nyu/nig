"""
Figure 2 (standalone): Coefficient plot for the Sharia RD.
Police and courts only, two bandwidth specifications (100 km / 150 km),
styled to match the reference figure (horizontal CI bars, annotated p-values,
boxed group labels).
"""
import os, warnings
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from sklearn.neighbors import BallTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')

os.chdir('/home/user/nig')

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

r2 = pd.read_excel('NIG_r2.csv.xlsx')
r2.columns = r2.columns.str.lower().str.strip()
lat_col = next(c for c in r2.columns if 'lat' in c)
lon_col = next(c for c in r2.columns if 'lon' in c)
r2 = r2.rename(columns={lat_col:'lat', lon_col:'lon'})
r2['lat'] = pd.to_numeric(r2['lat'], errors='coerce')
r2['lon'] = pd.to_numeric(r2['lon'], errors='coerce')
r2 = r2.dropna(subset=['lat','lon'])
r2['state']   = assign_state(r2['lat'].values, r2['lon'].values)
r2['dist_km'] = signed_dist_km(r2['lat'].values, r2['lon'].values, r2['state'].values)
r2['sharia']  = r2['state'].isin(SHARIA_STATES).astype(float)
for col in ['q43i','q43j']:
    r2[col] = pd.to_numeric(r2[col], errors='coerce')
    r2.loc[r2[col] >= 7, col] = np.nan

def run_rd(df, outcome_col, bandwidth):
    d = df.dropna(subset=[outcome_col, 'dist_km', 'sharia']).copy()
    d = d[d['dist_km'].abs() <= bandwidth]
    if len(d) < 50 or d[outcome_col].nunique() < 2:
        return None
    try:
        m = smf.ols('y ~ sharia + x + x:sharia',
                    data=d.assign(y=d[outcome_col], x=d['dist_km'])
                    ).fit(cov_type='HC3')
    except:
        return None
    coef = m.params.get('sharia', np.nan)
    se   = m.bse.get('sharia',   np.nan)
    pval = m.pvalues.get('sharia', np.nan)
    return None if (abs(coef) > 10 or se > 10) else {
        'coef': coef, 'se': se, 'pval': pval, 'n': len(d)}

# ── Colours ──────────────────────────────────────────────────────────────────
C_MAIN = '#3d7ab5'
C_PALE = '#9bbbd6'

# ── Build results ─────────────────────────────────────────────────────────────
OUTCOMES   = [('Trust in Police', 'q43i'), ('Trust in Courts', 'q43j')]
BANDWIDTHS = [(100, 'o', 'BW = 100 km'), (150, '^', 'BW = 150 km')]

rows = []
for label, col in OUTCOMES:
    for bw, marker, bw_label in BANDWIDTHS:
        r = run_rd(r2, col, bw)
        if r:
            rows.append({'label': label, 'col': col, 'bw': bw,
                         'bw_label': bw_label, 'marker': marker, **r})
df_res = pd.DataFrame(rows)

# ── Layout ────────────────────────────────────────────────────────────────────
# Each outcome group: top sub-row (BW100) + bottom sub-row (BW150)
# Groups separated by a larger gap; annotation box sits below each sub-row
SUB_H   = 0.35    # height per sub-row
ANN_H   = 0.22    # height for annotation text below sub-row
GRP_H   = 2 * (SUB_H + ANN_H) + 0.25   # total height per group
GRP_GAP = 0.35    # extra gap between groups

y_sub = {}   # (label, bw_idx) → y centre of CI bar
y_ann = {}   # (label, bw_idx) → y centre of annotation box
for g, (label, _) in enumerate(OUTCOMES):
    g_top = (len(OUTCOMES) - 1 - g) * (GRP_H + GRP_GAP)
    for bi, (bw, _, _) in enumerate(BANDWIDTHS):
        bar_y = g_top - bi * (SUB_H + ANN_H)
        y_sub[(label, bi)] = bar_y
        y_ann[(label, bi)] = bar_y - SUB_H * 0.7

# ── Figure ────────────────────────────────────────────────────────────────────
fig_h = len(OUTCOMES) * (GRP_H + GRP_GAP) + 0.4
fig, ax = plt.subplots(figsize=(6.5, fig_h))
ax.set_facecolor('white')
fig.patch.set_facecolor('white')
ax.spines[['top','right','left']].set_visible(False)
ax.spines['bottom'].set_color('#555555')
ax.spines['bottom'].set_linewidth(0.8)

for _, row in df_res.iterrows():
    bi    = 0 if row['bw'] == 100 else 1
    yy    = y_sub[(row['label'], bi)]
    coef  = row['coef']
    se    = row['se']
    pval  = row['pval']

    ci95 = (coef - 1.96*se, coef + 1.96*se)
    ci90 = (coef - 1.645*se, coef + 1.645*se)

    # Outer 95% CI (lighter, butt-capped so it looks like a flat bar)
    ax.plot(ci95, [yy, yy], color=C_PALE, linewidth=7,
            solid_capstyle='butt', alpha=0.60, zorder=2)
    # Inner 90% CI (darker, slightly narrower so outer wings show)
    ax.plot(ci90, [yy, yy], color=C_MAIN, linewidth=7,
            solid_capstyle='butt', alpha=0.88, zorder=3)
    # Point estimate
    ax.plot(coef, yy, marker=row['marker'], markersize=10,
            color='white', markeredgecolor=C_MAIN,
            markeredgewidth=2.0, zorder=4)

    # Annotation box (coefficient + p-value) below the bar
    sign   = '+' if coef >= 0 else ''
    ann    = f'{sign}{coef:.2f}  (P = {pval:.3f})'
    border = C_MAIN if pval < 0.05 else '#aaaaaa'
    y_a    = y_ann[(row['label'], bi)]
    ax.text(coef, y_a, ann,
            ha='center', va='center', fontsize=9.0,
            bbox=dict(boxstyle='square,pad=0.18', facecolor='white',
                      edgecolor=border, linewidth=0.9),
            zorder=5)

# Group labels to the left, using blended transform
label_tf = blended_transform_factory(ax.transAxes, ax.transData)
for g, (label, _) in enumerate(OUTCOMES):
    all_y = [y_sub[(label, bi)] for bi in range(len(BANDWIDTHS))]
    y_ctr = (max(all_y) + y_ann[(label, len(BANDWIDTHS)-1)]) / 2
    primary = df_res[(df_res['label'] == label) & (df_res['bw'] == 150)]
    sig = len(primary) > 0 and primary.iloc[0]['pval'] < 0.05
    fc  = C_MAIN    if sig else '#555555'
    ec  = C_MAIN    if sig else '#999999'
    fw  = 'bold'    if sig else 'normal'
    lw  = 1.4       if sig else 0.8
    ax.text(-0.04, y_ctr, label,
            transform=label_tf, ha='right', va='center',
            fontsize=10.5, fontweight=fw, color=fc, clip_on=False,
            bbox=dict(boxstyle='square,pad=0.30', facecolor='white',
                      edgecolor=ec, linewidth=lw),
            zorder=6)

# Zero reference line
ax.axvline(0, color='#333333', linewidth=0.9, linestyle='--', alpha=0.65, zorder=1)

# Bandwidth legend (top-right, inside axes)
bw_handles = [
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor=C_MAIN, markeredgewidth=2, markersize=8,
               label='BW = 100 km'),
    plt.Line2D([0],[0], marker='^', color='w', markerfacecolor='white',
               markeredgecolor=C_MAIN, markeredgewidth=2, markersize=8,
               label='BW = 150 km'),
]
leg = ax.legend(handles=bw_handles, fontsize=8.5, loc='upper right',
                framealpha=0.95, edgecolor='#cccccc', fancybox=False)
leg.get_frame().set_linewidth(0.6)

# Axis limits
lo = df_res.apply(lambda r: r['coef'] - 1.96*r['se'], axis=1).min()
hi = df_res.apply(lambda r: r['coef'] + 1.96*r['se'], axis=1).max()
ax.set_xlim(lo - 0.15, max(hi + 0.15, 0.22))
all_y_vals = list(y_sub.values()) + list(y_ann.values())
ax.set_ylim(min(all_y_vals) - 0.30, max(all_y_vals) + 0.30)
ax.set_yticks([])
ax.set_xlabel('Effect of Sharia state on institutional trust (0–3 scale)\n'
              'Local-linear RD estimate, HC3 standard errors',
              fontsize=9.5)
ax.set_title('Sharia Law and Institutional Trust\nat the State Boundary (R2, 2003)',
             fontsize=12, fontweight='bold', pad=10)
ax.tick_params(axis='x', labelsize=9)

plt.tight_layout(pad=1.5)
plt.savefig('sharia_rd_coef.png', dpi=200, bbox_inches='tight', facecolor='white')
print("Saved sharia_rd_coef.png")
