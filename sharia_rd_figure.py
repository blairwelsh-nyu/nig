"""
Sharia RD figure — two panels:
  Left:  Map of Nigeria showing Sharia boundary, state shading, respondent
         locations coloured by treatment status, and bandwidth rings.
  Right: Coefficient plot (police + courts, two bandwidths) in the style of
         the reference figure (horizontal CI bar + marker + annotated p-value).
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from shapely.geometry import LineString, Point
from sklearn.neighbors import BallTree
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')

os.chdir('/home/user/nig')

# ── Reproduce setup ──────────────────────────────────────────────────────────
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
acled['event_date'] = pd.to_datetime(acled['event_date'], errors='coerce')
state_cents = (acled.groupby('admin1')[['latitude','longitude']]
               .mean().reset_index())
state_cents.columns = ['state','lat','lon']

def assign_state(lat_arr, lon_arr):
    cent_rad = np.radians(state_cents[['lat','lon']].values)
    tree = BallTree(cent_rad, metric='haversine')
    _, idx = tree.query(np.radians(np.column_stack([lat_arr, lon_arr])), k=1)
    return state_cents['state'].values[idx.flatten()]

def signed_dist_km(lat_arr, lon_arr, state_arr, earth_r=6371.0):
    n = len(lat_arr)
    dist = np.full(n, np.nan)
    for i in range(n):
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
    d['y'] = d[outcome_col]; d['x'] = d['dist_km']
    try:
        m = smf.ols('y ~ sharia + x + x:sharia', data=d).fit(cov_type='HC3')
    except:
        return None
    coef = m.params.get('sharia', np.nan)
    se   = m.bse.get('sharia', np.nan)
    pval = m.pvalues.get('sharia', np.nan)
    if abs(coef) > 10 or se > 10:
        return None
    return {'coef': coef, 'se': se, 'pval': pval, 'n': len(d)}


# ── Colours ──────────────────────────────────────────────────────────────────
C_SHARIA    = '#4e7fa8'   # slate-blue (Sharia states / significant)
C_NONSHARIA = '#aabccc'   # pale blue  (non-Sharia)
C_LINE      = '#c0392b'   # red boundary line

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 6))
gs  = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.10)
ax_map  = fig.add_subplot(gs[0])
ax_coef = fig.add_subplot(gs[1])

# ═══════════════════════════════════════════════════════════════════════
#  LEFT PANEL — Map
# ═══════════════════════════════════════════════════════════════════════
ax = ax_map
ax.set_facecolor('#f7f9fb')

# Nigeria approximate bounding box
NGA_LON = (2.7, 14.8); NGA_LAT = (4.2, 14.0)
ax.set_xlim(*NGA_LON); ax.set_ylim(*NGA_LAT)
ax.set_aspect('equal')

# Faint grid
for lon in np.arange(4, 15, 2):
    ax.axvline(lon, color='white', linewidth=0.7, zorder=0)
for lat in np.arange(5, 14, 2):
    ax.axhline(lat, color='white', linewidth=0.7, zorder=0)

# Shade Sharia / non-Sharia background using ACLED respondent convex-hull proxy:
# Simply shade a rough rectangle north of the boundary in light blue, south in pale
# grey. This is an approximation — the actual boundary is drawn precisely below.
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

# Sharia region: above the boundary line (approximate fill polygon)
sharia_fill_x = ([c[0] for c in SHARIA_BOUNDARY_COORDS]
                 + [14.8, 14.8, 2.7, 2.7])
sharia_fill_y = ([c[1] for c in SHARIA_BOUNDARY_COORDS]
                 + [SHARIA_BOUNDARY_COORDS[-1][1], 14.0, 14.0, SHARIA_BOUNDARY_COORDS[0][1]])
ax.fill(sharia_fill_x, sharia_fill_y, color='#d6e4f0', alpha=0.55, zorder=1)

# Non-Sharia region fill (rest of Nigeria)
nonsharia_fill_x = ([c[0] for c in SHARIA_BOUNDARY_COORDS]
                    + [14.8, 14.8, 2.7, 2.7])
nonsharia_fill_y = ([c[1] for c in SHARIA_BOUNDARY_COORDS]
                    + [4.2, 4.2, 4.2, SHARIA_BOUNDARY_COORDS[0][1]])
ax.fill(nonsharia_fill_x, nonsharia_fill_y, color='#e8eef3', alpha=0.55, zorder=1)

# 150-km buffer band around boundary
# Approximate by offsetting boundary coords ±1.35° lat (150km ≈ 1.35°)
deg_150 = 150 / 111.0
upper_x = [c[0] for c in SHARIA_BOUNDARY_COORDS]
upper_y = [c[1] + deg_150 for c in SHARIA_BOUNDARY_COORDS]
lower_x = [c[0] for c in SHARIA_BOUNDARY_COORDS]
lower_y = [c[1] - deg_150 for c in SHARIA_BOUNDARY_COORDS]
band_x  = upper_x + lower_x[::-1] + [upper_x[0]]
band_y  = upper_y + lower_y[::-1] + [upper_y[0]]
ax.fill(band_x, band_y, color='#f0d080', alpha=0.22, zorder=2, label='150 km band')

# Sharia boundary line
bx = [c[0] for c in SHARIA_BOUNDARY_COORDS]
by = [c[1] for c in SHARIA_BOUNDARY_COORDS]
ax.plot(bx, by, color=C_LINE, linewidth=2.2, zorder=5, solid_capstyle='round',
        label='Sharia boundary (approx.)')

# State centroid labels (selected key states)
LABEL_STATES = {
    'Sokoto':   (0.0, 0.15), 'Kebbi':   (0.0, 0.0),  'Zamfara':  (0.0, 0.0),
    'Katsina':  (0.0, 0.0),  'Kano':    (0.0, 0.15), 'Jigawa':   (0.0, 0.0),
    'Kaduna':   (0.0, -0.2), 'Niger':   (0.0, 0.0),  'Bauchi':   (0.0, 0.0),
    'Gombe':    (0.0, 0.0),  'Yobe':    (0.0, 0.0),  'Borno':    (0.0, 0.15),
    'Kwara':    (0.0, 0.0),  'Nassarawa':(0.0,0.0),  'Plateau':  (0.0, 0.0),
    'Taraba':   (0.0, 0.0),  'Adamawa': (0.0, 0.0),
}
for _, row in state_cents.iterrows():
    if row['state'] in LABEL_STATES:
        off = LABEL_STATES[row['state']]
        color = C_SHARIA if row['state'] in SHARIA_STATES else '#5a6e7f'
        ax.text(row['lon'] + off[0], row['lat'] + off[1], row['state'],
                fontsize=6.0, ha='center', va='center', color=color,
                alpha=0.85, fontweight='semibold')

# Respondent scatter (only within 200 km for clarity)
near = r2[r2['dist_km'].abs() <= 200].copy()
far  = r2[r2['dist_km'].abs() >  200].copy()
ax.scatter(far['lon'],  far['lat'],  c='#cccccc', s=5, alpha=0.25, lw=0, zorder=3)
ax.scatter(near.loc[near['sharia']==0,'lon'], near.loc[near['sharia']==0,'lat'],
           c=C_NONSHARIA, s=14, alpha=0.70, lw=0, zorder=4, label='Non-Sharia (≤200 km)')
ax.scatter(near.loc[near['sharia']==1,'lon'], near.loc[near['sharia']==1,'lat'],
           c=C_SHARIA,    s=14, alpha=0.70, lw=0, zorder=4, label='Sharia (≤200 km)')

# Annotations
ax.text(5.5, 13.2, 'Sharia states', fontsize=8.5, color=C_SHARIA,
        fontweight='bold', ha='center')
ax.text(5.5, 6.5,  'Non-Sharia states', fontsize=8.5, color='#5a6e7f',
        fontweight='bold', ha='center')

ax.set_xlabel('Longitude (°E)', fontsize=9)
ax.set_ylabel('Latitude (°N)', fontsize=9)
ax.set_title('Nigeria: Sharia Boundary and Survey Respondents\n(Afrobarometer R2, 2003)',
             fontsize=10.5, fontweight='bold', pad=8)
ax.tick_params(labelsize=8)

leg_elements = [
    mpatches.Patch(color='#d6e4f0', alpha=0.7, label='Sharia zone'),
    mpatches.Patch(color='#e8eef3', alpha=0.7, label='Non-Sharia zone'),
    mpatches.Patch(color='#f0d080', alpha=0.5, label='150 km bandwidth'),
    plt.Line2D([0],[0], color=C_LINE,      linewidth=2, label='Boundary (approx.)'),
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=C_SHARIA,    markersize=6, label='Sharia respondents (≤200 km)'),
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=C_NONSHARIA, markersize=6, label='Non-Sharia respondents (≤200 km)'),
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#cccccc',   markersize=6, label='Respondents (>200 km)'),
]
ax.legend(handles=leg_elements, fontsize=6.8, loc='lower right',
          framealpha=0.85, edgecolor='#cccccc')

# ═══════════════════════════════════════════════════════════════════════
#  RIGHT PANEL — Coefficient plot (reference style)
# ═══════════════════════════════════════════════════════════════════════
ax = ax_coef
ax.set_facecolor('white')
ax.spines[['top','right','left']].set_visible(False)
ax.spines['bottom'].set_linewidth(0.8)

# Two outcomes × two bandwidth specifications
# Circle marker = BW 100 km;  Triangle marker = BW 150 km
OUTCOMES = [
    ('Trust in Police', 'q43i'),
    ('Trust in Courts', 'q43j'),
]
BANDWIDTHS = [(100, 'o'), (150, '^')]

# Compute results
rows = []
for label, col in OUTCOMES:
    for bw, marker in BANDWIDTHS:
        r = run_rd(r2, col, bw)
        if r:
            rows.append({'label': label, 'col': col, 'bw': bw,
                         'marker': marker, **r})
results_df = pd.DataFrame(rows)

# Layout: each outcome group occupies 1 unit; two sub-rows per group
# Groups are plotted top-to-bottom
group_gap  = 1.0   # space between outcome groups
sub_gap    = 0.32  # space between the two bw rows within a group
n_groups   = len(OUTCOMES)

y_positions = {}
for g, (label, col) in enumerate(OUTCOMES):
    y_top = (n_groups - 1 - g) * (group_gap + sub_gap)
    y_positions[(label, 0)] = y_top               # BW 100  (circle, top sub-row)
    y_positions[(label, 1)] = y_top - sub_gap     # BW 150 (triangle, bottom sub-row)

# Colors: use same blue for both, differentiated only by marker
CI_COLOR_INNER = C_SHARIA
CI_COLOR_OUTER = '#a0b8cc'

for _, row in results_df.iterrows():
    bw_idx = 0 if row['bw'] == 100 else 1
    yy     = y_positions[(row['label'], bw_idx)]
    coef   = row['coef']
    se     = row['se']
    pval   = row['pval']
    n      = row['n']
    marker = row['marker']

    # CI bars: outer = 95% CI, inner = 90% CI
    ci95_lo, ci95_hi = coef - 1.96*se, coef + 1.96*se
    ci90_lo, ci90_hi = coef - 1.645*se, coef + 1.645*se

    # Outer (95%) — lighter, thin
    ax.plot([ci95_lo, ci95_hi], [yy, yy],
            color=CI_COLOR_OUTER, linewidth=5, solid_capstyle='butt',
            alpha=0.55, zorder=2)
    # Inner (90%) — darker, thick
    ax.plot([ci90_lo, ci90_hi], [yy, yy],
            color=CI_COLOR_INNER, linewidth=5, solid_capstyle='butt',
            alpha=0.85, zorder=3)
    # Point estimate marker
    ax.plot(coef, yy, marker=marker, markersize=9,
            color='white', markeredgecolor=CI_COLOR_INNER,
            markeredgewidth=1.8, zorder=4)

    # Coefficient + p-value label box below the CI line
    sign_str = '+' if coef >= 0 else ''
    label_str = f'{sign_str}{coef:.2f}  (P = {pval:.3f})'
    ax.text(coef, yy - 0.14, label_str,
            ha='center', va='top', fontsize=8.5,
            bbox=dict(boxstyle='square,pad=0.15', facecolor='white',
                      edgecolor=CI_COLOR_INNER if pval < 0.05 else '#bbbbbb',
                      linewidth=0.8),
            zorder=5)

# Outcome group labels (bold, boxed, left side)
# Determine x position for labels (left of plot area)
x_label = -0.90
for g, (label, col) in enumerate(OUTCOMES):
    y_center = (y_positions[(label, 0)] + y_positions[(label, 1)]) / 2
    # Bold box if both specs are significant at p<0.05
    sub = results_df[results_df['label'] == label]
    significant = (sub['pval'] < 0.05).all()
    edge_color = C_SHARIA if significant else '#888888'
    lw         = 1.5 if significant else 0.8
    fw         = 'bold' if significant else 'normal'
    ax.text(x_label, y_center, label,
            ha='left', va='center', fontsize=10,
            fontweight=fw, color=CI_COLOR_INNER if significant else '#555555',
            bbox=dict(boxstyle='square,pad=0.3', facecolor='white',
                      edgecolor=edge_color, linewidth=lw),
            zorder=5)

# Bandwidth legend (markers)
bw_legend = [
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor=CI_COLOR_INNER, markeredgewidth=1.8,
               markersize=8, label='BW = 100 km'),
    plt.Line2D([0],[0], marker='^', color='w', markerfacecolor='white',
               markeredgecolor=CI_COLOR_INNER, markeredgewidth=1.8,
               markersize=8, label='BW = 150 km'),
]
ax.legend(handles=bw_legend, fontsize=8.5, loc='upper right',
          framealpha=0.9, edgecolor='#cccccc')

# Zero line
ax.axvline(0, color='#333333', linewidth=0.9, linestyle='--', alpha=0.7, zorder=1)

# Axis limits and labels
all_lo = results_df.apply(lambda r: r['coef'] - 1.96*r['se'], axis=1).min()
all_hi = results_df.apply(lambda r: r['coef'] + 1.96*r['se'], axis=1).max()
pad    = 0.12
ax.set_xlim(all_lo - pad, all_hi + pad)
y_all = list(y_positions.values())
ax.set_ylim(min(y_all) - 0.55, max(y_all) + 0.45)
ax.set_yticks([])
ax.set_xlabel('Effect of Sharia state on trust (0–3 scale)\n[local-linear RD, HC3 SE]',
              fontsize=9.5)
ax.set_title('Institutional Trust Discontinuity\nat Sharia Boundary (R2, 2003)',
             fontsize=10.5, fontweight='bold', pad=8)
ax.tick_params(axis='x', labelsize=8.5)

plt.tight_layout(pad=1.5)
plt.savefig('sharia_rd_figure.png', dpi=175, bbox_inches='tight', facecolor='white')
print("Saved sharia_rd_figure.png")
