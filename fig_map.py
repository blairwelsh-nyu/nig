"""
Figure 1 (standalone): Nigeria Sharia boundary map.
Uses Natural Earth 10m admin1 polygons for accurate state outlines.
Respondents coloured by which side of the boundary line they fall on
(not by centroid-state assignment, which has classification error near
the boundary).
"""
import os, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
from sklearn.neighbors import BallTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
warnings.filterwarnings('ignore')

os.chdir('/home/user/nig')

# ── Constants ────────────────────────────────────────────────────────────────
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

C_SHARIA    = '#3d7ab5'
C_NONSHARIA = '#b8cfe0'
C_FILL_S    = '#d6e8f5'
C_FILL_NS   = '#eef2f5'
C_BOUNDARY  = '#c0392b'
C_DOT_S     = '#1a5e96'
C_DOT_NS    = '#7fa8c9'
C_DOT_FAR   = '#d0d8df'

# ── Load state polygons ───────────────────────────────────────────────────────
nga = gpd.read_file('/tmp/nga_admin1.gpkg')
nga['sharia'] = nga['name'].isin(SHARIA_STATES)
# Normalise state names that differ between NE and ACLED
NE_NAME_FIX = {'Federal Capital Territory': 'Abuja'}
nga['name_acled'] = nga['name'].replace(NE_NAME_FIX)

# ── Load respondent data ──────────────────────────────────────────────────────
acled = pd.read_csv('nigeria_acled.csv')
state_cents = (acled.groupby('admin1')[['latitude','longitude']]
               .mean().reset_index())
state_cents.columns = ['state','lat','lon']

def assign_state(lat_arr, lon_arr):
    cent_rad = np.radians(state_cents[['lat','lon']].values)
    tree = BallTree(cent_rad, metric='haversine')
    _, idx = tree.query(np.radians(np.column_stack([lat_arr, lon_arr])), k=1)
    return state_cents['state'].values[idx.flatten()]

r2 = pd.read_excel('NIG_r2.csv.xlsx')
r2.columns = r2.columns.str.lower().str.strip()
lat_col = next(c for c in r2.columns if 'lat' in c)
lon_col = next(c for c in r2.columns if 'lon' in c)
r2 = r2.rename(columns={lat_col:'lat', lon_col:'lon'})
r2['lat'] = pd.to_numeric(r2['lat'], errors='coerce')
r2['lon'] = pd.to_numeric(r2['lon'], errors='coerce')
r2 = r2.dropna(subset=['lat','lon'])

# Colour by boundary side, not centroid-state, to avoid visual inconsistency
def point_above_boundary(lat, lon, boundary=boundary_line, earth_r=6371.0):
    """True if point is on the Sharia (north/above) side of the boundary line."""
    p = Point(lon, lat)
    lat_rad  = np.radians(lat)
    deg_lat  = np.pi * earth_r / 180.0
    deg_lon  = deg_lat * np.cos(lat_rad)
    km = boundary.distance(p) * np.sqrt(deg_lat**2 + deg_lon**2) / np.sqrt(2)
    # Find nearest boundary segment to determine which side
    # Simple heuristic: project lon onto boundary, compare lat to boundary lat
    bx = np.array([c[0] for c in SHARIA_BOUNDARY_COORDS])
    by = np.array([c[1] for c in SHARIA_BOUNDARY_COORDS])
    # Interpolate boundary lat at respondent's longitude
    if lon <= bx[0]:
        b_lat = by[0]
    elif lon >= bx[-1]:
        b_lat = by[-1]
    else:
        idx = np.searchsorted(bx, lon) - 1
        idx = np.clip(idx, 0, len(bx)-2)
        frac = (lon - bx[idx]) / max(bx[idx+1] - bx[idx], 1e-9)
        b_lat = by[idx] + frac * (by[idx+1] - by[idx])
    return lat >= b_lat

r2['on_sharia_side'] = [
    point_above_boundary(row['lat'], row['lon'])
    for _, row in r2.iterrows()
]

# Signed distance for bandwidth filter
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

r2['state']   = assign_state(r2['lat'].values, r2['lon'].values)
r2['dist_km'] = signed_dist_km(r2['lat'].values, r2['lon'].values, r2['state'].values)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 8.5))
ax.set_facecolor('#f0f4f8')
fig.patch.set_facecolor('white')

# State fills
nga_sharia    = nga[nga['sharia']]
nga_nonsharia = nga[~nga['sharia']]
nga_nonsharia.plot(ax=ax, color=C_FILL_NS, edgecolor='white', linewidth=0.5, zorder=2)
nga_sharia.plot(   ax=ax, color=C_FILL_S,  edgecolor='white', linewidth=0.5, zorder=2)

# State outlines (thin grey on top)
nga.plot(ax=ax, facecolor='none', edgecolor='#9ab0c0', linewidth=0.45, zorder=3)

# 150-km bandwidth band
deg_150 = 150 / 111.0
upper_x = [c[0] for c in SHARIA_BOUNDARY_COORDS]
upper_y = [c[1] + deg_150 for c in SHARIA_BOUNDARY_COORDS]
lower_x = [c[0] for c in SHARIA_BOUNDARY_COORDS]
lower_y = [c[1] - deg_150 for c in SHARIA_BOUNDARY_COORDS]
band_x  = upper_x + lower_x[::-1] + [upper_x[0]]
band_y  = upper_y + lower_y[::-1] + [upper_y[0]]
ax.fill(band_x, band_y, color='#f5c842', alpha=0.20, zorder=4, linewidth=0)
# Band edges (dashed)
ax.plot(upper_x, upper_y, color='#c8a000', linewidth=0.7,
        linestyle='--', alpha=0.55, zorder=5)
ax.plot(lower_x, lower_y, color='#c8a000', linewidth=0.7,
        linestyle='--', alpha=0.55, zorder=5)

# Sharia boundary
bx = [c[0] for c in SHARIA_BOUNDARY_COORDS]
by = [c[1] for c in SHARIA_BOUNDARY_COORDS]
ax.plot(bx, by, color=C_BOUNDARY, linewidth=2.2, solid_capstyle='round',
        solid_joinstyle='round', zorder=6)

# Respondents: far from boundary (grey), near non-Sharia (light blue), near Sharia (dark blue)
near = r2[r2['dist_km'].abs() <= 200]
far  = r2[r2['dist_km'].abs() >  200]

ax.scatter(far['lon'], far['lat'], s=4, color=C_DOT_FAR, alpha=0.35,
           linewidths=0, zorder=7)
near_ns = near[~near['on_sharia_side']]
near_s  = near[near['on_sharia_side']]
ax.scatter(near_ns['lon'], near_ns['lat'], s=13, color=C_DOT_NS, alpha=0.80,
           linewidths=0, zorder=8)
ax.scatter(near_s['lon'],  near_s['lat'],  s=13, color=C_DOT_S,  alpha=0.80,
           linewidths=0, zorder=8)

# State labels — border states only, placed at polygon centroids
BORDER_LABELS = {
    # Sharia side
    'Kebbi': (-0.3, 0.0), 'Niger': (0.0, 0.3), 'Kaduna': (0.0, 0.0),
    'Bauchi': (0.0, 0.0), 'Gombe': (0.3, 0.0), 'Borno': (0.0, 0.3),
    # Non-Sharia side
    'Kwara': (0.0, -0.1), 'Nassarawa': (0.5, 0.0), 'Plateau': (0.0, 0.0),
    'Taraba': (0.0, 0.0), 'Adamawa': (0.0, -0.3),
}
for _, row in nga.iterrows():
    if row['name'] in BORDER_LABELS:
        dx, dy = BORDER_LABELS[row['name']]
        cx = row.geometry.centroid.x + dx
        cy = row.geometry.centroid.y + dy
        is_s = row['sharia']
        ax.text(cx, cy, row['name'],
                fontsize=6.8, ha='center', va='center',
                color='#1a3d5c' if is_s else '#2c4a5e',
                fontweight='bold', alpha=0.9, zorder=10)

# Zone labels
ax.text(5.8, 13.0, 'Sharia states',     fontsize=10, fontweight='bold',
        color=C_SHARIA,    ha='center', zorder=11)
ax.text(5.8,  6.2, 'Non-Sharia states', fontsize=10, fontweight='bold',
        color='#4a6e82', ha='center', zorder=11)

# Legend
legend_elements = [
    mpatches.Patch(facecolor=C_FILL_S,  edgecolor='#9ab0c0', label='Sharia state'),
    mpatches.Patch(facecolor=C_FILL_NS, edgecolor='#9ab0c0', label='Non-Sharia state'),
    mpatches.Patch(facecolor='#f5c842', alpha=0.35,
                   edgecolor='#c8a000', label='150 km bandwidth'),
    Line2D([0],[0], color=C_BOUNDARY, linewidth=2.0, label='Sharia boundary (approx.)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C_DOT_S,  markersize=6,
           label='Sharia side (≤200 km)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C_DOT_NS, markersize=6,
           label='Non-Sharia side (≤200 km)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C_DOT_FAR, markersize=6,
           label='Respondents (>200 km)'),
]
leg = ax.legend(handles=legend_elements, fontsize=7.5, loc='lower left',
                framealpha=0.93, edgecolor='#c0ccd6', fancybox=False)
leg.get_frame().set_linewidth(0.6)

# Extent to Nigeria only
minx, miny, maxx, maxy = nga.total_bounds
pad = 0.4
ax.set_xlim(minx - pad, maxx + pad)
ax.set_ylim(miny - pad, maxy + pad)

ax.set_xlabel('Longitude (°E)', fontsize=9)
ax.set_ylabel('Latitude (°N)', fontsize=9)
ax.set_title('Nigeria: Sharia Law Boundary and\nAfrobarometer Respondents (R2, 2003)',
             fontsize=12, fontweight='bold', pad=10)
ax.tick_params(labelsize=8)
# Light grid
ax.grid(True, color='white', linewidth=0.5, alpha=0.7, zorder=1)

plt.tight_layout(pad=1.2)
plt.savefig('sharia_rd_map.png', dpi=200, bbox_inches='tight', facecolor='white')
print("Saved sharia_rd_map.png")
