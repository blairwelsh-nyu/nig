"""
Sharia RD coefficient plot — pre-insurgency window R2 (2003) / R3 (2005) / R4 (2008).
Follows dataviz skill: palette.md slots 1-3, mark specs, anatomy.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Color tokens (palette.md) ─────────────────────────────────────────────────
S = '#fcfcfb'          # chart surface (light)
INK   = '#0b0b0b'      # primary text
INK2  = '#52514e'      # secondary text
MUTED = '#898781'      # muted / axis labels
GRID  = '#e1e0d9'      # hairline gridlines
AXIS  = '#c3c2b7'      # baseline / spine

# Categorical slots 1-3 (validated all-pairs in light mode)
COL = {
    'relig':  '#1baf7a',   # slot 3 aqua   — Religious leaders (primary)
    'courts': '#eb6834',   # slot 2 orange — Courts trust
    'police': '#2a78d6',   # slot 1 blue   — Police trust
}

# ── Data ─────────────────────────────────────────────────────────────────────
# OLS EA-clustered, BW=150 km; CCT = rdrobust MSE-optimal (masspoints='off')
PANELS = [
    {
        'name': 'Religious Leaders',
        'key':  'relig',
        'theory': 'Immediate: Sharia embeds\nreligious authority in law',
        'coef': [0.430, 0.491, 0.464],
        'se':   [0.207, 0.176, 0.244],
        'pval': [0.038, 0.005, 0.058],
        'cct':  (0.562, 0.236, 0.017),   # R4 CCT (coef, se, p)
    },
    {
        'name': 'Courts Trust',
        'key':  'courts',
        'theory': 'Delayed: Sharia courts\noperational by R4',
        'coef': [0.216, 0.269, 0.523],
        'se':   [0.267, 0.174, 0.201],
        'pval': [0.418, 0.122, 0.009],
        'cct':  (0.601, 0.205, 0.003),
    },
    {
        'name': 'Police Trust',
        'key':  'police',
        'theory': 'Delayed: State-capacity\nspillover by R4',
        'coef': [0.136, 0.178, 0.496],
        'se':   [0.237, 0.163, 0.229],
        'pval': [0.565, 0.275, 0.030],
        'cct':  (0.464, 0.210, 0.027),
    },
]

ROUNDS = [0, 1, 2]           # y-positions: R2 at bottom, R4 at top
ROUND_LBL = ['R2  2003', 'R3  2005', 'R4  2008']

def star(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.10:  return '†'
    return ''

# ── Figure layout ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
fig.patch.set_facecolor(S)
plt.subplots_adjust(wspace=0.12)

XLIM = (-0.55, 1.10)

for ax, p in zip(axes, PANELS):
    color = COL[p['key']]
    ax.set_facecolor(S)
    ax.set_xlim(*XLIM)
    ax.set_ylim(-0.75, 2.85)

    # ── Grid (hairline, solid, muted) ─────────────────────────────────────────
    for y in ROUNDS:
        ax.axhline(y, color=GRID, lw=0.75, zorder=0)

    # ── Zero reference ────────────────────────────────────────────────────────
    ax.axvline(0, color=MUTED, lw=0.9, ls='--', alpha=0.65, zorder=1)

    # ── Connecting trend line (subtle, 0-3 order) ─────────────────────────────
    ax.plot(p['coef'], ROUNDS,
            color=color, lw=1.2, alpha=0.25, zorder=2, solid_capstyle='round')

    # ── CI bars + markers ─────────────────────────────────────────────────────
    for i, (y, coef, se, pval) in enumerate(zip(ROUNDS, p['coef'], p['se'], p['pval'])):
        ci = 1.96 * se
        sig = pval < 0.05
        marker = 'D' if sig else 'o'
        ms_fill = 10 if sig else 9

        # CI bar — 2px round cap
        ax.plot([coef - ci, coef + ci], [y, y],
                color=color, lw=2, solid_capstyle='round', zorder=3, alpha=0.88)

        # Surface ring (2px) then filled marker
        ax.plot(coef, y, marker=marker, ms=ms_fill + 4,
                color=S, markeredgecolor='none', zorder=4)
        ax.plot(coef, y, marker=marker, ms=ms_fill,
                color=color, markeredgecolor='none', zorder=5,
                alpha=1.0 if sig else 0.62)

        # Direct label — β + stars, above the CI endpoint
        label_x = coef
        label_y = y + 0.25
        ax.text(label_x, label_y, f'{coef:+.2f}{star(pval)}',
                ha='center', va='bottom', fontsize=8.5,
                color=INK if sig else INK2,
                fontweight='bold' if sig else 'normal')

    # ── CCT robustness note at R4 ─────────────────────────────────────────────
    cc, cse, cp = p['cct']
    ax.text(cc, 2 - 0.34, f'CCT: {cc:+.2f}{star(cp)}',
            ha='center', va='top', fontsize=7.2, color=MUTED, style='italic')

    # ── Theory annotation ─────────────────────────────────────────────────────
    ax.text(XLIM[1] - 0.02, -0.65, p['theory'],
            ha='right', va='bottom', fontsize=7.2, color=MUTED,
            style='italic', linespacing=1.35)

    # ── Panel title ───────────────────────────────────────────────────────────
    # Colored swatch dot + label in primary ink
    ax.set_title(f'  {p["name"]}', fontsize=11, fontweight='bold',
                 color=INK, loc='left', pad=9)
    # Draw swatch
    ax.plot(XLIM[0] + 0.03, 1 + (2.85 - (-0.75)) * 0.955 - (2.85 - (-0.75)),
            marker='o', ms=8, color=color, transform=ax.transData,
            clip_on=False, zorder=10)

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlabel('β (Sharia side vs. non-Sharia, 0–3 scale)', fontsize=8.5,
                  color=INK2, labelpad=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(AXIS)
    ax.spines['bottom'].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3)
    ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(['-0.4', '-0.2', '0', '+0.2', '+0.4',
                        '+0.6', '+0.8', '+1.0'], fontsize=7.5, color=MUTED)

# ── Y-axis (left panel only — sharey) ────────────────────────────────────────
axes[0].set_yticks(ROUNDS)
axes[0].set_yticklabels(ROUND_LBL, fontsize=9.5, color=INK2)
axes[0].spines['left'].set_visible(True)

# ── Legend: ◆ p<0.05  ● p≥0.05 ──────────────────────────────────────────────
leg_patches = [
    mpatches.Patch(color='none', label=''),    # spacer
    plt.Line2D([0],[0], marker='D', color='w', markerfacecolor=INK2,
               ms=8, label='◆  p < 0.05', markeredgecolor='none'),
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=MUTED,
               ms=8, label='●  p ≥ 0.05', markeredgecolor='none'),
]
fig.legend(handles=leg_patches, loc='upper right', bbox_to_anchor=(0.99, 1.01),
           ncol=3, frameon=False, fontsize=8.5, handletextpad=0.4)

# ── Suptitle ─────────────────────────────────────────────────────────────────
fig.suptitle('Sharia Implementation and Institutional Trust — Pre-Insurgency Window (2003–2008)',
             fontsize=12.5, fontweight='bold', color=INK, y=1.04, x=0.5, ha='center')

# ── Footnote ─────────────────────────────────────────────────────────────────
fig.text(0.5, -0.07,
         'Local-linear RD · EA-clustered SE · BW = 150 km · N ≈ 550–660 per round.  '
         'CCT row = rdrobust MSE-optimal bandwidth (masspoints off).\n'
         'R5 (2012) and R6 (2015) omitted: Boko Haram onset (2009) constitutes an independent '
         'negative shock to all trust measures in northern Nigeria.',
         ha='center', va='top', fontsize=7.5, color=MUTED, style='italic',
         linespacing=1.5)

plt.savefig('/home/user/nig/sharia_rd_r2r4_fig.png',
            dpi=220, bbox_inches='tight', facecolor=S)
print('Saved sharia_rd_r2r4_fig.png')
