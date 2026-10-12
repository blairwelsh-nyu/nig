"""
Nigeria Watch × Afrobarometer analysis.
Compares NW kidnapping coverage with ACLED, creates spatial exposure measure.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.neighbors import BallTree

NW_LIST    = 'nigeria_watch_list.csv'
NW_DETAILS = 'nigeria_watch_details.csv'
ACLED_FILE = 'nigeria_acled.csv'

# ─── 1. Load and clean Nigeria Watch ────────────────────────────────────────

def load_nw_list():
    df = pd.read_csv(NW_LIST, dtype=str)
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    df['end_date']   = pd.to_datetime(df['end_date'],   errors='coerce')
    df['deaths']     = pd.to_numeric(df['deaths'], errors='coerce').fillna(0)
    df['year'] = df['start_date'].dt.year
    return df


def load_nw_details():
    df = pd.read_csv(NW_DETAILS, dtype=str)
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    df['year'] = df['start_date'].dt.year
    df['deaths_detail'] = pd.to_numeric(df['Number of death(s)'], errors='coerce').fillna(0)
    return df


# ─── 2. Coverage comparison with ACLED ─────────────────────────────────────

def coverage_comparison():
    nw = load_nw_list()
    acled = pd.read_csv(ACLED_FILE)
    acled['event_date'] = pd.to_datetime(acled['event_date'], errors='coerce')
    acled['year'] = acled['event_date'].dt.year
    acled_viol = acled[acled['event_type'].isin([
        'Violence against civilians', 'Battles', 'Explosions/Remote violence'
    ])]

    # Annual event counts
    nw_annual   = nw.groupby('year').size().rename('NW_events')
    acled_annual = acled_viol.groupby('year').size().rename('ACLED_events')
    idx = pd.Index(range(1999, 2017))
    comp = pd.concat([nw_annual, acled_annual], axis=1).reindex(idx).fillna(0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Nigeria Watch vs ACLED: Coverage Comparison', fontsize=14, fontweight='bold')

    # Panel A: Annual events
    ax = axes[0, 0]
    comp.plot(ax=ax, color=['#2196F3', '#F44336'])
    ax.set_title('A. Annual event counts (violence)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Events')
    ax.legend()

    # Panel B: Kidnapping comparison
    ax = axes[0, 1]
    KIDNAP_RE = r'kidnap|abduct|ransom|hostage'
    nw_kidnap = nw[nw['title'].str.contains(KIDNAP_RE, case=False, na=False)]
    nw_k_annual = nw_kidnap.groupby('year').size().rename('NW_kidnap')

    acled_kidnap = acled[
        acled['sub_event_type'].str.contains('Abduction', na=False) |
        acled['notes'].str.contains(KIDNAP_RE, case=False, na=False)
    ]
    acled_k_annual = acled_kidnap.groupby('year').size().rename('ACLED_kidnap')

    pd.concat([nw_k_annual, acled_k_annual], axis=1).reindex(idx).fillna(0).plot(
        ax=ax, color=['#2196F3', '#F44336']
    )
    ax.set_title('B. Annual kidnapping events')
    ax.set_xlabel('Year')
    ax.set_ylabel('Kidnapping events')
    ax.legend()

    # Panel C: NW death distribution
    ax = axes[1, 0]
    nw_study = nw[(nw['year'] >= 1999) & (nw['year'] <= 2016)]
    deaths = nw_study['deaths'].clip(upper=20)
    ax.hist(deaths[deaths > 0], bins=20, color='#2196F3', edgecolor='white')
    ax.set_title('C. NW deaths per event (1–20 cap)')
    ax.set_xlabel('Deaths')
    ax.set_ylabel('Events')

    # Panel D: NW events by year with Afrobarometer fieldwork shaded
    ax = axes[1, 1]
    nw_annual.loc[1999:2016].plot(ax=ax, color='#2196F3', label='Nigeria Watch')
    # Shade survey windows
    SURVEYS = {
        'R1': ('2000-06-01', '2001-04-30'),
        'R2': ('2003-05-01', '2003-09-30'),
        'R4': ('2008-05-01', '2008-06-30'),
        'R5': ('2012-10-01', '2013-02-28'),
        'R6': ('2015-06-01', '2015-11-30'),
    }
    colors = ['#FF9800', '#4CAF50', '#9C27B0', '#F44336', '#00BCD4']
    for (rnd, (s, e)), c in zip(SURVEYS.items(), colors):
        s_yr = pd.Timestamp(s).year + pd.Timestamp(s).month / 12
        e_yr = pd.Timestamp(e).year + pd.Timestamp(e).month / 12
        ax.axvspan(s_yr, e_yr, alpha=0.25, color=c, label=rnd)
    ax.set_title('D. NW events with survey windows')
    ax.set_xlabel('Year')
    ax.set_ylabel('NW events')
    ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig('nigeria_watch_coverage.png', dpi=150, bbox_inches='tight')
    print("Saved nigeria_watch_coverage.png")
    plt.close()

    # Print summary stats
    print("\n=== Nigeria Watch Coverage Summary ===")
    print(comp.to_string())
    print(f"\nTotal NW events 1999-2016: {len(nw_study):,}")
    print(f"NW kidnap events 1999-2016: {len(nw_kidnap[(nw_kidnap.year >= 1999) & (nw_kidnap.year <= 2016)]):,}")

    # Afrobarometer survey window counts
    for rnd, (s, e) in SURVEYS.items():
        s_dt, e_dt = pd.Timestamp(s), pd.Timestamp(e)
        nw_w = nw[(nw['start_date'] >= s_dt) & (nw['start_date'] <= e_dt)]
        nw_k = nw_kidnap[(nw_kidnap['start_date'] >= s_dt) & (nw_kidnap['start_date'] <= e_dt)]
        acled_w = acled_viol[(acled_viol['event_date'] >= s_dt) & (acled_viol['event_date'] <= e_dt)]
        acled_k = acled_kidnap[(acled_kidnap['event_date'] >= s_dt) & (acled_kidnap['event_date'] <= e_dt)]
        print(f"{rnd} ({s[:7]}–{e[:7]}): "
              f"NW={len(nw_w)} events ({len(nw_k)} kidnap) | "
              f"ACLED={len(acled_w)} events ({len(acled_k)} kidnap)")

    return nw, acled


# ─── 3. LGA-level analysis of kidnapping detail data ────────────────────────

def kidnap_detail_analysis():
    try:
        det = load_nw_details()
    except FileNotFoundError:
        print("Details file not yet available — skipping LGA analysis")
        return None

    print(f"\n=== NW Kidnap Detail Analysis ({len(det)} events) ===")

    # Causes of violence distribution
    causes = det['Causes of violence'].value_counts().head(15)
    print("\nCauses of violence (top 15):")
    print(causes.to_string())

    # Protagonists
    prot = det['Protagonists'].value_counts().head(15)
    print("\nProtagonists (top 15):")
    print(prot.to_string())

    # LGA distribution
    lga = det['LGA'].value_counts().head(20)
    print("\nLGA (top 20):")
    print(lga.to_string())

    # Annual trend
    annual = det.groupby('year').size()
    print("\nAnnual kidnap events in NW:")
    print(annual.to_string())

    return det


if __name__ == '__main__':
    import os
    os.chdir('/home/user/nig')

    if not os.path.exists(NW_LIST):
        print(f"ERROR: {NW_LIST} not found. Run scrape_nigeria_watch.py first.")
        exit(1)

    print("=== Running Nigeria Watch analysis ===\n")
    nw, acled = coverage_comparison()
    det = kidnap_detail_analysis()
    print("\nAnalysis complete.")
