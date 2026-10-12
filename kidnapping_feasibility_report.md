# Feasibility Report: Kidnapping, Criminal Violence, and State Legitimacy in Nigeria
**Afrobarometer (Rounds 1–6) × ACLED**  
Date: 2026-10-12

---

## Executive Summary

The **short-run before/after design** proposed as the primary identification strategy is **not feasible** with these data. ACLED records zero "Abduction/forced disappearance" events during the R2 fieldwork window (Oct 2003), zero during R4 (May 2008), and only 9 events during R5 (Oct 2012–Jan 2013) spread across 36 states — too sparse for any credible before/after contrast. R1 has no individual interview dates; R6 dates are entirely corrupted to 1970-01-01.

However, several **alternative designs are feasible**, and the data support a credible study of political violence (including kidnapping as a core Boko Haram tactic) and state legitimacy across rounds. The strongest available design is a **Boko Haram difference-in-differences** for rounds 4–6, exploiting the clean 2008 pre-baseline against the 2012–15 escalation period, using outcomes including trust in police (q59h) and government performance on crime reduction (q65f) — both confirmed available in R5.

---

## 1. Data Inventory

### 1.1 Afrobarometer Rounds

| Round | n | Individual Dates | Date Range | % Geo-valid | Status |
|-------|---|-----------------|------------|-------------|--------|
| R1 | 3,603 | **NO** | ~Nov 2000 | 100% | No date column |
| R2 | 2,428 | **YES** | Oct 13–28, 2003 | 100% | 16-day window; clean |
| R3 | 2,363 | **YES** | Aug 28–Dec 31, 2005 | 100% | 4-month window; clean |
| R4 | 2,324 | **YES** | May 13–25, 2008 | 100% | 12-day window; clean |
| R5 | 2,400 | **YES** | Oct 30, 2012–Jan 19, 2013 | 100% | 2.5-month window; clean |
| R6 | 2,400 | **CORRUPTED** | All 1970-01-01 (Unix epoch) | 100% | Date field unusable |

All rounds have respondent-level latitude/longitude coordinates. Coordinates appear to be exact GPS point locations (not cluster-level displaced locations based on inter-respondent spread within localities). This is a significant asset for spatial analysis.

### 1.2 ACLED

- **Total rows:** 41,902 (including protests and strategic developments)
- **Violent events only:** 31,680 (after excluding protests and strategic developments)
- **Date range:** 1999–2025
- **Columns available:** event_id_cnty, event_date, year, time_precision, disorder_type, event_type, sub_event_type, actor1, assoc_actor_1, inter1, actor2, assoc_actor_2, inter2, civilian_targeting, admin1/2/3, location, latitude, longitude, geo_precision, source, source_scale, **notes** (event narrative), fatalities, tags, timestamp

**Geographic and temporal precision for abduction events:**
- geo_precision 1 (exact location): 49.5%
- geo_precision 2 (town/village): 49.7%  
- geo_precision 3 (admin unit): 0.9%
- time_precision 1 (exact day): 91.7%

**Nigeria coverage caution:** This is a Nigeria-only extract. Cross-border events (Borno/Cameroon, Lake Chad) and events first recorded by Cameroonian or Nigerien sources may be under-reported or absent.

---

## 2. Kidnapping Event Audit

### 2.1 ACLED Kidnapping Classification

ACLED uses `sub_event_type == "Abduction/forced disappearance"` for kidnapping events. Total: **3,480 events**, confirmed 100% coded as "Civilian targeting."

**Critical temporal distribution:**

| Period | Events | % of total | Notes |
|--------|--------|------------|-------|
| 1999–2008 | 78 | 2.2% | Mostly Niger Delta oil-worker kidnappings |
| 2009–2018 | 240 | 6.9% | Boko Haram insurgency; grows from ~10/yr to ~52/yr |
| 2019–2025 | 3,162 | 90.9% | NW banditry explosion (Zamfara, Katsina, Kaduna communal militias) |

**The NW banditry/criminal kidnapping surge post-dates all survey rounds entirely.** The data that would fuel the most interesting criminal-vs-insurgent comparison (2019+) has no contemporaneous Afrobarometer coverage.

**Events during survey fieldwork windows:**

| Round | Survey dates | Abduction events in window | Abduction events ±30d lookback |
|-------|-------------|--------------------------|-------------------------------|
| R1 | ~Nov 2000 (no dates) | Unknown | ~2 (2000 had 2 events) |
| R2 | Oct 13–28, 2003 | **0** | **0** |
| R3 | Aug 28–Dec 31, 2005 | **1** | ~3 |
| R4 | May 13–25, 2008 | **0** | 0 (3 events in 2008: Feb, Apr, Nov — none in May) |
| R5 | Oct 30, 2012–Jan 19, 2013 | **9** | ~25 (12-month lookback) |
| R6 | ~2015 (dates corrupted) | Unknown | ~28 |

**R5 abduction events (Oct 2012–Jan 2013, n=9):** All attributed to "Unidentified Armed Group (Nigeria)"; concentrated in Benue, Rivers, Delta, Enugu, Ogun, Bayelsa states. Only 1 event (Feb 2013) is Boko Haram in Borno — which is outside the survey window. These are political elite/ransomware kidnappings (a lawmaker, a minister's mother, a vice-chancellor), not mass civilian events.

### 2.2 Actor Classification (Criminal vs Insurgent)

| Category | Actor pattern | N events | Concentration |
|----------|---------------|----------|---------------|
| BH/ISWAP | Contains "Boko Haram" or "ISWAP" | ~230 | NE Nigeria, 2012–2025 |
| Criminal/communal | "Unidentified Armed Group", "Zamfara Communal Militia", "Katsina Communal Militia", etc. | ~3,000 | NW Nigeria, 2019+ |
| Niger Delta armed | "MEND", "Niger Delta", "Ijaw", "Pirates" | ~50 | Rivers, Delta, Bayelsa, 1999–2012 |
| Unknown/other | Other actors | ~200 | Scattered |

**Manual validation of 9 R5-window events confirms:** All 9 are classified as "criminal/political" (ransom-motivated or targeted at elites/officials), not insurgent. ACLED's own notes describe them as "gunmen kidnap minister's relative," "abducted politician," etc.

**False positive/negative risks:**
- False positives in "Abduction" category: Low. ACLED notes reviewed above confirm kidnapping. Classification is reliable.
- False negatives: Significant. 7,604 events mention "kidnap" or "abduct" in the **notes** field but are coded under other sub_event_types (Attack: 2,325; Armed clash: 1,374). Many Boko Haram raids that involved kidnapping were coded as "Attack" or "Armed clash" rather than "Abduction" because the primary event type was the raid itself. A notes-based broader classification would identify substantially more events but requires manual review of false positives.

### 2.3 Notes-based Broader Classification

Events where "notes" field contains "kidnap" or "abduct" but sub_event_type ≠ Abduction:
- Total: 4,124 additional events
- Of these: Attack (2,325), Armed clash (1,374), Peaceful protest (190), Mob violence (108)
- This broader classification is defensible as a robustness check but cannot replace the sub_event_type coding as a primary measure.

---

## 3. Outcome Variable Crosswalk

### 3.1 Confirmed Outcome Variables by Round

| Variable | R1 | R2 | R3 | R4 | R5 | R6 |
|----------|----|----|----|----|----|----|
| **Fear of crime (in home)** | Q88-type | Q11a (0–4) | Q11 | Q11 | Q11 | Q11a (0–3) |
| **Theft victimization** | Q88b | Q11b (0–4) | Q12 | Q12a | Q88a | Q11b |
| **Physical attack victim.** | Q88c | Q11c (0–4) | — | Q12b | Q88b | Q11c |
| **Trust in police** | — | Q43(police) | Q43 | Q43+Q44 | **Q59h** (0–3) | **Q52h** (0–3) |
| **Trust in army** | — | Q43(army) | Q43 | Q44a | **Q59i** (0–3) | **Q52i** (0–3) |
| **Trust in courts** | — | Q43(courts) | Q43 | Q44c | **Q59j** (0–3) | **Q52j** (0–3) |
| **Trust in president** | — | Q43(pres.) | Q43 | Q43 | **Q59a** (0–3) | **Q52a** (0–3) |
| **Govt: reducing crime** | — | — | — | — | **Q65f** (1–4) | **Q66f** (1–4) |
| **Govt: economy overall** | — | Q42 | Q42 | Q66-type | Q65a (1–4) | Q66a (1–4) |
| **Healthcare deprivation** | povhth (0–3) | Q9c (0–4) | Q8c (0–4) | Q8c (0–4) | Q8c (0–4) | Q8c (0–4) |

**Key findings:**
- **Trust in police (Q59h/Q52h) is cleanly available and identically worded in R5 and R6.** R2–R4 have trust items but at different question numbers and scales (R4 measures institutional *legitimacy* — "police have the right to make people obey" — not trust per se).
- **Government performance on reducing crime (Q65f/Q66f) is available in R5 and R6.** This is the most theory-central outcome for a state-legitimacy study and it is present in both post-BH-escalation rounds.
- **No equivalent crime performance question in R2–R4.** R2 Q66–Q71 cover democratic values, gender equality, and violent conflicts between groups — not government performance ratings.
- R6 dates are corrupted. R6 outcomes can be used cross-sectionally (violence exposure from ACLED vs. outcomes) but not for a short-run before/after design.

### 3.2 Scale Harmonization Notes

- R5 and R6 trust items (Q59/Q52): identical wording, 0–3 scale — directly poolable.
- R2 trust items: 0–3 scale, similar wording, but fewer institutions listed (no "reducing crime" performance item).
- R4 trust items: Q43 = "trust in government" (0–4), Q44 = institutional legitimacy (1–5 Likert) — different construct from R5/R6.
- **Recommendation:** Pool R5 and R6 for trust outcomes; treat R2 as a separate analysis. Do not pool R4 trust with R5/R6 without standardizing.

---

## 4. Feasibility Assessment by Design

### Design 1: Short-Run Local Kidnapping Exposure (Primary Design)

**Assessment: NOT FEASIBLE**

**Evidence:**
- R2 (Oct 2003): Zero ACLED abduction events within ±30 days anywhere in Nigeria.
- R4 (May 2008): Zero abduction events during the 12-day fieldwork window (3 events in 2008 total, all outside May).
- R3 (Aug–Dec 2005): 1 event total in the survey window.
- R5 (Oct 2012–Jan 2013): 9 events, all in different states, all elite-targeted rather than mass-civilian. Even under the most generous 50km / 30-day window, fewer than 30 respondents would be "post-event" — insufficient for any regression.
- R1: No individual interview dates.
- R6: Dates corrupted.

**Identification threat even if events existed:** Afrobarometer fieldwork proceeds sequentially by PSU/cluster. If interviewers cluster visits around known-safe periods and skip localities after violent incidents, interview timing is endogenous to violence. The brief correctly flags this; it would require fieldwork metadata (interviewer logs, PSU visit dates) to assess.

**Verdict:** Abandon this design for the available data. It is not fixable with imputation or looser windows; the event count is structurally too low for rounds 2–4.

---

### Design 2: Cumulative 12-Month Exposure Lookback

**Assessment: PARTIALLY FEASIBLE — R5 only; very limited otherwise**

| Round | Abduction events in prior 12 months | Notes |
|-------|-------------------------------------|-------|
| R2 | 0 | No events |
| R3 | ~3 | Effectively nothing |
| R4 | ~3 | Only Feb–Nov 2007 Niger Delta events |
| R5 | ~25 | 23 in 2012 + 2 early 2013; most in south, not NE |
| R6 | ~28 (2014–2015) | Dates corrupted; cannot assign individual windows |

For R5 only, a 12-month cumulative exposure measure has ~25 abduction events to work with. Cross-state variation is very limited (most events clustered in Rivers, Delta, and Benue; none in 20+ states). Geographic variation in exposure would be almost entirely binary: Niger Delta belt = 1, everywhere else = 0.

**More promising variant:** Use the notes-based broader classification (Attack + Abduction events mentioning "kidnap" in notes). This yields ~50 events in the R5 window, improving variation modestly. Boko Haram attacks in R5 that involved kidnapping as a tactic are captured here.

**Recommendation for this design:** If using, restrict to R5, use notes-based definition for robustness, and be explicit that this is cross-sectional with limited geographic variation. Do not claim causal identification.

---

### Design 3: Formative-Age Exposure

**Assessment: NOT FEASIBLE**

Afrobarometer does not collect birthplace or lifetime location history. The brief correctly notes this. Current residence ≠ childhood residence cannot be resolved. Skip.

---

### Design 4: Exposure to Vigilantes / Non-State Security Providers

**Assessment: LIMITED — check required**

ACLED does capture vigilante actors (Civilian JTF in Borno, CJTF, community vigilance groups). A cursory search of R5/R6 survey variables did not reveal a direct question measuring preference for or trust in vigilantes vs. state police. R5 Q11 asks why crimes aren't reported to police (options include "police wouldn't help" and "police involved in the crime") — this is relevant attitudinal data but not directly vigilante preference.

**Recommendation:** Verify whether R6 Q11a ("How often do people have to rely on community self-help rather than police?") or similar items exist. If not, this design is infeasible. Do not impute vigilante preference from generic safety attitudes.

---

### Design 5: Criminal vs Insurgent Kidnapping (Sub-Analysis)

**Assessment: FEASIBLE in principle, but event counts constrain to Boko Haram focus**

Classification is reproducible:
- **BH insurgent kidnappings:** Filter ACLED actor1/actor2 for "Boko Haram" → ~230 events in Nigeria, mostly 2012–2021, concentrated in Borno/Yobe/Adamawa. Importantly, many BH raids involving mass kidnapping are coded as "Attack" with kidnapping mentioned in notes (e.g., the Chibok abduction in April 2014 appears as an "Attack" event, not "Abduction/forced disappearance").
- **Criminal kidnappings:** "Unidentified Armed Group" + communal militia actors → ~1,800+ events, almost entirely 2019+, concentrated in Katsina/Zamfara/Kaduna/Niger/Sokoto.
- **Niger Delta/piracy kidnappings:** MEND, Pirates of Nigeria → ~50 events, 1999–2012, concentrated in Rivers/Bayelsa/Delta.

**Feasible sub-analysis:** Compare BH insurgent kidnapping exposure (for R5/R6 respondents in NE Nigeria) with general criminal violence exposure (Attacks, VAC) for respondents in the south. Requires accepting that the "criminal" comparison group is drawn from different years and different geography — a significant limitation.

---

## 5. What IS Feasible: Design Recommendation

### Recommended Primary Design: Boko Haram DiD (Implemented)

**Specification:** Difference-in-differences exploiting Boko Haram's escalation after 2011.
- Pre-period: R4 (May 2008) — zero BH activity (first events only in 2009)
- Post-period: R5 (Oct 2012–Jan 2013) and R6 (~2015)
- Treatment: Gaussian kernel-weighted BH fatality exposure within 50km (continuous), or binary BH-zone indicator (Borno, Yobe, Adamawa = core BH states)
- Fixed effects: State + Round (with state FE absorbing time-invariant geography)
- Outcomes: healthcare deprivation (already estimated), **trust in police (q59h)**, **trust in army (q59i)**, **govt performance on reducing crime (q65f)**

**Status:** Spatial pipeline implemented (`analysis_did_bh.py`). Healthcare deprivation results available. Trust and crime performance outcomes for R5 should be added.

**Key caveat:** Borno/Yobe/Adamawa states have **zero respondents in R6** — Afrobarometer could not access these states due to the active insurgency. This is a selection bias issue for R6 but also powerful descriptive evidence of the conflict's severity.

### Recommended Secondary Design: R5 Cross-Section with BH Exposure

**Specification:** Cross-sectional OLS for R5 only.
- Treatment: Log(BH kernel exposure) + Log(general violence exposure) in prior 12 months
- Outcomes: trust in police, trust in army, fear of crime, govt performance on reducing crime
- Controls: state FE, urban, age, gender, education, employment
- Clustered SE at state level

This design cannot make causal claims but tests the association between spatial proximity to BH violence and security perceptions/institutional trust. With n=2,400 and 36 states, power is adequate for moderate effects.

### Additional Analysis: Notes-Based Kidnapping as BH Sub-Type

For R5, construct a notes-based kidnapping indicator distinguishing:
1. Pure abduction events (sub_event_type = "Abduction/forced disappearance")
2. Attacks with kidnapping in notes (notes contains "kidnap" AND actor1 contains "Boko Haram")
3. Compare effect sizes: does kidnapping-specific BH exposure predict different outcomes than general BH violence?

---

## 6. Threats to Identification

1. **ACLED urban coverage bias (critical for all designs):** Pre-2016 ACLED data over-represents urban areas (newspaper/media-sourced). Respondents near recorded events are more likely to be urban, and urban residents have both better health outcomes AND better institutional trust. This creates a spurious *negative* correlation between violence exposure and deprivation — already documented in the R2 mental health and R5 healthcare deprivation analyses. **Mitigation:** Restrict to post-2011 BH analysis (BH events are geographically specific to northeast, not correlated with urbanization per se); include binary urban control; use state FE.

2. **Endogenous fieldwork timing:** Interviewers may avoid recently-affected areas, creating post-event under-sampling. Cannot assess without fieldwork metadata.

3. **IDP selection (R5/R6 BH zone):** Population displaced by BH violence is systematically absent from the survey frame. The observed NE respondents are "stayers" — potentially higher trust in security forces than those who fled. Estimated effects for BH zone should be read as effects on the non-displaced population.

4. **Parallel trends (DiD):** With only R4 as the pre-period, parallel trends cannot be tested statistically. Descriptive check: BH core states (Borno/Yobe/Adamawa) and non-BH states had similar healthcare deprivation in R4 (0.258 vs 0.268) — consistent with, though not proof of, parallel pre-trends.

5. **Measurement error in geocoding:** Afrobarometer coordinates appear to be GPS-precise. ACLED geo_precision 2 (town centroid, 50% of abduction events) introduces measurement error of ~5–15km, biasing estimates toward zero. Robustness checks at multiple radii partially address this.

6. **Scale non-equivalence across rounds:** Trust in police is at Q59h in R5 and Q52h in R6 (identical wording, 0–3 scale). R2 trust items exist but may cover different institutional sub-items at different positions. Do not pool across rounds without explicit harmonization checks.

---

## 7. Required Additional Data

Ranked by impact on identification quality:

1. **Nigeria Watch data** (mentioned in brief): Nigeria Watch reportedly covers criminal violence with much better local-level coverage pre-2010, including street crime and kidnapping. This would be essential for R1–R2 analyses and for distinguishing criminal from political violence in that period.

2. **Afrobarometer fieldwork metadata:** PSU visit dates and interviewer ID by cluster would enable the short-run design, allow clustering at the enumerator level, and test for fieldwork endogeneity.

3. **Chibok and other named BH kidnapping events:** The April 2014 Chibok schoolgirl abduction (276 girls) is the paradigmatic Nigerian kidnapping event but occurs *after* R5 fieldwork (Jan 2013) and before usable R6 data. A targeted event-study for R6 (if dates can be recovered) around Chibok would be high-impact.

4. **NBS crime statistics (National Bureau of Statistics):** State-level reported crime rates for triangulation of ACLED exposure measures.

5. **GDELT event data:** For pre-2010 period; broader media coverage than ACLED; could fill gaps in R1–R2 kidnapping exposure measurement.

---

## 8. Response to Brief's Five Final Questions

**1. Do these specific files support a credible short-run kidnapping exposure design?**
No. ACLED records zero abduction events during R2 and R4 fieldwork; only 9 events during R5 fieldwork across 36 states. This is structurally insufficient for any before/after contrast. The short-run design requires at least one order of magnitude more events proximate to fieldwork.

**2. Which rounds, units, windows, and outcomes produce meaningful variation?**
- **Round 5 only** for kidnapping-specific analysis (cumulative 12-month lookback, notes-based definition).
- **Rounds 4–6** for BH-specific DiD with trust outcomes; R5's Q59h (trust in police) and Q65f (government crime performance) are the key theory-central outcomes.
- No meaningful kidnapping variation in R1–R4.

**3. Main threats to identification and feasible falsification checks:**
- ACLED urban bias → compare urban/rural sub-samples; restrict to BH-specific events
- IDP selection → compare treatment effect size in core BH states (where selection is most severe) vs. secondary states
- Parallel trends → descriptive check of R4 baseline parity (already done); if R3 data can be geo-matched to states, add R3 as an additional pre-period point
- Future-event placebo → use 12-month *future* BH exposure (ACLED events in the 12 months *after* survey) as a falsification test; should show no effect on survey responses if identification is valid

**4. Best among five candidate designs:**
Design 5 framed as "BH insurgent violence (including kidnapping as a systematic tactic)" combined with DiD against the clean 2008 pre-period. This is the only design with sufficient treatment variation, a clean pre-period, and theory-relevant outcomes. It is implemented in `analysis_did_bh.py`.

**5. What to collect next:**
Nigeria Watch data is the single highest-priority addition — it would enable R1/R2 analyses with criminal violence and permit the criminal vs. insurgent comparison that gives the paper its theoretical contribution. Fieldwork metadata second (enables the short-run design the brief proposes as primary, at least for R3 and R5).

---

## Files

- `analysis_did_bh.py` — DiD implementation (R4–R6, BH exposure, healthcare deprivation outcome)
- `analysis_spatial.py` — Pooled R1–R6 spatial pipeline (all violence types)
- `analysis_r2_mental_health.py` — R2 mental health cross-section
- `did_maps.png` — Maps of BH events and deprivation change
- `did_results.png` — Event study and robustness plots
- `outcome_crosswalk.csv` — Variable-level crosswalk (see below)

---

*Next step: Add trust in police (q59h) and government crime performance (q65f) to the R5/R6 regression, then run the full preferred specification and append results to this report.*
