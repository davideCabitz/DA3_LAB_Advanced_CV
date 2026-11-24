# Foveated DA3 (enhancement #2): pre-registered statistical analysis plan

**Version 1, fixed on 2026-11-22, before any Kaggle run of `notebooks/06_foveated`.** The numbers
live in [configs/foveated_analysis.yaml](../configs/foveated_analysis.yaml). Every result record
stores that file's SHA-256, and the analysis warns when a run was made under a different plan. The
git history is the timestamp. After the results are in, the plan is not edited: a changed plan
gets a new version, and both analyses are reported.

Method and protocols: [da3_enhancement_report.md](da3_enhancement_report.md) §4.3 and
[reports/enhancement_2_foveated.md](../reports/enhancement_2_foveated.md). Code:
`src/da3/enhancements/foveated/{stats,analysis}.py`.

## 1. Questions and hypotheses

| Gate | Question (report §4.3) | Primary hypothesis | Primary comparison |
|---|---|---|---|
| G1a | Does lowering the resolution cost anything (headroom)? | All views at 252 px have a higher AbsRel than at 504 px, by more than the margin | AbsRel of all views, u252 / u504 |
| G1b | Do low-resolution context views carry information? | The focus views are more accurate with the context views than alone, by more than the margin | focus-view AbsRel, foveated / focus views alone |
| G3 | Does mixing resolutions in one pass work? | C1: the focus views are non-inferior to all-504. C2: the context views' poses are non-inferior to all-252 | C1: focus AbsRel, fov50_504-252 / u504. C2: context-pair AUC@30, fov50_504-252 / u252 |
| G2 | Does a virtual zoom view add detail? | C1: region AbsRel at least 10% better than a global 896-px pass. C2: DA3 recovers the zoom rotation within 1° | C1: zoom (DA3 cameras through E_c) / global 896 on oracle regions. C2: median R_v error |

G1 = G1a AND G1b; G3 = C1 AND C2; G2 = C1 AND C2. An AND of criteria needs no multiplicity
correction (intersection–union test).

## 2. Unit of analysis and uncertainty

- **The scene is the unit.** Views of a scene, and the several view subsets drawn from one scene,
  are correlated. Every quantity is first averaged within a scene; scenes are the independent
  observations.
- **Paired comparisons.** Both arms run on the same scenes and the same views. Ratios are ratios of
  scene means (test / reference).
- **95% percentile bootstrap over scenes** (4000 resamples, seed 0). The Wilcoxon signed-rank
  p-value of the per-scene pairs is reported too, but it never decides.

## 3. Decision rule: three outcomes, read off the CI

| Criterion type | PASS | FAIL | INCONCLUSIVE |
|---|---|---|---|
| non-inferiority, lower is better (G3 C1, G2 C1/C2): ratio ≤ 1 + m | upper CI bound ≤ 1 + m | lower bound > 1 + m | otherwise |
| non-inferiority, higher is better (G3 C2): ratio ≥ 1 − m | lower bound ≥ 1 − m | upper bound < 1 − m | otherwise |
| effect beyond the margin (G1a): ratio > 1 + m | lower bound > 1 + m | upper bound ≤ 1 + m (loss within the margin: "nearly lossless") | otherwise |
| improvement beyond the margin (G1b): ratio ≤ 1 − m | upper bound ≤ 1 − m | lower bound > 1 − m (no meaningful improvement: "context carries nothing") | otherwise |

A two-sided 95% CI bound is a one-sided test at 2.5%. INCONCLUSIVE is a real outcome: it says the
data cannot decide at this margin. It is reported as such and never rounded to PASS or FAIL.

## 4. Thresholds: chosen before the results, informed by data that cannot favour the method

Choosing a threshold after looking at the test result would make the test meaningless. The data
may inform a threshold only through quantities that do not involve the foveated results:

1. **Pre-registered margin m₀ = 3%** (relative). The report's G3 states "within 3%". The same
   tolerance applies to every non-inferiority or equivalence statement in this study (depth and
   pose, G1 and G3), so that one definition of "negligible" holds throughout.
2. **A/A noise floor (treatment-blind).** The original DA3 at 504 px is run twice on the same views,
   with only the reference view changed (the first view is moved to the middle focus view). This
   change is irrelevant by construction, so whatever it moves is noise of the method itself.
   - The noise floor of a metric family is the largest relative effect its 95% CI allows:
     max |CI bound − 1|.
   - A difference smaller than that cannot be called meaningful.
   - The margin used is therefore max(m₀, noise floor).
3. **Cap at 2 × m₀ = 6%.** A noisy measurement must give wider CIs, and so INCONCLUSIVE, never a
   looser criterion. A noise floor above the cap is flagged: the protocol cannot resolve the
   pre-registered tolerance.
4. **Full threshold sensitivity.** For each criterion the two margins at which its decision would
   change are reported (e.g. "PASS if margin ≥ 1.2%, FAIL if margin < 0.4%"). Readers can see the
   decision at any threshold without anything being chosen afterwards.

G2's thresholds (0.90 ratio, 1°) are the report's own pre-registered numbers and are not adjusted.

## 5. Multiplicity and selection

- **One primary configuration per gate,** as specified in the report: 8 views at 504 px and 8 at
  252 px for G3, and oracle regions for G2.
- **Secondary configurations** get simultaneous 97.5% CIs (Bonferroni over the two):
  - the report's mitigation, 280-px context views;
  - our mitigation, attention-mass balancing.
  They are reported as secondary and do not change the primary decision.
- **Robustness (reported, no decision):**
  - the primary configuration with 3 random focus selections instead of evenly spaced ones;
  - for G2, regions chosen randomly or by image texture instead of by GT detail.
- **Rescue analysis (pre-declared, the only selection step):** only if the primary G3 does not PASS.
  - The scenes are split 50/50 by a fixed seed, stratified by dataset.
  - On the dev half, the mitigation with the lowest C1 estimate among those whose C2 estimate is
    within the margin is selected. Point estimates only.
  - Its G3 decision is then made on the test half alone.
  - The primary result stays as it is. The rescue is reported next to it.

## 6. Sample size

- **Probe:** HiRoom (30 scenes) and 7Scenes (7), with 3 interleaved 16-view subsets per scene.
  - The test unit is the scene (37).
  - The subsets reduce within-scene noise; they are not extra samples.
- **Large-view protocol:** all 7 7Scenes scenes, 100–1000 views. The first 2 scenes are timed 3
  times (median); the rest once.
- **Region zoom:** all 11 ETH3D scenes, 2 source views each, one region per selector.

With only 7 scenes in the large-view protocol and 11 in the zoom, CIs will be wide. That is
reported, not fixed by changing the unit.

## 7. Descriptive analyses (no gate)

- **Sec. 6 runs:** paired scene-bootstrap CIs of every foveated run vs the original DA3 and vs the
  uniform-resolution curve at the same token count. The curve is interpolated per scene in log
  tokens.
- **Large-view protocol:** foveated vs each alternative at each view count, with CIs for the
  focus-pair AUC@30 difference and the focus-view AbsRel ratio, plus timing medians and ranges.

## 8. Reporting rules

- Every gate is reported, including FAIL and INCONCLUSIVE, with estimate, CI, margin and the margins
  at which the decision changes.
- Probe, large-view and zoom numbers are never mixed into Sec. 6 tables.
- Any deviation from this plan is listed below with its reason.

## Deviations log

**1. 2026-11-23: limited run inspected before the full run; exploratory diagnostics added.**
- What happened: a limited end-to-end run of `notebooks/06_foveated` (2 scenes per dataset, Sec. 6
  pose only, 100 / 250 views) was looked at. Its numbers are in
  [reports/enhancement_2_foveated.md](../reports/enhancement_2_foveated.md) §4, labelled as a smoke
  test.
- What did **not** change: `configs/foveated_analysis.yaml` (same SHA-256), the gates, the primary
  and secondary configurations, the margins and the rescue rule. The confirmatory decisions are read
  from the full run (`MAX_SCENES = 0`) exactly as specified above.
- What was added after seeing the limited run. Everything below is **exploratory**: descriptive, CIs
  only, no decision, reported in a separate block of `summary.md` and in columns marked `*`.
  - Cross-pair pose AUC (focus–context pairs), with rotation-only and translation-only errors. The
    limited run suggested this is the largest effect, and no gate covers it.
  - Per-view depth scale and AbsRel with one scale per resolution group.
  - The full-resolution context ceiling next to G1b (all views at 504 px vs the focus views alone).
  - Context views with 252-px detail on the 504-px grid (`diag_up252`), to tell a patch-scale
    mismatch from lost detail.
  - A context-resolution sweep (336 / 378 px), never gated.
  - The two-pass control (all views at 252 px + the focus views alone at 504 px, aligned on the focus
    cameras), in the probe and in the large-view protocol.
- Sec. 6 runs pose only (no TSDF / F1). The limited run showed that the two resolution groups come out
  at different depth scales, so fusing them would mostly measure that. F1 stays a descriptive
  quantity of the plan; it is not reported for this enhancement.
