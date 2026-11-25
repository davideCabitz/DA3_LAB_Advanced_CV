# Original DA3 vs enhancements

7 result records from: `C:\Users\Davide Cabitza\GitHub\DA3_LAB_Advanced_CV\experiments\results`

| Enhancement | Variant | Model | Tag | Setting |
|---|---|---|---|---|
| baseline | head | official:da3-small | sa_da3-small_hiroom-7scenes-dtu64_limited2 | official |
| baseline | ray_dlt | official:da3-small | sa_da3-small_hiroom-7scenes-dtu64_limited2__ray_dlt | official |
| baseline | ray_official | official:da3-small | sa_da3-small_hiroom-7scenes-dtu64_limited2__ray_official | official |
| self_audit | audit_fuse | official:da3-small | sa_da3-small_hiroom-7scenes-dtu64_limited2__audit_fuse | official |
| self_audit | audit_metrics | official:da3-small | sa_da3-small_hiroom-7scenes-dtu64_limited2__self_audit | official |
| self_audit | audit_select | official:da3-small | sa_da3-small_hiroom-7scenes-dtu64_limited2__audit_select | official |
| self_audit | oracle_select | official:da3-small | sa_da3-small_hiroom-7scenes-dtu64_limited2__oracle_select | official |

## Paper protocol (Sec. 6): original DA3 vs enhancements

#### da3-small · datasets 7scenes, dtu64, hiroom · scenes 6 (limited) · ≤ 100 frames
Differences in brackets are relative to the original DA3 (API default: camera-head poses). ▲ better, ▼ worse.

| Model | Enhancement | Variant | 7Scenes AUC@3 | 7Scenes AUC@30 | 7Scenes F1 w/o p. | DTU-64 AUC@3 | DTU-64 AUC@30 | HiRoom AUC@3 | HiRoom AUC@30 | HiRoom F1 w/o p. |
|---|---|---|---|---|---|---|---|---|---|---|
| da3-small | baseline | head | 29.3 | 88.3 | 50.9 | 40.7 | 93.1 | 3.9 | 70.3 | 9.4 |
| da3-small | baseline | ray_dlt | 22.6 (−6.7 ▼) | 87.1 (−1.2 ▼) | – | 1.9 (−38.8 ▼) | 79.6 (−13.5 ▼) | 1.0 (−2.9 ▼) | 60.9 (−9.4 ▼) | – |
| da3-small | baseline | ray_official | 32.4 (+3.2 ▲) | 89.2 (+0.9 ▲) | 53.8 (+2.9 ▲) | 33.2 (−7.5 ▼) | 91.9 (−1.2 ▼) | 3.6 (−0.4 ▼) | 70.5 (+0.2 ▲) | 17.4 (+8.1 ▲) |
| da3-small | self_audit | audit_fuse | 31.6 (+2.3 ▲) | 88.9 (+0.6 ▲) | 53.5 (+2.6 ▲) | 38.1 (−2.5 ▼) | 92.8 (−0.3 ▼) | 3.9 (−0.0) | 69.3 (−1.0 ▼) | 8.5 (−0.8 ▼) |
| da3-small | self_audit | audit_select | 29.4 (+0.1 ▲) | 88.3 (+0.0) | 50.3 (−0.6 ▼) | 35.1 (−5.6 ▼) | 92.1 (−1.0 ▼) | 2.8 (−1.2 ▼) | 67.2 (−3.2 ▼) | 11.3 (+2.0 ▲) |
| da3-small | self_audit | oracle_select | 33.9 (+4.7 ▲) | 89.5 (+1.2 ▲) | – | 43.3 (+2.7 ▲) | 93.4 (+0.3 ▲) | 3.7 (−0.2 ▼) | 72.8 (+2.5 ▲) | – |

## Enhancement-specific metrics: self_audit

### audit_model

| Model | Variant | Dataset | scheme | coef gap_rot_deg | coef gap_center | coef gap_focal | coef origin_spread | coef pinhole_resid_deg | coef neg_logconf_depth_mean | coef neg_logconf_depth_p10 | coef neg_logconf_ray_mean | coef reproj_disagree | coef ec_depth_change | coef ec_rot_change_deg | coef ec_ray_change_deg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| da3-small | audit_metrics | all | leave-one-dataset-out | -0.034 | 0.009 | -0.041 | 0.059 | 0.135 | 0.011 | 0.006 | -0.146 | 0.061 | 0.070 | 0.021 | 0.154 |

### depth_uncertainty

| Model | Variant | Dataset | scenes | coverage@50% | |error|@50% | coverage@90% | |error|@90% | NLL | AUSE (rel.) |
|---|---|---|---|---|---|---|---|---|---|
| da3-small | audit_metrics | 7Scenes · DA3 confidence + GT calibration from other datasets | 2.000 | 0.574 | 0.074 | 0.818 | 0.082 | 1.054 | – |
| da3-small | audit_metrics | 7Scenes · Self-Audit triple collocation (no GT) | 2.000 | 0.398 | 0.102 | 0.720 | 0.180 | 1.906 | – |
| da3-small | audit_metrics | 7Scenes · upper bound: per-scene GT fit | 2.000 | 0.700 | 0.200 | 0.918 | 0.019 | -0.840 | – |
| da3-small | audit_metrics | 7Scenes · AUSE of DA3 confidence (same for all) | 2.000 | – | – | – | – | – | 0.284 |
| da3-small | audit_metrics | HiRoom · DA3 confidence + GT calibration from other datasets | 2.000 | 0.948 | 0.448 | 0.997 | 0.097 | -0.641 | – |
| da3-small | audit_metrics | HiRoom · Self-Audit triple collocation (no GT) | 2.000 | 0.599 | 0.146 | 0.930 | 0.043 | -1.462 | – |
| da3-small | audit_metrics | HiRoom · upper bound: per-scene GT fit | 2.000 | 0.498 | 0.004 | 0.912 | 0.012 | -1.479 | – |
| da3-small | audit_metrics | HiRoom · AUSE of DA3 confidence (same for all) | 2.000 | – | – | – | – | – | 0.527 |
| da3-small | audit_metrics | All · DA3 confidence + GT calibration from other datasets | 4.000 | 0.761 | 0.261 | 0.907 | 0.090 | 0.207 | – |
| da3-small | audit_metrics | All · Self-Audit triple collocation (no GT) | 4.000 | 0.498 | 0.124 | 0.825 | 0.112 | 0.222 | – |
| da3-small | audit_metrics | All · upper bound: per-scene GT fit | 4.000 | 0.599 | 0.102 | 0.915 | 0.016 | -1.160 | – |
| da3-small | audit_metrics | All · AUSE of DA3 confidence (same for all) | 4.000 | – | – | – | – | – | 0.405 |

### failure_detection

| Model | Variant | Dataset | views | AUROC>15° | AUROC>5° | Spearman | AURC (deg) | gross failures |
|---|---|---|---|---|---|---|---|---|
| da3-small | audit_metrics | 7Scenes · DA3 confidence (original) | 200.000 | – | – | -0.092 | 3.050 | 0.000 |
| da3-small | audit_metrics | 7Scenes · Self-Audit | 200.000 | – | – | -0.001 | 3.153 | – |
| da3-small | audit_metrics | 7Scenes · Self-Audit w/o E_c residual | 200.000 | – | – | 0.307 | 2.686 | – |
| da3-small | audit_metrics | 7Scenes · oracle (sorted by true error) | 200.000 | – | – | – | 2.200 | – |
| da3-small | audit_metrics | DTU-64 · DA3 confidence (original) | 128.000 | – | – | 0.120 | 2.231 | 0.000 |
| da3-small | audit_metrics | DTU-64 · Self-Audit | 128.000 | – | – | 0.021 | 2.209 | – |
| da3-small | audit_metrics | DTU-64 · Self-Audit w/o E_c residual | 128.000 | – | – | 0.139 | 2.162 | – |
| da3-small | audit_metrics | DTU-64 · oracle (sorted by true error) | 128.000 | – | – | – | 1.877 | – |
| da3-small | audit_metrics | HiRoom · DA3 confidence (original) | 35.000 | 0.618 | 0.893 | 0.563 | 6.238 | 1.000 |
| da3-small | audit_metrics | HiRoom · Self-Audit | 35.000 | 0.853 | 0.800 | 0.363 | 6.342 | – |
| da3-small | audit_metrics | HiRoom · Self-Audit w/o E_c residual | 35.000 | 0.824 | 0.840 | 0.497 | 6.139 | – |
| da3-small | audit_metrics | HiRoom · oracle (sorted by true error) | 35.000 | – | – | – | 5.095 | – |
| da3-small | audit_metrics | All · DA3 confidence (original) | 363.000 | 0.948 | 0.988 | 0.133 | 2.779 | 1.000 |
| da3-small | audit_metrics | All · Self-Audit | 363.000 | 0.986 | 0.972 | 0.129 | 2.851 | – |
| da3-small | audit_metrics | All · Self-Audit w/o E_c residual | 363.000 | 0.983 | 0.987 | 0.384 | 2.580 | – |
| da3-small | audit_metrics | All · oracle (sorted by true error) | 363.000 | – | – | – | 2.106 | – |

### gates

| Model | Variant | Dataset | result | value | criterion |
|---|---|---|---|---|---|
| da3-small | audit_metrics | G1 detector signals | n.a. | 1/1 datasets; 7Scenes: no gross failures; DTU-64: no gross failures; HiRoom: gap_rot_deg ρ=0.67 AUROC=1.00, pinhole_resid_deg ρ=0.54 AUROC=0.79 | Spearman(head–ray gap or pinhole residual, rotation error) ≥ 0.3 and AUROC(>15°) ≥ 0.7 on ≥ 2 datasets |
| da3-small | audit_metrics | G2 pose-source choice | FAIL | 1/3 datasets; 7Scenes: best single 32.4, fused 31.6, oracle 33.9; DTU-64: best single 40.7, fused 38.1, oracle 43.3; HiRoom: best single 3.9, fused 3.9, oracle 3.7 | oracle − better single source ≥ 2 AUC@3, or fusion ≥ +0.5 AUC@3, on a majority of datasets |
| da3-small | audit_metrics | G3 GT-free calibration (F2) | FAIL | 62% of 16 confidence bins within [0.7, 1.4] (HiRoom); median ratio 1.35 | triple-collocation sigma / true sigma within 0.7–1.4 in ≥ 75% of confidence bins |
| da3-small | audit_metrics | G4 E_c fixed-point residual | FAIL | AUROC(>15°) 0.986 with vs 0.983 without the E_c signals (Δ +0.003) | adding the E_c residual raises the audit AUROC by ≥ 0.03 |

### pose_source

| Model | Variant | Dataset | scenes | AUC@3 | AUC@30 | views using ray pose (%) |
|---|---|---|---|---|---|---|
| da3-small | audit_metrics | 7Scenes · camera head (original, API default) | 2.000 | 29.253 | 88.300 | – |
| da3-small | audit_metrics | 7Scenes · ray map, official RANSAC (original option) | 2.000 | 32.414 | 89.235 | – |
| da3-small | audit_metrics | 7Scenes · ray map, our DLT | 2.000 | 22.556 | 87.139 | – |
| da3-small | audit_metrics | 7Scenes · Self-Audit per-view choice | 2.000 | 29.401 | 88.328 | 0.500 |
| da3-small | audit_metrics | 7Scenes · Self-Audit per-view fusion | 2.000 | 31.589 | 88.898 | – |
| da3-small | audit_metrics | 7Scenes · oracle per-view choice (uses GT) | 2.000 | 33.946 | 89.501 | – |
| da3-small | audit_metrics | DTU-64 · camera head (original, API default) | 2.000 | 40.683 | 93.094 | – |
| da3-small | audit_metrics | DTU-64 · ray map, official RANSAC (original option) | 2.000 | 33.201 | 91.892 | – |
| da3-small | audit_metrics | DTU-64 · ray map, our DLT | 2.000 | 1.910 | 79.624 | – |
| da3-small | audit_metrics | DTU-64 · Self-Audit per-view choice | 2.000 | 35.078 | 92.122 | 75.000 |
| da3-small | audit_metrics | DTU-64 · Self-Audit per-view fusion | 2.000 | 38.145 | 92.779 | – |
| da3-small | audit_metrics | DTU-64 · oracle per-view choice (uses GT) | 2.000 | 43.337 | 93.414 | – |
| da3-small | audit_metrics | HiRoom · camera head (original, API default) | 2.000 | 3.909 | 70.321 | – |
| da3-small | audit_metrics | HiRoom · ray map, official RANSAC (original option) | 2.000 | 3.557 | 70.503 | – |
| da3-small | audit_metrics | HiRoom · ray map, our DLT | 2.000 | 1.043 | 60.906 | – |
| da3-small | audit_metrics | HiRoom · Self-Audit per-view choice | 2.000 | 2.756 | 67.163 | 50.362 |
| da3-small | audit_metrics | HiRoom · Self-Audit per-view fusion | 2.000 | 3.887 | 69.275 | – |
| da3-small | audit_metrics | HiRoom · oracle per-view choice (uses GT) | 2.000 | 3.722 | 72.814 | – |
| da3-small | audit_metrics | All · camera head (original, API default) | 6.000 | 24.615 | 83.905 | – |
| da3-small | audit_metrics | All · ray map, official RANSAC (original option) | 6.000 | 23.058 | 83.877 | – |
| da3-small | audit_metrics | All · ray map, our DLT | 6.000 | 8.503 | 75.889 | – |
| da3-small | audit_metrics | All · Self-Audit per-view choice | 6.000 | 22.411 | 82.538 | 41.954 |
| da3-small | audit_metrics | All · Self-Audit per-view fusion | 6.000 | 24.540 | 83.651 | – |
| da3-small | audit_metrics | All · oracle per-view choice (uses GT) | 6.000 | 27.002 | 85.243 | – |

### signals

| Model | Variant | Dataset | Spearman | AUROC>15° | AUROC>5° | available (%) |
|---|---|---|---|---|---|---|
| da3-small | audit_metrics | gap_rot_deg | 0.251 | 1.000 | 0.996 | 100.000 |
| da3-small | audit_metrics | gap_center | 0.093 | 0.975 | 0.964 | 100.000 |
| da3-small | audit_metrics | gap_focal | 0.047 | 0.401 | 0.877 | 100.000 |
| da3-small | audit_metrics | origin_spread | 0.338 | 0.931 | 0.979 | 100.000 |
| da3-small | audit_metrics | pinhole_resid_deg | 0.392 | 0.981 | 0.984 | 100.000 |
| da3-small | audit_metrics | neg_logconf_depth_mean | 0.133 | 0.948 | 0.988 | 100.000 |
| da3-small | audit_metrics | neg_logconf_depth_p10 | 0.065 | 0.641 | 0.724 | 100.000 |
| da3-small | audit_metrics | neg_logconf_ray_mean | -0.045 | 0.978 | 0.990 | 100.000 |
| da3-small | audit_metrics | reproj_disagree | 0.452 | 0.992 | 0.989 | 100.000 |
| da3-small | audit_metrics | ec_depth_change | 0.493 | 0.901 | 0.908 | 100.000 |
| da3-small | audit_metrics | ec_rot_change_deg | 0.266 | 1.000 | 0.904 | 100.000 |
| da3-small | audit_metrics | ec_ray_change_deg | 0.341 | 1.000 | 0.995 | 100.000 |
