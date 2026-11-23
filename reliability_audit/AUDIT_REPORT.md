# ADC2021 strict baseline — independent reliability audit

- Source: `/kaggle/working/lhc_adc2021_strict_baseline`
- Background checksum verified in notebook metadata: **True** (file not rehashed by this audit).
- Dataset overlap flag in notebook metadata: background_equals_blackbox = `False`.
- Background counts: `{'train': 6725957, 'model_validation': 1345191, 'calibration': 2690383, 'test': 2690384}`. Split indices not included in uploaded ZIP.
- Architecture bottleneck width = **64**, input width = **57**. This is not dimensional compression.
- Isolation Forest is compared here only by ROC-AUC and AP, because it lacks independently calibrated low-FPR results.
- Low-FPR uncertainty below assumes approximately independent test background events, with fixed thresholds learned from independent calibration data.

## Independent-background FPR checks
- Nominal 1e-03: observed **0.001044461** (2810/2690384); exact 95% CI [0.001006215, 0.001083786]; **above target (95% CI)**; one-sided excess p=0.01118.
- Nominal 1e-04: observed **0.0001074196** (289/2690384); exact 95% CI [9.539127e-05, 0.0001205448]; **inconclusive**; one-sided excess p=0.1185.
- Nominal 1e-05: observed **1.672624e-05** (45/2690384); exact 95% CI [1.220026e-05, 2.238096e-05]; **above target (95% CI)**; one-sided excess p=0.0008832.

## Model comparison, exploratory

- A_to_4leptons: AE ROC-AUC 0.8490; IF ROC-AUC 0.8820; IF−AE = +0.0330.
- h0_to_tautau: AE ROC-AUC 0.5953; IF ROC-AUC 0.6567; IF−AE = +0.0614.
- hplus_to_taunu: AE ROC-AUC 0.6289; IF ROC-AUC 0.8296; IF−AE = +0.2007.
- leptoquark_to_btau: AE ROC-AUC 0.6881; IF ROC-AUC 0.7798; IF−AE = +0.0918.

## Scientific interpretation and next checks

1. Do not label calibrated TPR as achieved TPR at a target FPR if the realized independent-test FPR is higher.
2. Repeat full training with at least 3 independent training seeds; retain exact split assignment, input checksums, environment and loss history.
3. Calibrate and test Isolation Forest at the same 3 FPR targets using disjoint background sets before claiming superiority.
4. Examine extreme false positives for per-object reconstruction contributions, zero padding, and extreme pT; this is diagnostic, not established novel research.
5. If testing reduced bottleneck size (e.g., 16 or 32), treat it as an ablation and reserve the signal/test data for final evaluation.
6. No raw per-event score vectors or original dataset files are included, so resampling, duplicate-event detection and signal-file checksums cannot be independently verified here.
7. Blackbox candidate ranks are unlabelled: they cannot establish detection sensitivity or discovery.

This audit is not itself a novel algorithm or a claim of publishable improvement.
