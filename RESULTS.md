# Results — verified ADC2021 baseline (seed 42)

These numbers come from `results/` and `reliability_audit/`, produced from the executed Kaggle notebook. They **replace** the earlier unverified ROC-AUC ~0.98–0.99 results for the purposes of the current benchmark.

## Summary

| Signal | AE ROC-AUC | Isolation Forest ROC-AUC |
|---|---:|---:|
| A → 4 leptons | 0.848979 | 0.882010 |
| h⁰ → ττ | 0.595309 | 0.656670 |
| h⁺ → τν | 0.628943 | 0.829646 |
| Leptoquark → bτ | 0.688069 | 0.779836 |

Unweighted mean ROC-AUC: AE **0.6903**, Isolation Forest **0.7870**. The latter has **not** been evaluated at calibrated low-FPR operating points.

## FPR calibration reliability

From 2,690,384 independent test-background events:

| Target | False positives | Measured FPR | 95% exact binomial CI |
|---|---:|---:|---|
| 1e-3 | 2,810 | 0.00104446 | [0.00100622, 0.00108379] |
| 1e-4 | 289 | 0.000107420 | [0.0000953913, 0.000120545] |
| 1e-5 | 45 | 0.0000167262 | [0.0000122003, 0.0000223810] |

At the nominal 1e-5 calibrated threshold, the background test FPR is **1.673e-5**, so do **not** present the corresponding signal TPR as achieved TPR at FPR=1e-5.

## What can be concluded?

The reproducible baseline demonstrates moderately variable discrimination across four simulated signal models, with a meaningful weakness at extreme false-positive requirements. The exploratory Isolation Forest has higher ROC-AUC across all four signals but lacks direct low-FPR comparisons. Only one seed was run. No novel algorithm, discovery, or deployable trigger performance has been established.

For full limitations and statistical assumptions see `reliability_audit/AUDIT_REPORT.md`. For configuration and data checks see `results/run_metadata.json`.
