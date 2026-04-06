# Results Summary

## 1. Dataset summary

| Measure | Value |
|---|---:|
| BlackBox `Particles` shape | (4,210,492, 19, 4) |
| Background `X_train` | (2,560,000, 57) |
| Background `X_val` | (640,000, 57) |
| Background `X_test` | (800,000, 57) |
| Final input shape | (2,560,000, 57) |
| Combined array shape shown | (3,360,000, 57) |

## 2. Preprocessing summary

The notebook preprocessing does the following:

- keeps the first 3 features (`pT`, `eta`, `phi`)
- applies log scaling to `pT`
- flattens the event representation to 57 dimensions
- normalizes using training mean and standard deviation

## 3. Model summary

| Item | Value |
|---|---|
| Model type | Dense autoencoder |
| Optimizer | Adam(1e-3) |
| Loss | MSE |
| Epochs | 50 |
| Batch size | 2048 |
| Scheduler | ReduceLROnPlateau |

## 4. Late training-stage values shown in output

| Epoch | Train loss | Val loss | Learning rate |
|---|---:|---:|---:|
| 44 | 0.0448 | 0.0365 | 3.9063e-06 |
| 45 | 0.0445 | 0.0365 | 3.9063e-06 |
| 46 | 0.0442 | 0.0365 | 1.9531e-06 |
| 47 | 0.0449 | 0.0365 | 1.9531e-06 |
| 48 | 0.0449 | 0.0365 | 1.9531e-06 |
| 49 | 0.0448 | 0.0365 | 9.7656e-07 |
| 50 | 0.0456 | 0.0365 | 9.7656e-07 |

## 5. Background anomaly-score summary

One output block reports:

| Statistic | Value |
|---|---:|
| Min | 2.2199405e-05 |
| Mean | 0.018628767 |
| Max | 325.1223 |

Another output block reports:

| Statistic | Value |
|---|---:|
| Min | 0.0010068659 |
| Mean | 0.06552155 |
| Max | 8799.588 |

These values are both present in the notebook outputs.

## 6. Signal ROC-AUC results

| Signal file | ROC-AUC |
|---|---:|
| Ato4l_lepFilter_13TeV.h5 | 0.9929 |
| hToTauTau_13TeV_PU20.h5 | 0.9788 |
| hChToTauNu_13TeV_PU20.h5 | 0.9878 |
| leptoquark_LOWMASS_lepFilter_13TeV.h5 | 0.9872 |

## 7. Operating-point results

| Measure | Value |
|---|---:|
| Optimal threshold shown | 0.00685768062248826 |
| TPR at FPR 0.001 | 0.09423158240932156 |

## 8. Isolation Forest latent-space results

| Measure | Value |
|---|---:|
| Anomalies detected in test set | 817 |
| Anomalies detected in signal set | 1334 |

## 9. Interpretation

The experiment shows that:

- the autoencoder separates background from benchmark signals well in terms of overall ROC-AUC
- all reported benchmark ROC-AUC values are high, between 0.9788 and 0.9929
- low-FPR operating points are much harder than the overall ROC-AUC suggests
- the pipeline was also applied to blackbox data, not only labeled signal-vs-background comparisons

The main conclusion is that the method is effective as a ranking-based anomaly detector, but strict trigger-like constraints remain difficult.
