# Unsupervised LHC Anomaly Detection — ADC2021 Baseline and Reliability Audit

**Research reproduction | Simulation benchmark | Background-only training**

This repository contains a reproducible *baseline experiment* for detecting unusual particle-collision events in the [ADC2021](https://mpp-hep.github.io/ADC2021/) dataset, together with a statistical reliability audit.

**Status:** Baseline and initial audit completed. Independent-seed replications, event-level score exports, and a calibrated low-FPR comparison of alternative models are **not yet completed**. This is **not** a new-physics discovery, a deployed LHC trigger, or an established novel algorithm.

> **Main finding:** A dense autoencoder reaches ROC-AUC **0.595–0.849** across four simulated Beyond Standard Model (BSM) signals. Its nominal 10⁻⁵ calibration threshold produces an observed held-out background FPR of **1.673 × 10⁻⁵**, so strict-FPR detection performance cannot be claimed at exactly 10⁻⁵. An exploratory latent-space Isolation Forest improves ROC-AUC across all four signals; its low-FPR performance remains unmeasured.


## Quick navigation

- **[Executed benchmark notebook](notebooks/LHC_ADC2021_Strict_Baseline_Executed.ipynb)** — actual code, training, and outputs
- **[Verified results and caveats](RESULTS.md)** — what the run measured
- **[Reliability audit](reliability_audit/AUDIT_REPORT.md)** — strict-FPR checks and uncertainty
- **[Interview research story](PROJECT_STORY.md)** — motivation, lessons, and next steps

## 1. Why this experiment?

At high event rates, selecting unusual collisions while rejecting Standard Model (SM) background requires both effective anomaly ranking and reliable false-positive control. This project asks:

1. What can a background-only dense autoencoder detect across four different simulated BSM processes?
2. Does a threshold calibrated on independent SM background meet its target false-positive rate on held-out SM data?
3. Does latent-space Isolation Forest merit further evaluation?

The benchmark comes from Govorkova et al.'s public LHC simulation dataset. No experimental real-time detector data are used here.

## 2. Data and provenance

Use the **official v2** records linked by the [ADC2021 challenge](https://mpp-hep.github.io/ADC2021/):

| Role | Official Zenodo record |
|---|---|
| SM background training sample | [5046428](https://zenodo.org/records/5046428) |
| A → 4 leptons | [7152590](https://zenodo.org/records/7152590) |
| h⁰ → ττ | [7152614](https://zenodo.org/records/7152614) |
| h⁺ → τν | [7152617](https://zenodo.org/records/7152617) |
| Leptoquark → bτ | [7152599](https://zenodo.org/records/7152599) |
| Unlabeled blackbox | [5072068](https://zenodo.org/records/5072068) |

The executed notebook verifies the official SM-background file against MD5:

`08cb4801229f9d480567f8d3f5cac134`

It records **13,451,915** background events, partitioned with seed 42 as follows:

| Partition | Events | Purpose |
|---|---:|---|
| Training | 6,725,957 | Fit scaler and autoencoder |
| Model validation | 1,345,191 | Early stopping / model selection |
| Threshold calibration | 2,690,383 | Choose anomaly-score thresholds |
| Background test | 2,690,384 | Independent FPR estimation |

The four labeled BSM samples are used for **evaluation**, not autoencoder training or threshold calibration. The unlabeled blackbox is used only for ranking events.

**Data are not redistributed in GitHub.** Download them from Zenodo, follow the corresponding usage conditions, and attach them to a Kaggle notebook or update the input paths.

## 3. Method

### 3.1 Features and preprocessing

Events contain up to 19 physics objects and four original attributes per object. The baseline keeps `pT`, `eta`, and `phi`, drops the original type channel, applies `log1p(max(pT, 0))`, and flattens the event into **57 features**. Mean and standard deviation are learned from **training background only** and reused unchanged at evaluation.

### 3.2 Dense autoencoder

- Architecture: **57 → 256 → Dropout(0.30) → 128 → 64 → 128 → 256 → 57**
- Reconstruction loss: mean squared error (MSE)
- Optimizer: Adam, starting learning rate `0.001`
- Batch size: **2,048**; up to **60** epochs with early stopping
- Best validation epoch in this run: **29**, best validation MSE ≈ **0.02165**
- Anomaly score: mean squared reconstruction error per event
- The 64-unit latent layer is **not a compressive bottleneck** relative to 57 inputs.

For low-FPR estimates, thresholds are derived only from the held-out **calibration background** and assessed separately on the **test background**. An exploratory Isolation Forest is fitted on latent representations of a subset of training-background events.

## 4. Results (single executed run, seed 42)

### 4.1 Discrimination across BSM benchmarks

| Signal | AE ROC-AUC | Isolation Forest ROC-AUC | AE TPR at nominal 10⁻⁵ calibration threshold |
|---|---:|---:|---:|
| A → 4 leptons | 0.8490 | 0.8820 | 3.2750% |
| h⁰ → ττ | 0.5953 | 0.6567 | 0.0974% |
| h⁺ → τν | 0.6289 | 0.8296 | 0.0629% |
| Leptoquark → bτ | 0.6881 | 0.7798 | 0.0593% |
| **Unweighted macro mean** | **0.6903** | **0.7870** | — |

**Important:** The last column reflects a threshold **calibrated toward** a nominal FPR of 10⁻⁵, **not** a threshold proven to achieve test FPR ≤ 10⁻⁵. Isolation Forest has not yet been evaluated using independently calibrated low-FPR thresholds; its ROC-AUC advantage is exploratory.

The signal-vs-background class proportions differ by process, so average-precision values should be interpreted alongside their prevalence; see `results/signal_metrics.csv`.

### 4.2 Independent background FPR audit

| Nominal calibration target | Test false positives / test background | Observed test FPR | Approximate exact 95% interval |
|---|---|---|---|
| 10⁻³ | 2,810 / 2,690,384 | 1.0445 × 10⁻³ | [1.0062, 1.0838] × 10⁻³ |
| 10⁻⁴ | 289 / 2,690,384 | 1.0742 × 10⁻⁴ | [0.9539, 1.2054] × 10⁻⁴ |
| 10⁻⁵ | 45 / 2,690,384 | 1.6726 × 10⁻⁵ | [1.2200, 2.2381] × 10⁻⁵ |

The reported intervals treat test events as approximately independent and the calibration thresholds as fixed; systematic shifts and threshold-estimation uncertainty are not covered by these intervals. In this run, the strictest operating point exceeds its nominal FPR target.

### 4.3 Blackbox ranking

`results/blackbox_top_1e-5_fraction.csv` contains the selected unlabeled blackbox event IDs at a fraction of 10⁻⁵. The notebook also produces `blackbox_top_1000.csv` for exploratory ranking. **No blackbox ground truth is available here, so rankings do not establish a discovery or signal purity.**

## 5. Reliability, limitations, and responsible interpretation

- Only **one random seed** was evaluated; cross-seed variability is unknown.
- Calibration at **10⁻⁵** depends on only **26 calibration-background exceedances** in this run.
- Isolation Forest was evaluated by ROC-AUC and average precision but **not** by low-FPR TPR.
- Padded particle slots and the periodicity of `phi` are **not explicitly modeled** by this baseline.
- Original split indices and raw per-event score arrays were **not included** in the collected result bundle; the audit therefore cannot independently verify sample membership or recompute confidence intervals by resampling scores.
- Computational cost, trigger latency, and hardware deployment are **not established**.
- The audit is **an analysis of one completed run**, not independent replication or evidence of algorithmic novelty.

The `reliability_audit/AUDIT_REPORT.md` file details these checks.

## 6. Reproduce in Kaggle

1. Create a Kaggle notebook and attach the official ADC2021 v2 files above.
2. Open `notebooks/LHC_ADC2021_Strict_Baseline_Executed.ipynb`. The executable cells expect input datasets under `/kaggle/input/`; review the configuration cell if your paths differ.
3. Ensure there is enough temporary disk space for the downloaded background under `/kaggle/working/`; Kaggle `/kaggle/input/` is read-only. Enable notebook internet access if background download is required.
4. Run all cells in order. The default output directory is `/kaggle/working/lhc_adc2021_strict_baseline`.
5. To generate a reliability audit for the new run, open `notebooks/LHC_Reliability_Audit_AddOn.ipynb` and execute it after the baseline. The included `reliability_audit/` outputs reflect the documented historical run, not an automatic re-audit of a new run.

The executed notebook records TensorFlow **2.21.0** and visible GPUs. Exact reproducibility also requires the original data versions, dependency versions, device/runtime configuration, and saved split assignments; not all are captured in the existing run.

## 7. Repository structure

```text
.
├── README.md
├── RESULTS.md
├── PROJECT_STORY.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── LHC_ADC2021_Strict_Baseline_Executed.ipynb
│   └── LHC_Reliability_Audit_AddOn.ipynb
├── results/
│   ├── run_metadata.json
│   ├── signal_metrics.csv
│   ├── calibration_thresholds.csv
│   ├── isolation_forest_metrics.csv
│   └── blackbox_top_1e-5_fraction.csv
├── reliability_audit/
│   ├── AUDIT_REPORT.md
│   ├── AUDIT_SUMMARY.json
│   ├── fpr_reliability.csv
│   ├── model_comparison.csv
│   └── signal_efficiencies_with_achieved_fpr.csv
└── scripts/
    ├── LHC_Reliability_Audit_AddOn.py
    └── export_non_h5_kaggle.py
```

*GitHub does not contain the large dataset files. The saved model checkpoints and scaler are optional small artifacts that can be exported from Kaggle for restoration.*

## 8. Next research steps

The immediate next iteration is **reliability-focused**: export per-event scores and split identifiers, repeat the baseline across at least three independent seeds, and evaluate Isolation Forest under independently calibrated low-FPR thresholds. Only after these tests should physics-aware masking, angular features, or changes to latent dimensionality be considered controlled architectural ablations.

These plans are **not completed results** and do not imply guaranteed improvement.

## 9. References and attribution

- E. Govorkova, E. Puljak, T. Aarrestad, M. Pierini, K. A. Wozniak, and J. Ngadiuba. *LHC physics dataset for unsupervised New Physics detection at 40 MHz*. Scientific Data **9**, 2022. [arXiv:2107.02157](https://arxiv.org/abs/2107.02157).
- [ADC2021 benchmark and challenge](https://mpp-hep.github.io/ADC2021/).
- Dataset files and their respective authors, release/version information, and licenses: see the linked Zenodo records.

**Project implementation and reliability audit:** Muhammad Ali Murad. This is an independent benchmark reproduction using publicly available simulated data; it does **not** imply affiliation with CERN, Fermilab, or the dataset authors.

**Historical note:** Earlier exploratory notebooks and README versions reported substantially higher ROC-AUC values. Those results were not established under the verified and independently calibrated evaluation protocol reported here; they must not be mixed with this strict baseline.
