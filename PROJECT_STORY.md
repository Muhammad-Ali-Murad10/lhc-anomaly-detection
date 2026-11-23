# Professor Interview — LHC Anomaly Detection Research Story

## Short explanation (about 45 seconds)

> I became interested in a simple question: if the LHC collects enormous numbers of collision events, could an AI model help flag unusual events without knowing beforehand what new physics would look like? I used the public ADC2021 simulated dataset to investigate this with an unsupervised autoencoder. Initially, some results looked extremely strong, but when I reviewed the data handling, I realized they were not reliable enough. I checked the dataset checksum, separated training, validation, calibration, and testing, and reran the experiment. The corrected model achieved ROC-AUC scores from 0.595 to 0.849 across four signals. I learned that a high ROC-AUC is not enough: when targeting just one false positive in 100,000 events, calibration becomes very difficult. The most interesting next question is why an Isolation Forest ranked anomalies better than the autoencoder in my results, and whether that advantage remains at strict false-positive rates.

## If the professor asks, “What was your contribution?”

I implemented and debugged the benchmark pipeline; verified the official training dataset; isolated the data splits to avoid leakage; evaluated four simulated signal processes; calibrated anomaly thresholds on held-out background; and audited the actual false-positive counts and uncertainty. The result is a **carefully documented baseline reproduction**, not a new algorithm or new-physics discovery.

## If they ask, “What did you learn?”

1. **Correct data handling matters more than impressive-looking metrics.** Earlier runs were not trustworthy enough to support strong claims.
2. **ROC-AUC and rare-event detection are different questions.** A model can rank reasonably well overall but still fail to keep a very strict false-positive budget.
3. **Independent calibration must be checked.** At nominal FPR 10⁻⁵, the held-out test observed **45 false positives out of 2,690,384 background events**, or **1.673 × 10⁻⁵**.
4. **Unexpected comparisons can create the next research question.** A latent-space Isolation Forest showed better ROC-AUC on all four benchmark signals, but I still need to measure its low-FPR efficiency before concluding it is better.

## If they ask, “What would you do next?”

First, repeat the experiment with multiple seeds and save per-event scores and split identifiers. Second, evaluate the Isolation Forest at **the same independently calibrated false-positive rates** as the autoencoder. Third, inspect the high-score background events to see whether physics-aware features or padded particles explain the failures. Only then would I propose and test model changes.

## Clear claim boundary

This project uses **simulated, publicly released particle-collision events**, not a live CERN detector stream. It demonstrates an independently evaluated **baseline** and a meaningful research question; it does **not** demonstrate discovery of new physics, proven novelty, or deployment on the LHC trigger.

**Evidence:** [Experiment notebook](notebooks/LHC_ADC2021_Strict_Baseline_Executed.ipynb) · [Results](RESULTS.md) · [Reliability audit](reliability_audit/AUDIT_REPORT.md)
