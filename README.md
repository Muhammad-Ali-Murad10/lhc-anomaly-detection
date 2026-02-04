# Unsupervised New Physics Detection using Deep Learning at the LHC

## Project Overview

This project implements a deep learning–based anomaly detection pipeline designed to identify potential signatures of new physics within proton collision events from the Large Hadron Collider (LHC).

Modern particle detectors produce enormous data streams — reaching tens of terabytes per second — requiring real-time filtering systems to determine which events should be retained for further study. :contentReference[oaicite:0]{index=0}  

Traditional searches rely heavily on supervised strategies targeting known physics signatures. However, such approaches risk missing unexpected phenomena. Recent research therefore explores **unsupervised anomaly detection**, where models learn typical event behavior and flag rare outliers that may correspond to undiscovered physical processes. :contentReference[oaicite:1]{index=1}  

This repository reproduces such a framework using an autoencoder neural network trained exclusively on Standard Model background events.

---

## Key Contributions

✔ Built a fully reproducible anomaly detection pipeline  
✔ Trained a deep autoencoder on millions of collision events  
✔ Implemented percentile-based anomaly thresholding  
✔ Successfully identified rare anomalous events within a mixed dataset  
✔ Structured the project following research-grade reproducibility practices  

---

## Dataset

This project uses the publicly available dataset introduced in:

**Govorkova et al., "LHC physics dataset for unsupervised New Physics detection at 40 MHz."**

The dataset emulates a realistic trigger-level data stream and is pre-filtered to include events containing at least one electron or muon. :contentReference[oaicite:2]{index=2}  

Each event contains particle-level physics features such as:

- Transverse momentum (pT)  
- Pseudorapidity (η)  
- Azimuthal angle (φ)  
- Missing transverse energy (MET)  

Events are zero-padded to maintain fixed input dimensions, mirroring real trigger systems. :contentReference[oaicite:3]{index=3}  

The full dataset contains over **8 million events**, including background processes and a hidden mixture of potential new physics signals. :contentReference[oaicite:4]{index=4}  

---

## ⚠️ Dataset Availability

Due to their large size (multiple gigabytes), datasets are **NOT stored in this repository**.

Download them from the official publication sources (Zenodo links referenced in the paper).

After downloading, organize your files as follows:


---

## Methodology

### Model Architecture

A dense autoencoder was designed to learn compressed representations of normal collision events.

**Input dimension:** 57 features  

Architecture:

57 → 128 → 64 → 16 → 64 → 128 → 57


Total parameters: **33,481**

The model was trained to minimize reconstruction error, enabling it to distinguish typical events from anomalous ones.

---

### Training

- Trained exclusively on background events  
- Optimized using Mean Squared Error (MSE)  
- Validated on held-out background samples  

**Final validation loss:** `0.0938`

---

## Anomaly Detection Strategy

After training:

1. Reconstruction errors were computed for background samples  
2. The anomaly score distribution was analyzed  
3. The threshold was defined at the **99.9th percentile**

Anomaly Threshold: 8.4339


Events exceeding this threshold were classified as anomalous.

---

## Results

### BlackBox Dataset Evaluation

| Metric | Value |
|--------|--------|
| Total Events | 4,210,492 |
| Detected Anomalies | 5,767 |
| Normal Events | 4,204,725 |
| Anomaly Rate | **0.14%** |

The small anomaly fraction aligns with real-world expectations, where potential new physics signatures are rare.

---

## Reproducibility

This repository is designed for full reproducibility:

✔ Fixed random seed  
✔ Configurable hyperparameters  
✔ Explicit dataset paths  
✔ Dependency versioning  

See:

requirements.txt
seed.txt
