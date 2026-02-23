# Unsupervised Anomaly Detection in LHC 40 MHz Trigger Data

> **TL;DR**: This repo implements a **fully unsupervised deep autoencoder** on the  
> **“LHC physics dataset for unsupervised New Physics detection at 40 MHz”**  
> (Govorkova *et al.*, Scientific Data 2022, CERN/FNAL).  
> Trained only on **Standard Model background**, the model reaches  
> **ROC–AUC ≈ 0.99** on all four benchmark signals and produces a ranked  
> list of the most anomalous events in the blind blackbox sample.

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Dataset & Physics Context](#2-dataset--physics-context)
- [3. Methodology](#3-methodology)
  - [3.1 Preprocessing](#31-preprocessing)
  - [3.2 Model Architecture](#32-model-architecture)
  - [3.3 Training Setup](#33-training-setup)
- [4. Results](#4-results)
- [5. Limitations & Future Work](#5-limitations--future-work)
- [6. How to Run](#6-how-to-run)
- [7. References](#7-references)
- [8. Acknowledgements](#8-acknowledgements)

---

## 1. Overview

The goal of this project is to study **unsupervised anomaly detection** for  
**New Physics at the LHC**, using the public benchmark:

> *LHC physics dataset for unsupervised New Physics detection at 40 MHz*  
> by E. Govorkova, E. Puljak, T. Aarrestad, M. Pierini,  
> K. A. Woźniak, and J. Ngadiuba (CERN, Univ. of Vienna, Fermilab, Caltech).

In this work, we:

- Train a **fully-connected autoencoder** on **background-only** events
- Use **reconstruction error** as an anomaly score
- Evaluate on four labeled New Physics benchmarks
- Rank events from a **blind blackbox sample** and output the **Top-K anomalies**

Despite being implemented on **Google Colab with limited RAM, storage, and session time**,  
this simple baseline shows that an AE can already separate signal from background very well,  
and highlight promising rare events in a realistic trigger-like data stream.

---

## 2. Dataset & Physics Context

At the **Large Hadron Collider (LHC)**:

- Proton–proton collisions occur at rates of order **10⁸ per second**
- The raw detector output is **tens of terabytes per second**
- Experiments like ATLAS and CMS must reduce this to **≈ 10³ events/s** for storage using **real-time trigger systems**

Traditional triggers are optimized for **known signatures** (e.g., high-pT leptons, missing energy, invariant mass windows).  
This raises a key concern:

> *What if truly unexpected New Physics does not match any of these patterns and gets discarded?*

Govorkova *et al.* propose a dataset and challenge where:

- Events emulate a **Hardware/Level-1 trigger stream**
- They are **pre-filtered** by requiring at least one high-pT electron or muon
- Each event contains up to **19 reconstructed objects**:
  - MET
  - Up to 4 electrons
  - Up to 4 muons
  - Up to 10 jets
- Each object is described by 4 features:
  - \( p_T \), \( \eta \), \( \phi \), and an **object-type ID**
- Missing objects are **zero-padded**, yielding arrays of shape `(N_events, 19, 4)`

The dataset includes:

- A **background “cocktail”** (`BKG_dataset.h5`) with splits:
  - `X_train`, `X_val`, `X_test` – treated as **Standard Model only**
- Four **labeled New Physics benchmark signals**:
  - A → 4ℓ  
  - h⁰ → ττ  
  - h± → τν  
  - Leptoquark (LQ) → bτ
- A **blackbox sample** (mixture of background + hidden signals) with no labels, intended for **blind evaluation**

Our work follows the spirit of the **ADC2021 community challenge** associated with this dataset.

---

## 3. Methodology

### 3.1 Preprocessing

For each event, we construct a **57-dimensional feature vector**:

1. **Feature selection**

   - Use only the first **3 features** per object:
     - \( p_T \), \( \eta \), \( \phi \)
   - Ignore the 4th object-type channel in this baseline
   - With 19 objects × 3 features → **19 × 3 = 57 features**

2. **Physics-motivated scaling**

   - Apply log scaling to transverse momentum:
     \[
     p_T \leftarrow \log(1 + \max(p_T, 0))
     \]

3. **Flattening and normalization**

   - Flatten events from `(19, 3)` to `(57,)`
   - Compute **mean and standard deviation** on the background training set
   - Normalize all data (train/val/test, signals, blackbox) via:
     \[
     x_{\text{norm}} = (x - \mu) / \sigma
     \]

Note: We experimented with more advanced preprocessing  
(e.g., φ → (sin φ, cos φ) and masking padded entries using the object-type flag),  
but due to time and resource constraints these were not fully integrated in the final stable baseline.

---

### 3.2 Model Architecture

We use a **fully-connected autoencoder**:

- **Input**: 57-D normalized feature vector

- **Encoder**:
  - Dense(256, ReLU)
  - Dense(128, ReLU)
  - Dense(64, ReLU)
  - Bottleneck: Dense(32, ReLU)

- **Decoder** (symmetrical):
  - Dense(64, ReLU)
  - Dense(128, ReLU)
  - Dense(256, ReLU)
  - Output: Dense(57, linear)

The autoencoder is trained **only on background events**, so it learns the  
“typical” Standard Model manifold, without ever seeing signal labels.

---

### 3.3 Training Setup

- Loss: **Mean Squared Error (MSE)** between input and reconstructed output
- Optimizer: **Adam**
  - Initial learning rate: `1e-3`
  - Learning rate reduced on plateau (e.g. halved when validation loss stalls)
- Training:
  - ~**50 epochs**
  - Batch size: **1024**
  - Train on `X_train`, validate on `X_val`
- Implementation: TensorFlow/Keras, developed in **Google Colab**

We also set random seeds (Python, NumPy, TensorFlow) and deterministic settings where possible  
to improve reproducibility.

---

## 4. Results

### 4.1 Background reconstruction error

On the background test sample (`X_test`):

- **Min score** ≈ 2.2 × 10⁻⁵  
- **Mean score** ≈ 1.86 × 10⁻²  
- **Max score** ≈ 3.25 × 10²  

Only a small fraction of background events have very large reconstruction error,  
consistent with a few hard outliers.

### 4.2 Signal separation

For each signal benchmark, we form:

- `y = 0` for background test events  
- `y = 1` for signal events  
- `score = MSE reconstruction error`

We then compute the **ROC–AUC**:

- A → 4ℓ:        **≈ 0.993**  
- h⁰ → ττ:       **≈ 0.979**  
- h± → τν:       **≈ 0.988**  
- LQ → bτ:       **≈ 0.987**

Combined **Precision–Recall AUC** (background + all signals): **≈ 0.955**

This shows that a simple fully-connected AE, trained without labels,  
can rank signal events above background extremely well.

### 4.3 TPR at fixed FPR

We also study **trigger-like operating points**:

- TPR @ FPR = 10⁻³  → **≈ 0.094**  
- TPR @ FPR = 10⁻⁴  → **≈ 0.0099**  
- TPR @ FPR = 10⁻⁵  → **≈ 0.00245**

Interpretation:

- At **0.1% background false-positive rate**, we recover about **9%** of signal events.
- Pushing FPR down to 10⁻⁴–10⁻⁵ selects only the **most extreme tail** of the signal distribution.

Crucially, the background and signal score distributions **overlap**, so there is **no single threshold** that:

- flags **all** signal events as anomalies,  
- rejects **all** background, and  
- also yields a fixed small fraction of anomalies in the blackbox.

### 4.4 Blackbox ranking

For the **blind blackbox sample**:

- We compute the anomaly score for each event
- Sort events by score (descending)
- Extract the **Top-K most anomalous events**

We provide a file (e.g.):

- `adc2021_top1000.csv` – containing the **IDs of the 1,000 most anomalous blackbox events**,  
  following the ADC2021 challenge format.

You can adjust `K` to control how many candidate events you want to inspect.

---

## 5. Limitations & Future Work

This project was carried out entirely on **free Google Colab**, which imposes:

- Limited **RAM** and **disk space**
- Short **GPU session time** (often tens of minutes to ~1–2 hours)
- No persistent high-end compute environment

Because of these constraints, we focused on obtaining a **clean, well-understood baseline**  
rather than exploring very large or complex models.

With more compute and guidance, the next steps would be:

1. **Physics-informed preprocessing**
   - Use the **object-type flag** to build masks and ignore padded entries in the loss
   - Encode φ as **(sin φ, cos φ)** to handle periodicity
   - Explore per-object or per-class loss weighting (e.g., give more weight to leptons)

2. **Richer models**
   - Set/graph-based architectures that treat the 19 objects as a permutation-invariant set
   - Variational or denoising autoencoders
   - Normalizing flows or other density-based anomaly scores

3. **Trigger-oriented deployment**
   - Study latency and resource usage under realistic trigger constraints
   - Explore model compression and FPGA mapping (e.g. via **hls4ml**)
   - Co-design algorithms with hardware and bandwidth limits in mind, as emphasized in the original paper.

---

## 6. How to Run

> **Note:** The dataset itself is not hosted in this repo.  
> You must download it from the official sources (Govorkova *et al.* / ADC2021).

1. **Clone the repository**

```bash
git clone <your-github-url>.git
cd <your-repo-name>
