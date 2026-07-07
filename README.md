# Orthogonal Residual Concept Bottleneck Models (O-CBM+R) for Time-Series Discovery

This repository contains the official PyTorch implementation for the paper:
**"Identifiability and Generalization Bounds of Orthogonal Residual Concept Bottleneck Models for Time-Series Discovery"** (JMLR 2026).

## Overview
Concept Bottleneck Models (CBMs) offer interpretability but suffer from severe accuracy degradation in complex environments like financial time-series when unobserved causal factors exist. 
We propose O-CBM+R, which adds an unsupervised residual pathway constrained by an **Orthogonal Penalty (Cosine Similarity)**. This geometrically forces the residual subspace to discover novel, unmapped market micro-structures without leaking the predefined human concepts.

## Key Highlights
- **Identifiability & Generalization**: Mathematically bounds concept leakage and guarantees that the residual subspace uniquely captures unobserved causal factors.
- **Performance Recovery**: Recovered the predictive accuracy drop of Strict CBMs on the massive FI-2010 dataset from $22.63\%$ back to $68.17\%$ without sacrificing interpretability.
- **Optimized Data Pipeline**: Custom zero-RAM data pipeline using numpy memory mapping (`mmap_mode='r'`) and raw string parsing to parse highly-transposed data (149 rows, 400,000+ columns) instantaneously without Out-Of-Memory crashes.

## Dataset Preparation
1. Download the **FI-2010 Limit Order Book Dataset**.
2. Extract the dataset such that the directory structure is `fi2010_data/extracted/BenchmarkDatasets/NoAuction/`.

## Installation

Ensure you have Python 3.9+ installed. Then, run:
```bash
pip install -r requirements.txt
```

## Running the Code

To reproduce the ablation study (ERM vs Strict CBM vs O-CBM+R), execute:

```bash
python train_evaluate.py
```

During the first run, the data pipeline will automatically convert the highly inefficient `.txt` files into `.npy` binary caches. Subsequent runs will use zero-RAM memory mapping.
