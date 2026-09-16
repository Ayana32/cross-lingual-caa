# Cross-Lingual CAA Sycophancy Steering

MSc Dissertation — University of Sheffield, 2026  
**Ayana (Misun Kim)** | Supervised by Dr. Cass Zhao & Dr. Marco Valentino

---

## Overview

This project investigates whether Contrastive Activation Addition (CAA) steering vectors extracted from English sycophancy data generalise across languages without target-language adaptation. Experiments are conducted on two models across five languages, with representational geometry analysis and random baseline validation.

---

## Research Questions

- **RQ1:** Do English-derived CAA steering vectors reduce sycophancy in non-English languages?
- **RQ2:** How does cross-lingual transfer vary across languages, layers, and model architectures?
- **RQ3:** Are the observed effects direction-specific, or artefacts of intervention magnitude?
- **RQ4:** What is the relationship between cross-lingual vector geometry and behavioural transfer?

---

## Models & Setup

| | Qwen3.5-9B | Gemma4-12B-IT |
|---|---|---|
| Checkpoint | `Qwen/Qwen3.5-9B` | `google/gemma-4-12b-it` |
| Multiplier | α = −1.5 | α = −1.0 |
| Layer sweep | L10–L23 | L20–L33 |
| Primary layer | L15 | L26 |
| Framework | EasyEdit2 | EasyEdit2 |
| HPC | Stanage (Sheffield) | Stanage (Sheffield) |

---

## Languages & Datasets

**Task:** Sycophancy subset of [Model-Written Evaluations](https://github.com/anthropics/evals) (Perez et al., 2022)

| Code | Language | Train (vector) | Eval (held-out) |
|------|----------|---------------|-----------------|
| EN | English | 800 | 200 |
| IT | Italian | 800 | 200 |
| ZH | Chinese | 800 | 200 |
| KO | Korean | 800 | 200 |
| KK | Kazakh | 800 | 200 |

Translations generated with GPT-4o. Kazakh dataset corrected for systematic mistranslation (86 items).  
Train/eval split: seed=42, random shuffle.

---

## Repository Structure

```
cross-lingual-caa/
├── data/
│ └── sycophancy/ # 800/200 train/eval split per language
├── cross_lingual/
│ ├── analyze_sweep.py # Layer sweep analysis (paired intersection)
│ ├── parse_utils.py # Output parsing utilities
│ └── submit_*.sh # HPC job submission scripts
├── geometry/
│ ├── pca_analysis_qwen35.py # Cosine + norm analysis (Qwen3.5)
│ ├── pca_analysis_gemma4.py # Cosine + norm analysis (Gemma4)
│ ├── cosine_delta_scatter.py # Geometry-behaviour scatter (Qwen3.5)
│ ├── cosine_delta_scatter_gemma4.py # Geometry-behaviour scatter (Gemma4)
│ └── plot_geometry_figures.py # Heatmap + norm figures (both models)
├── random_baseline/
│ ├── run_control.py # Control vector application
│ ├── submit_control.sh # HPC array job submission
│ ├── analyze_random_baseline_qwen35.py
│ ├── analyze_random_baseline_gemma4.py
│ └── configs/ # Base configs for control experiments
├── hparams/
│ └── Steer/ # EasyEdit2 configs
└── steering/
└── vectors_generate.py # CAA vector extraction
```


---

## Key Results (Sycophancy, primary layer)

### ΔSR at English-selected layer (negative = sycophancy reduced)

| Language | Qwen3.5 L15 | Gemma4 L26 |
|----------|-------------|------------|
| EN | −18.5 pp | −22.2 pp |
| IT | −21.1 pp | −14.2 pp |
| ZH | −24.9 pp | −14.1 pp |
| KO | −19.1 pp | −14.7 pp |
| KK | −19.3 pp | −15.1 pp |

All bootstrap 95% CIs exclude zero (paired bootstrap, 2,000 resamples).

### Native Vector Comparison

- **Qwen3.5-9B**: No statistically reliable difference between EN-derived and native vectors (all CIs include zero).
- **Gemma4-12B-IT**: EN-derived vector outperforms native vectors in all four target languages (all CIs exclude zero).

---

## Conda Environment

```bash
conda activate easyedit2
export PYTHONPATH=/users/acp25mk/EasyEdit2:${PYTHONPATH:-}
```

