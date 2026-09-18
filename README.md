# Cross-Lingual Activation Steering in Large Language Models

**Representational Alignment and Behavioural Transfer**

MSc Dissertation, University of Sheffield (2026)
MSc Speech and Natural Language Processing
Supervisors: Dr Cass Zhao, Dr Marco Valentino

---

## Overview

Alignment methods for large language models are usually built and validated in English. This project asks whether an activation-steering direction extracted from English data continues to work when the same model is used in another language, and whether representational geometry can predict that.

I extract Contrastive Activation Addition (CAA) steering vectors from English sycophancy data and apply them without modification to English, Italian, Chinese, Korean, and Kazakh inputs across three open-weight model families. A separate cross-task evaluation uses English-derived truthfulness vectors on a forced-choice TruthfulQA task.

**Headline finding: high cross-lingual representational similarity is not sufficient for behavioural transfer.** Steering directions that are strongly aligned across languages (cosine 0.92 to 0.99) produce uneven and sometimes opposite behavioural effects, and the pattern reverses across tasks within the same model.

---

## Research questions

1. Do English-derived steering vectors reduce sycophancy zero-shot in typologically diverse target languages?
2. Is cross-lingual transfer consistent across model families?
3. Does transfer generalise from sycophancy to a different behaviour (truthfulness)?
4. Do language-specific vectors outperform transferred English vectors?
5. Are the observed effects direction-specific rather than a consequence of generic activation perturbation?
6. Does cross-lingual cosine similarity or vector norm predict behavioural effect size?

---

## Setup

| | |
|---|---|
| **Method** | Contrastive Activation Addition (Rimsky et al.) |
| **Models** | Qwen3.5-9B, Gemma4-12B-IT, OLMo3-7B-Instruct |
| **Languages** | English, Italian, Chinese, Korean, Kazakh |
| **Primary behaviour** | Sycophancy (Perez et al., Model-Written Evaluations) |
| **Cross-task** | Forced-choice TruthfulQA |
| **Data split** | 1,000 items; 800 train (vector extraction) / 200 held-out eval, seed 42 |
| **Layer ranges** | Qwen3.5 L13–18, Gemma4 L24–28, OLMo3 L12–16 |
| **Multipliers** | α = −1.5 (Qwen3.5, OLMo3), α = −1.0 (Gemma4) |
| **Infrastructure** | PyTorch, HuggingFace Transformers, SLURM on Stanage HPC |

Layer and multiplier selection used **English results only**. No target-language outcome informed any tuning decision.

---

## Key results

### Sycophancy steering at the English-selected reference layer

Change in sycophancy rate (ΔSR, percentage points). Negative values indicate reduced sycophancy.

| Language | Qwen3.5-9B (L15) | Gemma4-12B-IT (L26) | OLMo3-7B-Instruct (L14) |
|---|---|---|---|
| English | −18.5 | −22.2 | −14.2 |
| Italian | −21.1 | −14.2 | +6.0 |
| Chinese | −24.9 | −14.1 | −4.5 |
| Korean | −19.1 | −14.7 | +2.0 |
| Kazakh | −19.3 | −15.1 | +5.0 |

Bootstrap 95% CIs exclude zero for all five languages in Qwen3.5-9B and Gemma4-12B-IT. In OLMo3-7B-Instruct only the English interval excludes zero.

### Cross-task reversal

Change in truthfulness rate (ΔTR, pp) on forced-choice TruthfulQA:

| Language | Qwen3.5-9B (L18) | OLMo3-7B-Instruct (L14) |
|---|---|---|
| English | +3.0 | +6.0 |
| Italian | +0.0 | **+9.0** |
| Chinese | +1.0 | **+9.0** |
| Korean | −1.0 | +4.0 |
| Kazakh | −1.0 | **+12.0** |

Bold marks intervals excluding zero. Gemma4-12B-IT is excluded because of a baseline ceiling effect (89–91%).

The two models invert: Qwen3.5-9B transfers sycophancy steering reliably but shows no reliable cross-lingual truthfulness transfer. OLMo3-7B-Instruct fails to transfer sycophancy beyond English but produces reliable truthfulness gains in three of four target languages. Cross-lingual transfer is behaviour-dependent within a model and model-dependent within a behaviour.

### Representation geometry

Mean EN-target cosine similarity:

| Model | Range | Behavioural transfer |
|---|---|---|
| Qwen3.5-9B | 0.918 (KK) to 0.981 (ZH) | Reliable, all languages |
| Gemma4-12B-IT | 0.961 (KK) to 0.987 (IT) | Reliable, all languages |
| OLMo3-7B-Instruct | 0.382 (KK) to 0.838 (ZH) | English only |

Neither cosine similarity nor L2 norm consistently tracks steering magnitude or optimal intervention depth. Within-language Spearman correlations between layer-wise cosine similarity and |ΔSR| range from ρ = 0.77 (ZH, Qwen3.5) to ρ = −0.47 (KO, OLMo3).

### Control experiments

Norm-matched Gaussian and permutation controls (20 seeds each) confirm direction-specificity. In Qwen3.5-9B at L15, controls produce mean ΔSR between −2.97 and +1.64 pp against −18.5 to −24.9 pp for the learned vector. In Gemma4-12B-IT the same perturbations collapse parseable generation entirely, which is itself evidence that the learned direction is not an arbitrary norm-matched displacement.

### English-derived vs language-specific vectors

Extracting a vector directly in the target language did not help. In Qwen3.5-9B no gap was statistically reliable. In Gemma4-12B-IT the English-derived direction was **stronger** than the native direction in all four target languages (gaps +3.3 to +7.5 pp, all CIs excluding zero).

---

## Repository structure

```
steering/              CAA vector extraction (vectors_generate.py, steering.py)
cross_lingual/         Sweep submission and behavioural analysis
geometry/              PCA, cosine similarity, vector norm analysis
random_baseline/       Norm-matched Gaussian and permutation controls
analysis/              Bootstrap CIs, aggregate result tables
hparams/               Hydra configs for every reported run
data/
  sycophancy/          EN + 4 target languages, train/eval splits
  truthfulqa/          Forced-choice TruthfulQA, 5 languages
translation_review/    Manual translation audits (15 fixed indices per language)
results/               Raw per-run outputs and geometry analysis
figures/               Figures appearing in the dissertation
slurm/                 HPC job submission scripts
utils/                 Translation pipeline and coverage analysis
docs/                  Pipeline documentation
```

---

## Reproducing the experiments

1. **Extract steering vectors**
   `steering/run_generate_qwen35.sh`, `run_generate_gemma4.sh`, `run_generate_olmo3.sh`. Configs live in `hparams/Steer/experiment_hparams/sycophancy_experiment/`.

2. **Run the layer sweep**
   `cross_lingual/submit_qwen35_syco_multilang_sweep.sh` and the equivalent Gemma4 and OLMo3 scripts.

3. **Parse and score**
   `cross_lingual/evaluate_results.py` uses the language-aware parser in `parse_utils.py`. Baseline and steered outputs are parsed independently; an item is dropped from paired calculations if either is unparseable.

4. **Bootstrap intervals**
   `analysis/bootstrap_ci.py` and `analysis/bootstrap_ci_tqa.py`.

5. **Controls**
   `random_baseline/make_random_baseline_vectors.py`, then `random_baseline/submit_control.sh`, then `random_baseline/score_controls.py`.

6. **Geometry**
   `geometry/pca_analysis_*.py` and `geometry/cosine_delta_scatter_*.py`.

Generated steering vectors are not committed. Running step 1 reproduces them.

---

## Methodological notes

**Translation pipeline.** An initial NLLB-200-distilled-1.3B pass produced mistranslated agreement expressions, dropped persona information, and corrupted answer labels. GPT-4o was adopted for the final translations after supervisory review. All 1,000 items per language passed automated structural verification. A 15-item sample per language, drawn from fixed seed-42 indices, was manually audited; the audit forms are in `translation_review/`. One systematic Kazakh mistranslation (a term meaning "neutrality" used where "bias" was intended) affected 86 items and was corrected before any experimental run.

**Positional bias control.** Unsteered (A)-selection rates in OLMo3-7B-Instruct reached 69.0% for Italian and 68.0% for Korean. Answer-order swap experiments dropped these to 47.0% and 47.5%, with item-level swap consistency of 55.0% and 61.5%. The non-English OLMo3 results are therefore reported as a boundary case rather than as clean behavioural measurement.

**Metric direction.** The dataset field `answer_matching_behavior` marks the sycophantic answer, not the correct one. The metric is sycophancy rate, where high means more sycophantic, so a successful intervention produces a negative delta.

**Parseability.** Paired parseable sample sizes are reported alongside every behavioural result so that apparent reductions cannot be produced by selective exclusion of unparseable steered outputs.

---

## What this contributes

The dissertation separates three properties that are often conflated: whether a behavioural direction can be recovered, whether it is geometrically stable across languages, and whether intervening along it produces the intended behavioural effect. These diverge in multilingual settings.

The practical consequence is that cross-lingual reliability of an alignment intervention has to be established through behavioural evaluation in each target language. Geometric proxies do not substitute for it, and evidence from one behaviour does not license claims about another.

---

## Contact

Misun Kim
MSc Speech and Natural Language Processing, University of Sheffield
[misunkim@sheffield.ac.uk](mailto:misunkim@sheffield.ac.uk)
