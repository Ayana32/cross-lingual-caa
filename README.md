# Cross-Lingual Generalisation of LLM Steering Interventions

**MSc Dissertation — University of Sheffield, 2025–2026**  
Supervised by **Dr. Cass Zhao** and **Dr. Marco Valentino**

## Overview

This repository provides a public overview of my MSc dissertation on the cross-lingual generalisation of activation-based LLM steering interventions.

The project investigates whether behavioural steering directions learned in English transfer reliably across languages, and why similar internal steering directions may produce different behavioural effects across linguistic and model settings.

The work focuses on multilingual LLM alignment, representation steering, behavioural evaluation, and the relationship between internal representation geometry and downstream model behaviour.

## Research Questions

- Do English-derived steering vectors transfer zero-shot across languages?
- Are behavioural effects consistent across model families and linguistic settings?
- How do English-derived vectors compare with language-specific steering vectors?
- Are observed effects direction-specific, rather than consequences of generic activation perturbation?
- How does representational similarity relate to behavioural transfer?
- Do observed transfer patterns generalise across different behavioural and factuality tasks?

## Research Scope

- **Method:** Contrastive Activation Addition and activation-based representation steering
- **Models:** Multiple open-weight LLM families
- **Languages:** English, Italian, Chinese, Korean, and Kazakh
- **Primary behaviour:** Sycophancy
- **Additional evaluations:** Formal reasoning and multilingual TruthfulQA
- **Analysis:** Cross-lingual transfer, native-vector comparison, statistical controls, and representation geometry
- **Infrastructure:** PyTorch, Hugging Face Transformers, Linux HPC, and SLURM

## Experimental Framework

The dissertation work includes:

- English-derived steering-vector extraction
- Layer-wise intervention experiments
- Zero-shot cross-lingual behavioural evaluation
- Language-specific vector comparisons
- Paired bootstrap confidence intervals
- Norm-matched Gaussian and permutation controls
- Cross-model comparison
- Representation-similarity analysis
- Multilingual dataset construction and translation-quality review
- Cross-task evaluation on formal reasoning

## Current Findings

The completed experiments provide evidence that activation-based behavioural steering can transfer across typologically diverse languages and multiple model families.

The findings indicate that:

- transfer strength varies across languages and models;
- representational similarity alone does not fully explain behavioural outcomes;
- English-derived and language-specific vectors may differ in effectiveness;
- steering effects are substantially stronger than matched random and permutation controls;
- behavioural interventions may transfer beyond the original extraction task.

Detailed numerical results and model-specific analyses are withheld while the work is being prepared for publication.

## Ongoing TruthfulQA Extension

The project is being extended to multilingual TruthfulQA to examine whether cross-lingual steering patterns generalise from sycophancy and formal reasoning to factuality and truthfulness evaluation.

Current work includes:

- reviewing multilingual translations for semantic fidelity and label consistency;
- aligning evaluation items across languages using shared indices;
- correcting translation and annotation issues before experimentation;
- developing a shared multilingual evaluation protocol with research collaborators;
- assessing common open-weight models across languages and tasks.

TruthfulQA results are not yet included because dataset validation and experimental preparation are ongoing.

## Research Status

🚧 **Ongoing research**

- Core MSc dissertation experiments completed
- Cross-model, cross-lingual, native-vector, and control analyses completed
- TruthfulQA dataset validation and evaluation extension in progress
- Work being extended into a collaborative manuscript with a five-person research team
- Detailed results and selected artefacts will be released after publication decisions are finalised

## Technology Stack

**Research:** Python · PyTorch · Hugging Face Transformers · NumPy · pandas  
**Evaluation:** Behavioural evaluation · Bootstrap analysis · Representation analysis · Translation-quality review  
**Infrastructure:** Linux · SLURM · HPC · Git · Weights & Biases

## Availability

This repository currently contains only a public research overview. Experimental code, datasets, numerical results, and manuscript materials are not publicly released while the work is being prepared for publication.

Selected artefacts may be released following dissertation completion and publication decisions, subject to collaboration, licensing, and data-sharing constraints.

## Contact

For research discussion or collaboration:

**Misun Kim**  
MSc Speech and Natural Language Processing  
University of Sheffield  
misunkim@sheffield.ac.uk
