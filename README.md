# LLM Failure Patterns Analysis

## Overview
This repository contains the code, benchmark data, visualizations, and paper for a research project studying failure patterns in large language models across attention and hard learning benchmarks.

The project investigates whether failures in attention-intensive settings and learning-intensive settings arise from separate weaknesses or from a shared set of structural vulnerabilities. The analysis focuses on factors such as instructional interference, target position, conflict strength, and model identity.

## Research Question
Do large language models fail in fundamentally different ways across attention and learning tasks, or are many of these failures driven by the same structural factors?

## Repository Contents

### Paper
- `Understanding Failure Patterns in LLM.pdf`  
  Research paper presenting the unified analysis, methodology, results, and conclusions.

### Notebook
- `do-llms-fail-the-same-way-a-unified-study.ipynb`  
  Main notebook used for analysis, experimentation, and result development.

### Core Scripts
- `generate_attention_benchmark.py`  
  Generates the attention benchmark with controlled prompt perturbations and evaluation-ready outputs.

- `generate_learning_hard_benchmark.py`  
  Generates the hard learning benchmark for belief updates, delayed supervision, exception rules, and conflicting mappings.

- `run_learning_hard_eval_all_models.py`  
  Runs evaluation across multiple LLMs on the hard learning benchmark and saves model outputs.

- `failure_model.py`  
  Supports predictive modeling and structured failure analysis.

- `analysis_plots.py`  
  Generates performance visualizations and summary plots.

- `combined_failure_analysis.py`  
  Combines benchmark outputs into a unified analysis framework.

- `learning_failure_prediction.py`  
  Builds predictive models for failure analysis using benchmark metadata.

### Benchmark Data
- `attention_benchmark_v8_full.csv`
- `learning_hard_benchmark_v1.csv`

These files contain the benchmark datasets used for evaluation and analysis.

### Key Figures
- `attention_vs_learning_heatmap.png`
- `noise_heatmap.png`
- `rf_feature_importance.png`
- `logistic_coefficients.png`
- `attention_model_accuracy.png`
- `learning_model_accuracy.png`

These figures summarize major findings from the project, including cross-benchmark performance trends, feature importance, and model-level robustness differences.

## Methodology
This project combines:
- controlled benchmark generation
- prompt perturbation analysis
- multi-model evaluation
- descriptive statistics
- visualization
- classical machine learning for failure prediction

The goal is not just to measure average performance, but to understand whether failures are systematic, predictable, and structurally organized.

## Main Findings
Key findings from the study include:
- instructional interference is one of the strongest and most consistent sources of degradation
- end-position target placement significantly reduces performance
- model robustness remains broadly consistent across benchmark families
- many failure outcomes can be predicted using structural metadata alone

## Models Evaluated
The broader study evaluates multiple open-source LLM families, including:
- Gemma
- Llama
- Qwen
- Mistral

## Skills Demonstrated
- Python
- Benchmark Design
- LLM Evaluation
- Prompt Robustness Analysis
- Data Analysis
- Predictive Modeling
- Statistical Analysis
- Data Visualization
- Research Communication

## Purpose
This repository is intended to showcase a research-oriented AI/ML project that combines empirical evaluation, analysis, and reproducible experimentation to better understand LLM reliability.
