<!-- ## Explainability Testing with LLMs on AutoML

This repository is designed to test and observe the explainability capability of different Large Language Models (LLMs) in the context of AutoML.
It provides an environment to experiment, evaluate, and compare how various models generate explanations for automated machine learning processes.

## Installation
Python 3.10 needs to be installed in the preamble

Follow the steps below to set up and run the project:

1. Clone the Repository
```
- git clone https://github.com/amelinenar/explanation-experiment-llm.git
- cd your-repo-name
```
2. Install Dependencies
```
pip install -r requirements.txt
```
3. Configure Environment Variables
Fill in the required values inside the .env file:
```
URL_API=' '
ROOT_DIR=' '
```

##  Objective

Evaluate how different LLMs explain AutoML workflows.
Test clarity, accuracy, and interpretability of model-generated explanations.
Provide a reproducible framework for researchers and developers to experiment.


## How to run
Variable list:

CLASSIFICATION_DATASET = ['299_libras_move','185_baseball_dataset', '1491_one_hundred_plants', '1567_poker_hand' ]
REGRESSION_DATASET = ['advertising_dataset','534_cps_85_wages', '196_autoMpg', '26_radon_seed_dataset']
SEMI_SUPERVISED_DATASET = ['LL0_1053_jm1', 'SEMI_155_pokerhand_dataset', 'SEMI_1217_click_dataset', 'SEMI_1459_artificial_characters_dataset']
TIMESERIES_DATASET = ['stock_market', '56_sunspots_monthly', 'LL1_crime_chicago', '57_sunspot']
TASK = ['REGRESSION', 'CLASSIFICATION', 'SEMISUPERVISED' , 'TIME_SERIES_FORECAST']
LLMs = {'deepseek-r1:14b','llama4', 'gpt-4.1-mini'}
PROMPT = ['summarization_prompt', 'judging_prompt']
# SUMMARIZATION_PROMPT = ['zeroshot', 'zeroshot_instruction', 'fewshot', 'chain_of_thought','zeroshot_CoT','fewshot+instruction']
SUMMARIZATION_PROMPT = ['Hierachical prompt']
HIERARCHICAL_PROMPT = ['Hierachical prompt']
#HIERARCHICAL_PROMPT = ['phase_segmentation', 'micro_data_summarization', 'micro_search_space', 'micro_evaluation_prompt', 'meso_pattern_prompt', 'macro_narrative_prompt', 'verification_prompt']
JUDGING_PROMPT = ['judge_zeroshot']
#JUDGING_PROMPT = ['judge_zeroshot', 'judge_zeroshot+instruction']
#prompt_for_promptType ={'summarization'}
LLMs_judge = { 'deepseek-r1:14b','llama4', 'gpt-4.1-mini'}


.env file 
URL_API=' '
FILE_PATH=''
ROOT_DIR=''
KEY=''  ( for gpt-4.1-mini)


### Flat Prompting

- 
Parameters: fit, summarize, judge, run_all

a) fit parameter

To fit the notebook or python file : python main.py fit
The parameter can be modify on the constant.py file. experiment_setup/utils/constant.py.  
The logs for autosklearn is available in the folder autosklearn_logs for the CLASSIFICATION and REGRESSION task.  




To summarize  : python main.py summarize 
To judge: python main.py judge




### Hierarchical Prompting 




 -->


# Explainability of LLMs for AutoML Workflows

## Overview

This repository evaluates the explainability capabilities of different Large Language Models (LLMs) in the context of Automated Machine Learning (AutoML).

The framework:

- Runs AutoML experiments
- Generates natural language summaries using LLMs
- Evaluates the quality of generated explanations using judge models
- Two prompt strategies were used  **Flat Prompting** and **Hierarchical Prompting** 

The project is designed for reproducible research on LLM-based explainability for AutoML systems.

---

# Features

- AutoML experiment execution
- Flat prompting summarization pipeline
- Hierarchical summarization pipeline
- LLM-based evaluation/judging framework
- Multi-dataset experimentation
- Support for multiple LLMs
- CSV result aggregation

---

# Project Structure

```bash
.
├── main.py
├── prompt.py
├── result_script.py
├── pipeline/
│   └── Hierarchical_pipeline.py
├── utils/
│   ├── utils.py
│   └── constant.py
├── autosklearn_logs/
├── datasets/
├── requirements.txt
└── .env
```

---

# Requirements

- Python 3.10+
- pip

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/amelinenar/explanation-experiment-llm.git
cd explanation-experiment-llm
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Configure Environment Variables

Create a `.env` file in the root directory:

```env
URL_API=""
ROOT_DIR=""
KEY=""   # Required for OpenAI models such as gpt-4.1-mini
```

### Environment Variables

| Variable | Description |
|---|---|
| `URL_API` | API endpoint for local or remote LLM service |
| `ROOT_DIR` | Root directory of the project |
| `KEY` | OpenAI API key |

---

# Supported Tasks

```python
TASK = [
    "REGRESSION",
    "CLASSIFICATION",
    "SEMISUPERVISED",
    "TIME_SERIES_FORECAST"
]
```

---

# Supported Datasets

## Classification

```python
[
    '299_libras_move',
    '185_baseball_dataset',
    '1491_one_hundred_plants',
    '1567_poker_hand'
]
```

## Regression

```python
[
    'advertising_dataset',
    '534_cps_85_wages',
    '196_autoMpg',
    '26_radon_seed_dataset'
]
```

## Semi-Supervised

```python
[
    'LL0_1053_jm1',
    'SEMI_155_pokerhand_dataset',
    'SEMI_1217_click_dataset',
    'SEMI_1459_artificial_characters_dataset'
]
```

## Time Series

```python
[
    'stock_market',
    '56_sunspots_monthly',
    'LL1_crime_chicago',
    '57_sunspot'
]
```

---

# Supported LLMs

```python
LLMs = {
    'deepseek-r1:14b',
    'llama4',
    'gpt-4.1-mini'
}
```

Judge models:

```python
LLMs_judge = {
    'deepseek-r1:14b',
    'llama4',
    'gpt-4.1-mini'
}
```

---

# Prompting Strategies

## Flat Prompting

Supported summarization prompts:

```python
SUMMARIZATION_FLAT_PROMPT = [
    'zeroshot',
    'zeroshot_instruction',
    'fewshot',
    'chain_of_thought',
    'zeroshot_CoT',
    'fewshot+instruction'
]
```

## Hierarchical Prompting

Pipeline stages:

1. Fact Extraction
2. Macro Summarization
3. Verification
4. Revised Summary Generation

---

# Running the Project

## 1. Run Full Pipeline

### Flat Prompting Pipeline

```bash
python main.py run_all_FlatPrompting
```

This command will:

- Train AutoML models
- Generate summaries
- Evaluate summaries

---

### Hierarchical Prompting Pipeline

```bash
python main.py run_all_HierarchicalPrompting
```

This command will:

- Generate hierarchical summaries
- Verify summaries
- Evaluate summaries

---

# Individual Commands

## Train AutoML Models

```bash
python main.py fit
```

---

## Train a Specific Dataset

```bash
python main.py fit_1 <TASK_NAME> <DATASET_NAME>
```

Example:

```bash
python main.py fit_1 TIME_SERIES_FORECAST stock_market
```

---

# Flat Prompting

## Generate Summaries

```bash
python main.py flat_summarization
```

## Evaluate Summaries

```bash
python main.py judge
```

---

# Hierarchical Prompting

## Generate Hierarchical Summaries

```bash
python main.py Hierachical_summarization
```

## Evaluate Hierarchical Summaries

```bash
python main.py judge_H
```

---

# Generate CSV Results

```bash
python main.py generate_csv_file
```

This aggregates evaluation results into:

```bash
result.csv
```

---

# Output Structure

## Flat Prompting Results

```bash
results/
└── <AUTOML>/
    └── <TASK>/
        └── <DATASET>/
            └── <LLM>/
                └── <PROMPT>/
```

## Hierarchical Prompting Results

```bash
results_Hierarchical_Prompting/
└── <AUTOML>/
    └── <TASK>/
        └── <DATASET>/
            └── <LLM>/
```

---

# Evaluation Framework

The repository evaluates explanations generated by LLMs using judge models.

Evaluation dimensions may include:

- Clarity
- Accuracy
- Completeness
- Relevance


Evaluation outputs are saved as:

```bash
evaluation_<judge_model>.txt
```

---

# Notes

- `AUTOSKLEARN` is only supported for:
  - Classification
  - Regression

- Time series forecasting is skipped in some pipelines.

- Existing AutoSklearn logs are stored in:

```bash
autosklearn_logs/
```

---

# Example Workflow

```bash
# Step 1 — Train models
python main.py fit

# Step 2 — Generate flat summaries
python main.py flat_summarization

# Step 3 — Evaluate summaries
python main.py judge

# Step 4 — Aggregate results
python main.py generate_csv_file
```

---

<!-- # Citation

```bibtex
@misc{llm_automl_explainability,
  title={Explainability Evaluation of LLMs for AutoML Workflows},
  author={Your Name},
  year={2026}
}
```

---

# License

MIT License

---

# Future Improvements

- Add more LLM backends
- Support additional AutoML frameworks
- Add visualization dashboards
- Introduce quantitative explainability metrics
- Improve hierarchical verification strategies -->