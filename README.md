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


## Implementation Setup

This explanation property was implemented in Python 3.10.

The experiments were conducted on a Linux operating system (Ubuntu 22.04.5 LTS) using the following hardware configuration:

- **CPU:** Intel® Core™ i7-8650U @ 1.90 GHz × 8  
- **Memory:** 16 GB RAM  
- **Disk Capacity:** 512.1 GB  

The open-source LLMs were remotely installed on a machine equipped with:

- **Processor:** Intel® Core™ i7-14700K @ 3.4 GHz

The experiment took 4 days. 


# Features

- AutoML experiment execution
- Flat prompting summarization pipeline
- Hierarchical summarization pipeline
- LLM-based evaluation/judging framework
- Multi-dataset experimentation
- Support for multiple LLMs
- CSV result aggregation
- Graph generation

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

- Python 3.10
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
URL_gpt = "https://api.openai.com/v1/responses" #(in case you use  gpt-4.1-mini )
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
#### Best Performing Model

Among the evaluated models, `gpt-4.1-mini` produced the best overall results in terms of:

- explanation quality
- clarity
- consistency
- evaluation scores

It achieved the strongest performance across several datasets and prompting strategies.



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


---

# Generate graph

```bash
python main.py generate_graph
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

- `AUTOSKLEARN` only support:
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
# Step 1 - Train models
python main.py fit

# Step 2 - Generate flat summaries
python main.py flat_summarization

# Step 3 - Generate Hierarchical summaries
python main.py Hierachical_summarization

# Step 4 - Evaluate flat summaries
python main.py judge

# Step 5 - Evaluate Hierarchical summaries
python main.py judge_H


# Step 6 - Aggregate results
python main.py generate_csv_file


# Step 7 - Generate graphs
python main.py generate_graph
```

---


# Running a Subset of the Experiments

It is possible to run experiments on a reduced configuration instead of executing the full pipeline.

To do this, modify the configuration variables inside:

```bash
utils/constant.py
```
You can select:

- A single task
- A single dataset
- A single LLM
- A specific prompting strategy

Example configuration:

```bash
TASK = ['CLASSIFICATION']

CLASSIFICATION_DATASET = [
    '299_libras_move'
]

LLMs = {
    'gpt-4.1-mini'
}
```
This setup will run the experiment only for:

- the CLASSIFICATION task
- the dataset 299_libras_move
- the model gpt-4.1-mini




---

## Using Ollama

Add this subsection under **Installation** or after **Environment Variables**.


# Using Ollama for Local LLM Inference

This project also supports local LLM inference using Ollama.

#### Install Ollama

Follow the official installation guide:

- Linux/macOS: https://ollama.com/download
- Windows: https://ollama.com/download/windows

After installation, start the Ollama server:

```bash
ollama serve
```
Then download the desired model, for example:
```bash
ollama pull llama4
ollama pull deepseek-r1:14b
```
Update the .env file with the Ollama API endpoint:

```bash
URL_API="http://localhost:11434/api/generate"
```

Important Note About Performance

Running large models locally can be slow if the machine does not have sufficient computational resources (RAM/GPU).

Inference speed highly depends on:

available GPU memory
CPU performance
model size

Large models such as deepseek-r1:14b may require significant resources for acceptable performance.





# License

MIT License





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