## Explainability Testing with LLMs on AutoML

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

