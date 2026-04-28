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

### Flat Prompting

- 
Parameters: fit, summarize, judge

To fit the notebook or python file : python main.py fit
To summarize  : python main.py summarize 
To judge: python main.py judge


### Hierarchical Prompting 

