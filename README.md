## Explainability Testing with LLMs on AutoML

This repository is designed to test and observe the explainability capability of different Large Language Models (LLMs) in the context of AutoML.
It provides an environment to experiment, evaluate, and compare how various models generate explanations for automated machine learning processes.

## 🚀 Getting Started

Follow the steps below to set up and run the project:

1.  Create a Virtual Environment (Python 3.10 required)

2. Clone the Repository
```
- git clone https://github.com/amelinenar/explanation-experiment-llm.git
= cd your-repo-name
```
3. Install Dependencies
```
pip install -r requirements.txt
```
4. Configure Environment Variables
Fill in the required values inside the .env file:
```
URL_API=' '
ROOT_DIR=''
```

6. (Optional) Resolve Potential SSH Issues

If you encounter SSH-related errors while installing dependencies, run:
```
python -m pip install --upgrade pip setuptools wheel
```

## 🎯 Objective

Evaluate how different LLMs explain AutoML workflows.
Test clarity, accuracy, and interpretability of model-generated explanations.
Provide a reproducible framework for researchers and developers to experiment.
