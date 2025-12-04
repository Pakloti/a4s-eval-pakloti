# **A4S – Data Drift Metric & Model Robustness Evaluation**



This project implements a **data drift metric** and is used to evaluate how different machine learning models react when the input data distribution changes.

It is part of the A4S evaluation framework at the University of Luxembourg.



------



## **Features**



- **Data Drift Metric** implemented using *Evidently*
- **Synthetic drift generation** over 50 iterations
- **Model evaluation under drift** (Logistic Regression, Random Forest, SVM)
- **Final notebook** with visual analysis and model comparison



------



## **Installation**



This project uses uv for environment and dependency management.

``` 
uv venv
source .venv/bin/activate
uv sync
```



------



## **1. Generate Drift Metric Scores**



This script creates 50 drifted versions of the test dataset and computes the drift score for each one.

` uv run python a4s_eval/analysis/generate_drift_metric_dataset.py `

Output file: → tests/data/measures/data_drift.csv

This CSV contains one drift score per synthetic scenario.



------



## **2. Train & Evaluate ML Models**



Three models are trained on the reference dataset and evaluated across all drifted datasets:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)



Run:

` uv run python a4s_eval/analysis/run_models.py `

Outputs:

tests/data/measures/results_logreg.csv
tests/data/measures/results_randomforest.csv
tests/data/measures/results_svm.csv

Each file contains the model accuracy for each drift scenario.



------



## **Notebook Analysis**



The notebook Data_Drift_Metric_Notebook.ipynb provides:



- visualization of drift over time
- comparison of model accuracies
- correlation between drift and model performance
- final robustness comparison.



------



## **Acknowledgments**



This work was completed as part of the **Advanced Software Systems (A4S)** course at the **University of Luxembourg**.



------

