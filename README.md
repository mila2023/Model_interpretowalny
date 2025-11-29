# Model_interpretowalny

## XGBoost Model:
* zbiór_10.csv - data to analyse
* XGBoost_model.ipynb - XGBoost model report in Jupyter
* XGBoost_model_worse_version.ipynb - worse XGBoost model report in Jupter, used to compare Precision/Recall when using different methods
* XGBoost_model_ready.pkl - finished model exported with joblib

-> tu run: (usage: $ .\run_XGBoost.ps1)
* run_XGBoost.ps1 - file that runs the XGBoost model
* run_XGBoost_analysis.py - raw code from "XGBoost_model.ipynb" without Jupyter markdown
* requirements.txt - file specifying the project's dependencies for both XGBoost and LR

## Logistic Regression Model:
* zbiór_10.csv - data to analyse
* analiza_regresji.ipynb - logistic regression model report in Jupyter
* best_regression_model.plk - logistic regression model chosen for further analysis
* wyniki_analizy.joblib - dictionary containing selected test cases, their indices, and computed SHAP values

-> to run: (usage: $ .\run_Regression.ps1)
* run_Regression.ps1 - file that runs the Logistic Regression model
* run_Regression_analysis.py - raw code from "analiza_regresji.ipynb" without Jupyter markdown
* requirements.txt - file specifying the project's dependencies for both XGBoost and LR

-> bibliography.txt - materials used during project
