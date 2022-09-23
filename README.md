# Credit approval model, api and dashboard

Credit approval model based on the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/overview) Kaggle competition.
The model is then deployed through an API to be used by a simple customer-oriented dashboard.

*Note: Comments and docstrings are in French.*

**Contents**:
- <u>1. modelisation.ipynb</u>: Notebook detailing the preprocessing and modelling steps.
- <u>2. dashboard.py</u>: Python script used for the dashboard (using Streamlit)
- <u>3. api.py</u>: Python script used for the API (using Flask)
- feat_function.py: preprocessing kernel, based on [Aguiar's Kaggle kernel](https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features)
- functions.py: custom functions used
- loan_model.py: custom class used for final model
- data: processed test data, used for dashboard
- models: exported final models
- plots: data used for plotting
- .streamlit: theming file for streamlit dashboard
