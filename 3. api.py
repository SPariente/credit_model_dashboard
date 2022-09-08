import flask
from flask import request, jsonify
import numpy as np
import pandas as pd
import lightgbm as lgb
from imblearn import under_sampling, over_sampling, pipeline
import shap
import joblib

# Chargement des paramètres du modèle
model = joblib.load('models/final_model.joblib')
with open('models/final_thresh.txt') as file:
    pos_thresh = float(file.read())

# Création du modèle
class loan_model():
    def __init__(self, model, pos_thresh):
        self.model = model
        self.pos_thresh = pos_thresh
    def fit(self, X, y):
        self.model.fit(X, y)
        return self.model
    def predict_proba(self, X):
        self.prob_pred_raw = self.model.predict_proba(X)
        self.prob_pred = self.prob_pred_raw - np.array(
            [.5-self.pos_thresh,
             self.pos_thresh-.5])
        return self.prob_pred
    def predict(self, X):
        self.y_pred = np.apply_along_axis(
            np.argmax,
            axis=1,
            arr=self.predict_proba(X))
        return self.y_pred
    
import __main__
__main__.loan_model = loan_model

model = loan_model(model, pos_thresh)

# Chargement du modèle SHAP
explainer = joblib.load('models/explainer.joblib')

# App
app = flask.Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return """<h1>API du modèle de décision de crédit</h1>"""

@app.route('/upload/model', methods=['POST'])
def model_request():
    
    # Lecture des données client
    content = request.get_json()
    X = pd.read_json(content)
    
    # Prédiction du modèle
    response = model.predict(X)[0]
    
    # Réponse : prédiction et threshold utilisé par modèle
    return jsonify({'model_output': int(response),
                    'model_threshold':model.pos_thresh})


@app.route('/upload/shap', methods=['POST'])
def shap_request():
    
    # Lecture des données client
    content = request.get_json()
    X = pd.read_json(content)
    
    # Détail par features avec SHAP
    shap_values = explainer(X,
                            max_evals=max(2*model.model.n_features_in_+1, 500))
    
    # Réponse : composants du shap.Explanation
    shap_values_dict = {
        'values': shap_values.values.tolist(),
        'base_values': shap_values.base_values.tolist(),
        'data': shap_values.data.tolist(),
        'feature_names': shap_values.feature_names
    }
    
    return jsonify(shap_values_dict)

@app.after_request
def add_header(response):
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store'
    return response

if __name__ == '__main__':
    app.run()