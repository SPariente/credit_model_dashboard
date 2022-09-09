import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from feat_functions import *
from functions import *
import shap
import requests
import streamlit as st
from ast import literal_eval
import json
import plotly.graph_objects as go

st.title('Dashboard crédit client')

# Importation des bases de données, et mise en cache
@st.cache
def import_data():
    data_set = pd.read_csv('data/reduced_data.csv')
    data_set.set_index('SK_ID_CURR', inplace=True)
    data_set.index = data_set.index.astype('str')
    return data_set

# Texte d'attente duraznt chargement
data_load_state = st.text('Chargement en cours...')

data_set = import_data()

with open('plots/hist_data.json', 'r') as file:
    hist_data = json.load(file)

with open('plots/data_dict.json', 'r') as file:
    data_dict = json.load(file)

# Définition des couleurs utilisées, sur base des couleurs SHAP
colors = shap.plots.colors
red_shap = '#%02x%02x%02x' % tuple((colors.red_rgb*255).astype(int))
blue_shap = '#%02x%02x%02x' % tuple((colors.blue_rgb*255).astype(int))

decision_colors = [blue_shap, red_shap]
decision_text = ['Accepté', 'Refusé']

with open('models/top_features.txt', 'r') as file:
    features = literal_eval(file.read())

# Notification du chargement terminé
data_load_state.text("Chargement terminé.")

# Zone de saisie de l'identifiant client
st.subheader('Identifiant client')
cust_id = st.text_input("Saisir l'identifiant client",
                        placeholder='Identifiant client',
                        value='',
                        max_chars=6)

# Appel de l'API, et mise en cache
@st.cache(suppress_st_warning=True)
def api_call(cust_id):

    if cust_id not in data_set.index:
        id_found = False
        st.text('Identifiant client non trouvé, merci de vérifier la saisie.')

    else:
        id_found = True
        cust_X = data_set.loc[[cust_id]]   

    if id_found:
        data = cust_X[features].to_json()
        r_model = requests.post("http://oc-model-api.herokuapp.com/upload/model", json=data, verify=False)
        r_shap = requests.post("http://oc-model-api.herokuapp.com/upload/shap", json=data, verify=False)

        decision = r_model.json()['model_output']
        thresh = r_model.json()['model_threshold']

        shap_values = shap.Explanation(
                values=np.array(r_shap.json()['values']),
                base_values=np.array(r_shap.json()['base_values']),
                data=np.array(r_shap.json()['data']),
                feature_names=r_shap.json()['feature_names']
            )
        
        def_vars = pd.read_csv('data/HomeCredit_columns_description.csv', encoding='cp1252')
        def_vars = def_vars[['Row', 'Description']]
        def_vars.drop_duplicates(subset='Row', inplace=True)
        def_vars.set_index('Row', inplace=True)
        
    else:
        cust_X = None
        decision = None
        thresh = None
        shap_values = None
        def_vars = None
        
    return id_found, cust_X, decision, thresh, shap_values, def_vars

id_found, cust_X, decision, thresh, shap_values, def_vars = api_call(cust_id)

if id_found:
    st.subheader('Décision de crédit')
    
    fig = go.Figure(go.Indicator(
        mode = "gauge",
        value = shap_values[0][:,1].values.sum() + shap_values[0][:,1].base_values,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {'axis': {'range': [0, 1],
                          'tickmode': 'array',
                          'tickvals': [0, thresh, 1],
                          'ticktext': [0, thresh, 1]
                         },
                 'steps' : [
                     {'range': [0, thresh], 'color': blue_shap},
                     {'range': [thresh, 1], 'color': red_shap}],
                 'bar': {'color': 'ivory'},
                },
        title = {'text': "Score attribué au client"}
    ))
    
    fig.add_annotation(x=0.5,
                       y=0,
                       text=f'Statut :<br><span style="color:{decision_colors[decision]}">{decision_text[decision]}</span>',
                       font={'size': 40},
                       showarrow=False)

    fig.update_layout(font = {'color': "white", 'family': "Arial"})
    st.plotly_chart(fig)
    
    shap_col1, hist_col1 = st.columns(2)
    
    with shap_col1:
        st.subheader('Visualisation des principaux drivers de la décision')
    with hist_col1:
        st.subheader('Visualisation de la position relative du client sur ces indicateurs')

    n_feats = st.slider('Nombre de caractéristiques client à afficher',
                        min_value=1,
                        max_value=len(shap_values.feature_names),
                        value=5,
                        step=1)
    
    shap_col2, hist_col2 = st.columns(2)
    with shap_col2:
        shap_wat = shap.waterfall_plot(shap_values[0][:,1],
                                       max_display=n_feats+1,
                                       show=False)
        shap_wat.set_size_inches(5,2*n_feats-1)
        st.pyplot(shap_wat, bbox_inches='tight')
    with hist_col2:
        
        key_features = np.argsort(-np.abs(shap_values.values[0][:,1]))[:n_feats]
        key_features = np.array(shap_values.feature_names)[key_features]
        
        top_feat_fig = plot_top_feat_hist(key_features, hist_data, cust_X, decision_colors[decision])
        top_feat_fig.set_size_inches(5,2*n_feats-1)
        st.pyplot(top_feat_fig, bbox_inches='tight')

if id_found:
    st.subheader("Visualisation personnalisée")
    sel_feat = st.selectbox(label="Saisir une information client à analyser", 
                            options=features)
    feat_col1, feat_col2 = st.columns(2)
    with feat_col1:
        st.markdown(f"Valeur : **{cust_X[sel_feat].iloc[0]}**",
                    unsafe_allow_html=True)
        mapping = [feat == sel_feat for feat in shap_values.feature_names]
        shap_impact = shap_values[0][:,1].values[mapping][0]
        st.markdown(f'Impact sur décision de crédit : **<span style="color:{decision_colors[int(shap_impact>0)]}">{shap_impact:+.3f}</span>**',
                    unsafe_allow_html=True)
        
        st.markdown(f"Définition : *{data_dict[sel_feat]}*",
                unsafe_allow_html=True)
        
    with feat_col2:
        feat_fig = plot_top_feat_hist([sel_feat], hist_data, cust_X, decision_colors[decision])
        feat_fig.set_size_inches(5,2)
        st.pyplot(feat_fig, bbox_inches='tight')