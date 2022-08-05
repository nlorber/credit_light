import pandas as pd
import streamlit as st
import requests
import shap
import numpy as np
import matplotlib.pyplot as pl

st.set_option('deprecation.showPyplotGlobalUse', False)

def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}
    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data)
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
    return response.json()


def main():
    URI_1 = 'http://127.0.0.1:8000/predict_score'
    URI_2 = 'http://127.0.0.1:8000/explain_score'

    st.title('Credit Scoring')

    id_number = st.number_input('Entrez le numéro d\'identification', min_value=100001, max_value=456255, value=123456, step=1,
                                help='Tel qu\'indiqué dans la catégorie SK_ID_CURR')

    threshold_type = st.selectbox('Sélectionnez le seuil', ('Strict', 'Moyen', 'Tolérant'), index=1,
                                help='Score à atteindre pour l\'octroiement du crédit')

    result_details = st.checkbox('Détail du résultat')

    graph_options = st.multiselect('Sélection de graphiques',
                                    ['Force plot', 'Bar plot', 'Waterfall', 'Decision plot'],
                                    ['Force plot'], disabled=(1-result_details),
                                    help=('Graphiques à afficher. Cochez \'Détail du résultat\' pour pouvoir choisir.'))

    predict_btn = st.button('Prédire')

    if predict_btn:

        data = {
            'id_number': id_number
        }
        
        pred = None
        pred = request_prediction(URI_1, data)['score']

        if threshold_type == 'Strict': threshold = 0.547
        elif threshold_type == 'Tolérant': threshold = 0.163
        else: threshold = 0.266


        if pred == -1:
            st.error('Désolé, ce profil n\'existe pas')
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(' ')
            with col2:
                st.metric('Score', np.round(pred, decimals=2), delta = np.round(pred-threshold, decimals=2),
                            help='En bas : écart relatif au seuil à atteindre')
            with col3:
                st.write(' ')

            if pred >= threshold:
                st.success('Crédit accordé')
                st.balloons()
            else:
                st.warning('Crédit refusé')

        
            if result_details == 1:
                with st.spinner('Chargement des détails...'):
                    shap_pred = request_prediction(URI_2, data)
                    sp_feat_names = shap_pred['feat_names']
                    sp_value = np.array(shap_pred['value'])
                    sp_base_value = shap_pred['base_value']
                    sp_data = np.array(pd.Series(shap_pred['data']).replace('missing_value', np.nan))
                    shap_exp = shap._explanation.Explanation(sp_value, sp_base_value, sp_data, feature_names=sp_feat_names)

                    shap.initjs()

                    if 'Force plot' in graph_options:
                        shap.force_plot(shap_exp, matplotlib=True)
                        st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0.1)
                        pl.clf()

                    if 'Bar plot' in graph_options:
                        shap.plots.bar(shap_exp)
                        st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0.1)
                        pl.clf()

                    if 'Waterfall' in graph_options:
                        shap.plots.waterfall(shap_exp)
                        st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0.1)
                        pl.clf()
                    
                    if 'Decision plot' in graph_options:
                        shap.decision_plot(shap_exp.base_values, shap_exp.values, shap_exp.feature_names)
                        st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0.1)
                        pl.clf()



if __name__ == '__main__':
    main()
