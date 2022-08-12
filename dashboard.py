import re
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

def spread(value):
    new_value = int(np.round(10*((11**value) - 1), decimals=0))
    return new_value

def main():
    # Commentez ou decommentez une paire d'URIs parmi les deux ci-dessous,
    # selon que vous vouliez faire tourner l'application localement ou en ligne
    #URI_1 = 'http://127.0.0.1:8000/predict_score'
    #URI_2 = 'http://127.0.0.1:8000/explain_score'
    URI_1 = 'https://credit-api-oc.herokuapp.com/predict_score'
    URI_2 = 'https://credit-api-oc.herokuapp.com/explain_score'

    st.title('Credit Scoring')

    id_number = st.number_input('Entrez le numéro d\'identification', min_value=100001, max_value=456255, value=123456, step=1,
                                help='Tel qu\'indiqué dans la catégorie SK_ID_CURR')

    threshold_type = st.selectbox('Sélectionnez le seuil', ('Strict', 'Moyen', 'Tolérant'), index=1,
                                help='Score à atteindre pour l\'octroiement du crédit')

    st.text('Un seuil plus laxiste facilite l\'obtention du crédit, mais diminue la précision.\nÀ titre indicatif :\n\
        Seuil : \'Strict\' \t Précision : 99.1%\n\
        Seuil : \'Moyen\' \t Précision : 98%\n\
        Seuil : \'Tolérant\' \t Précision : 93.9%\n\
        ')

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
        
        request_pred = request_prediction(URI_1, data)
        pred = request_pred['score']
        idx = request_pred['index']
        user_details = request_pred['details']


        if threshold_type == 'Strict': threshold = 0.9858675212142891
        elif threshold_type == 'Tolérant': threshold = 0.7664397179939818
        else: threshold = 0.9605169323777106

        pred_spread = spread(pred)
        threshold_spread = spread(threshold)


        if pred == -1:
            st.error('Désolé, ce profil n\'est pas répertorié.\
                \nVouliez-vous dire {} ?'.format(idx))

        else:

            if user_details[0] == 0: genre = 'un homme'
            elif user_details[0] == 1: genre = 'une femme'
            else: genre = 'Value error'
            age = int((-user_details[1])//365.25)
            if user_details[2] == 0: job = 'sans emploi'
            else:
                job = 'occupant un emploi depuis {:.2f} ans'.format((-user_details[2])/365.25)
            if user_details[3] == 1:
                car = 'Possède une voiture '
                if user_details [4] == 0: immo = 'et une propriété immmobilière'
                elif user_details [4] == 1: immo = 'mais pas d\'immmobilier'
                else: immo = 'Value error'
            elif user_details[3] == 0:
                car = 'Ne possède pas de voiture '
                if user_details [4] == 0: immo = 'mais possède une propriété immmobilière'
                elif user_details [4] == 1: immo = 'ni d\'immmobilier'
                else: immo = 'Value error'
            else:
                car = 'Value error.'
                if user_details [4] == 0:
                    immo = 'Possède une propriété immmobilière'
                elif user_details [4] == 1: immo = 'Ne possède pas d\'immmobilier'
                else: immo = 'Value error'

            st.write('Profil utilisateur : le client numéro {} est {} de {} ans, {}. {} {}.'.format(id_number, genre, age, job, car, immo))

            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(' ')
            with col2:
                st.metric('Score', '{}/100'.format(pred_spread), delta = pred_spread-threshold_spread,
                            help='En bas : écart relatif au seuil à atteindre')
            with col3:
                st.write(' ')

            if pred >= threshold:
                st.success('Crédit accordé')
                st.balloons()
            else:
                st.warning('Crédit refusé')

        
            if (result_details == 1) & (pred != -1):
                with st.spinner('Chargement des détails...'):
                    shap_pred = request_prediction(URI_2, data)
                    sp_feat_names = shap_pred['feat_names']
                    sp_value = np.array(shap_pred['value'])
                    sp_base_value = shap_pred['base_value']
                    sp_data = np.array(pd.Series(shap_pred['data']).replace('missing_value', np.nan))
                    shap_exp = shap._explanation.Explanation(sp_value, sp_base_value, sp_data, feature_names=sp_feat_names)

                    shap.initjs()

                    if 'Force plot' in graph_options:
                        st.caption('Force plot : les scores f(x) (en gras) et base value correspondent respectivement au score de\
                            l\'utilisateur et à la moyenne de la population. Ceux-ci sont indiqués en échelle logarithmique. Les segments\
                            rouges indiquent les paramètres ayant contribué à augmenter le score avec, pour les plus importants d\'entre\
                            eux, les valeurs renseignées dans ces catégories. A l\'inverse, les segments bleus correspondent aux\
                            catégories ayant contribué à faire baisser le score individuel.')
                        shap.force_plot(shap_exp, matplotlib=True)
                        st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0.1)
                        pl.clf()


                    if 'Bar plot' in graph_options:
                        st.caption('Bar plot : sont représentés ici les contributions relatives et alignées des facteurs primordiaux\
                        pour le calcul du score du client présent. En bleu apparaissent les facteurs ayant un impact négatif sur le score,\
                        en rouge ceux ayant un impact positif. La longueur des segments est proportionnelle à leurs contributions\
                        respectives, et les valeurs renseignées pour chacun de ces paramètres sont indiqués dans la colonne de gauche.')
                        shap.plots.bar(shap_exp)
                        st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0.1)
                        pl.clf()

                    if 'Waterfall' in graph_options:
                        st.caption('Waterfall plot : les scores f(x) (en haut) et E[f(x)] (en bas) correspondent respectivement au score de\
                            l\'utilisateur et à la moyenne de la population. Ceux-ci sont indiqués en échelle logarithmique. Les segments\
                            rouges indiquent les paramètres ayant contribué à augmenter le score. A l\'inverse, les segments bleus\
                            correspondent aux catégories ayant contribué à faire baisser le score individuel. Dans les deux cas,\
                            la longueur du segment est proportionnelle à la contribution de ces facteurs, et la valueur renseignée\
                            dans ces catégories est indiquée à gauche du graphique.')
                        shap.plots.waterfall(shap_exp)
                        st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0.1)
                        pl.clf()
                    
                    if 'Decision plot' in graph_options:
                        st.caption('Decision plot : la barre grise correspond à la moyenne de la population. Sont représentés ici les\
                            paramètres ayant le plus affecté\
                            le score individuel, et comment celui-ci a été affecté : bifurcation vers la gauche lorsque le paramètre a fait\
                            baisser le score, vers la droite lorsqu\'il l\'a augmenté.')
                        shap.decision_plot(shap_exp.base_values, shap_exp.values, shap_exp.feature_names)
                        st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0.1)
                        pl.clf()



if __name__ == '__main__':
    main()
