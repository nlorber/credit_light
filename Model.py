import pandas as pd 
from pydantic import BaseModel
import joblib
import shap


class SK_ID(BaseModel):
    id_number: int

class CreditModel:
    def __init__(self):
        print('\nLoading database...\n')
        self.df = pd.read_csv('credit_light.csv')
        print('Database loaded\n')
        self.feats = [f for f in self.df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
        print('Loading ML model...\n')
        self.model = joblib.load('clf.pkl')
        self.explainer = shap.TreeExplainer(self.model)
        print('Model loaded\n')
        print('Loading analytical data...\n')
        self.sp_values = joblib.load('shap_light_values.pkl')
        self.sp_base_values = joblib.load('shap_light_base_values.pkl')
        self.sp_feat_names = joblib.load('shap_light_feat_names.pkl')
        print('Analytical data loaded\n')
    
    def predict_score(self, id_number):
        if id_number in set(self.df['SK_ID_CURR']):
            data_input = self.df[self.df['SK_ID_CURR'] == id_number][self.feats]
            score = self.model.predict_proba(data_input)[0][1]
            good_idx = id_number
        else:
            score = -1
            good_idx = min(self.df['SK_ID_CURR'].to_list(), key=lambda x: abs(x-id_number))
        return score, good_idx

    def explanation(self, id_number):
        rank = self.df[self.df['SK_ID_CURR'] == id_number].index[0]
        return self.sp_values[rank], self.sp_base_values[rank], self.df.loc[rank, self.feats].values, self.sp_feat_names