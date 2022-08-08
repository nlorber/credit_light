import uvicorn
from fastapi import FastAPI
from Model import CreditModel, SK_ID
import pandas as pd

app = FastAPI()
model = CreditModel()

@app.get('/')
def test():
    return {'message': 'Hello, stranger'}

@app.post('/predict_score')
def calc_score(id: SK_ID):
    data = id.dict()
    score, good_idx = model.predict_score(data['id_number'])
    print(score, good_idx)
    return {
        'score': score,
        'index': good_idx
    }

@app.post('/explain_score')
def calc_score(id: SK_ID):
    data = id.dict()
    sp_value, sp_base_value, sp_data, sp_feat_names = model.explanation(data['id_number'])
    return {
        'value': sp_value.tolist(),
        'base_value': sp_base_value.tolist(),
        'data': pd.Series(sp_data.reshape(-1)).fillna('missing_value').tolist(),
        'feat_names': sp_feat_names
    }