# credit_light

This is the light-dataset version of the LGBM-based scoring app, comprising 1000 individuals of the 300+ thousands of the original dataset. Unfortunately, due to file-size issues, the fullit is not deployable onto Heroku.

This version is deployed at : https://credit-web.herokuapp.com/ for the interactive dashboard and https://credit-api-oc.herokuapp.com/docs for the API.

The present code can also be ran locally, if one first uncomments the lines
#URI_1 = 'http://127.0.0.1:8000/predict_score'
#URI_2 = 'http://127.0.0.1:8000/explain_score'
in the dashboard.py file, and instead comments the lines
URI_1 = 'https://credit-api-oc.herokuapp.com/predict_score'
URI_2 = 'https://credit-api-oc.herokuapp.com/explain_score'

After this change is done, type:
    uvicorn app:app
in the terminal. Once the API is successfully deployed, run the command:
    streamlit run dashboard.py
for the dashboard to start. 

The full-dataset, local-only version of the app can be found at : https://github.com/nlorber/credit_full_app