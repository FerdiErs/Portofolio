import streamlit as st 
import pandas as pd 
import numpy as np 
import pickle 
import json 
import joblib as jb
from tensorflow.keras.models import load_model
from feature_engine.outliers import Winsorizer

#load models
final_pipeline = jb.load('final_pipeline.pkl')
model_ann = load_model('model.h5')

#load data 
df = pd.read_csv('https://raw.githubusercontent.com/FerdiErs/SQL/main/churn.csv')

def run():

    st.markdown("<h1 style='text-align: center;'>Churn predictor</h1>", unsafe_allow_html=True)
    # description

    st.subheader('Will your customer churn?')


    with st.form('key=form_prediction') :
        Age = st.number_input('AGE',min_value=10,max_value=70,step=1)
        Region = st.selectbox('Region', df['region_category'].unique())
        Member = st.selectbox('Membership Type', df['membership_category'].unique())
        offer = st.selectbox('Preferred Offer',df['preferred_offer_types'].unique())
        Internet = st.selectbox('Your Connectivity',df['internet_option'].unique())
        last_login = st.slider('last login',min_value=0,max_value=365)
        time_spent = st.slider('TimeSpent',min_value=0,max_value=10000)
        transaction_value = st.number_input('Money spent',min_value=10,max_value=99999999,step=1)
        login_days = st.number_input('login streak',min_value=0,max_value=99999999)
        points_in_wallet= st.number_input('wallet money',min_value=0,max_value=99999999)
        past_complaint= st.selectbox('complaint', df['past_complaint'].unique())
        feedback = st.selectbox('feedback', df['feedback'].unique())


        submitted = st.form_submit_button('Predict')

    data_inf = {
        'age': Age,
        'region_category': Region,
        'membership_category': Member,
        'preferred_offer_types': offer,
        'internet_option': Internet,
        'days_since_last_login': last_login,
        'avg_time_spent': time_spent,
        'avg_transaction_value': transaction_value,
        'avg_frequency_login_days': login_days,
        'points_in_wallet': points_in_wallet,
        'past_complaint': past_complaint,
        'feedback':feedback
    }

    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    if submitted:
        # transfrom data 
        data_inf_transform = final_pipeline.transform(data_inf)

        #modelling data 
        y_pred_inf = model_ann.predict(data_inf_transform)
        
        if y_pred_inf >= 0.5:
           st.write('## Customer will churn')
        else :
           st.write('## Customer will not churn')

if __name__=='__main__':
    run()