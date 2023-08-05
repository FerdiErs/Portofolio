import streamlit as st 
import pandas as pd 
import numpy as np 
import pickle 
import json 
import joblib as jb

#load models
model = jb.load('model.pkl')

#load data 
df = pd.read_csv('https://raw.githubusercontent.com/FerdiErs/SQL/main/DataScienceSalaries.csv')

def run():

    st.markdown("<h1 style='text-align: center;'>Salary Estimator</h1>", unsafe_allow_html=True)
    # description

    st.subheader('Please check your salary here.')


    with st.form('key=form_prediction') :
        year = st.selectbox('Work Year', df['work_year'].unique())
        experience = st.selectbox('Experience', df['experience_level'].unique())
        employment = st.selectbox('Employee Type', df['employment_type'].unique())
        job = st.selectbox('Job Title', sorted(df['job_title'].unique()))
        residence = st.selectbox('Country Origin', sorted(df['employee_residence'].unique()))
        remote =  st.selectbox('Remote', df['remote_ratio'].unique())
        location = st.selectbox('Company location',  sorted(df['company_location'].unique()))
        size =  st.selectbox('Company Size', df['company_size'].unique())


        submitted = st.form_submit_button('Predict')

    inf = {
    'work_year': year,
    'experience_level': experience,
    'employment_type': employment,
    'job_title' : job,
    'employee_residence':residence,
    'remote_ratio': remote,
    'company_location': location,
    'company_size': size
    }

    data_inf = pd.DataFrame([inf])
    st.dataframe(data_inf)

    if submitted:

        # Predict using bagging 
        y_pred_inf = model.predict(data_inf)

        st.write('with this experience you should get salary around')
        st.write('# $', str(int(y_pred_inf)))
        st.write('NOTE : Please remember this model is not 100% correct please check again with another website about paycheck like glassdoor')


if __name__=='__main__':
    run()

