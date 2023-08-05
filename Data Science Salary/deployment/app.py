import streamlit as st 
import eda
import prediction 

navigation = st.sidebar.selectbox('Choose Page : ', ('Description','Salary Estimator'))

if navigation == 'Description':
    eda.run()
else:
    prediction.run()