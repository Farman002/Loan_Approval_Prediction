import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained Naive Bayes model
with open('naive_bayes_model.pkl', 'rb') as file:
    classifier = pickle.load(file)

st.title('Loan Approval Prediction App')

# User input
Gender = st.selectbox('Gender', ['Male', 'Female'])
Married = st.selectbox('Married', ['Yes', 'No'])
Dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
Education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
ApplicantIncome = st.number_input('Applicant Income', min_value=0)
CoapplicantIncome = st.number_input('Coapplicant Income', min_value=0)
LoanAmount = st.number_input('Loan Amount', min_value=0)
Credit_History = st.selectbox('Credit History', [0, 1])

# Preprocess input with only 8 features
input_data = np.array([Gender == 'Male', Married == 'Yes', int(Dependents[0]), Education == 'Graduate',
                      ApplicantIncome, CoapplicantIncome, LoanAmount, Credit_History]).reshape(1, -1)

# Predict button
if st.button('Predict'):
    prediction = classifier.predict(input_data)
    if prediction[0] == 1:
        st.success('🎉 Congratulations! Based on the information provided, your loan application is likely to be approved. Our analysis shows that you meet the required criteria. Please proceed with the next steps to finalize your application.')
    else:
        st.error('Unfortunately, your loan is not approved based on the provided information. Consider reviewing your inputs or contacting support for more details.')
