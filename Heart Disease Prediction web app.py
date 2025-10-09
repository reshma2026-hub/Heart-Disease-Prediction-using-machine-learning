# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 12:31:31 2025

@author: HP
"""
import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open("trained_model.sav", 'rb'))

# creating a function for prediction
def heartdisease_prediction(input_data):
    # change the input data to a numpy array
    input_data_as_numpy_array = np.array(input_data, dtype=float)

    # reshape the numpy array as we are predicting for only one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The person does not have a heart disease'
    else:
        return 'The person has a heart disease'


def main():
    # title
    st.title('Heart Disease Prediction Web App')

    # getting input data from user
    age = st.text_input('Age of the Person')
    sex = st.text_input('Sex of the Person (1 = Male, 0 = Female)')
    cp = st.text_input('Chest Pain Type')
    trestbps = st.text_input('Resting Blood Pressure')
    chol = st.text_input('Cholesterol Value')
    fbs = st.text_input('Fasting Blood Sugar (1 = True, 0 = False)')
    restecg = st.text_input('Resting Electrocardiographic Result')
    thalach = st.text_input('Maximum Heart Rate Achieved')
    exang = st.text_input('Exercise-induced Angina (1 = Yes, 0 = No)')
    oldpeak = st.text_input('ST Depression Induced by Exercise')
    slope = st.text_input('Slope of the ST Segment')
    ca = st.text_input('Number of Major Vessels Blocked (0–3)')
    thal = st.text_input('Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)')

    # code for prediction
    diagnosis = ''

    # button for prediction
    if st.button('Heart Disease Test Result'):
        try:
            diagnosis = heartdisease_prediction([
                age, sex, cp, trestbps, chol, fbs, restecg,
                thalach, exang, oldpeak, slope, ca, thal
            ])
        except ValueError:
            st.error("⚠️ Please enter all numeric values correctly.")
            return

    st.success(diagnosis)


if __name__ == '__main__':
    main()
