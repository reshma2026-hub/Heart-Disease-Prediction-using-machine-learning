# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 12:31:31 2025

@author: HP
"""

import numpy as np
import pickle
import streamlit as st


#loading the save model
loaded_model=pickle.load(open("D:/Heart Disease Prediction/trained_model.sav",'rb'))


#creating a function for prediction
def heartdisease_prediction(input_data):
    
    # change the input data to a numpy array
    input_data_as_numpy_array = np.array(input_data)

    # reshape the numpy array as we are predicting for only one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]==0):
      return 'The person does not have a heart disease'
    else:
      return 'The person has a heart disease'
  
    
 def main():
     # giving title
     st.title('Heart Disease Prediction Web App')
     
     # getting input data from user
     age	sex	cp	trestbps	chol	fbs	restecg	thalach	exang	oldpeak	slope	ca	thal	
     age=st.text_input('Age of the Person')
     sex=st.text_input('Sex of the Person')
     cp=st.text_input('Constrictive Pericarditis')
     trestbps=st.text_input('Resting Blood Pressure value')
     chol=st.text_input('Cholesterol value')
     fbs=st.text_input('Fasting Blood Pressure value')
     restecg=st.text_input('Resting Electrocardiographic Result')
     thalach=st.text_input('Cardiac Stress')
     exang=st.text_input('Exercise-induced Angina')
     oldpeak=st.text_input('ST depression')
     slope=st.text_input('ST Segment Slope ')
     ca=st.text_input('The Number of Blood Vessel blocked')
     thal=st.text_input('Thalassemia')
        
     
      # code for prediction
      diagnosis= ''
      
      # creating a button for prediction
      if st.button('Heart Disease Test Result'):
          diagnosis=heartdisease_prediction([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])
          
      st.success(diagnosis)
      
      
      
if __name__ == '__main__':
    main()
     
     

     
