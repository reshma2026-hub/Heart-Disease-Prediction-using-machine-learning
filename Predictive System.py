# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 12:19:39 2025

@author: HP
"""

import numpy as np
import pickle 


#loading the save model
loaded_model=pickle.load(open("D:/Heart Disease Prediction/trained_model.sav",'rb'))


input_data = (63,1,3,145,233,1,0,150,0,2.3,0,0,1)

# change the input data to a numpy array
input_data_as_numpy_array = np.array(input_data)

# reshape the numpy array as we are predicting for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
  print('The person does not have a heart disease')
else:
  print('The person has a heart disease')