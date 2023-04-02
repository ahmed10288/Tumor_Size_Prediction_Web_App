import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Load the saved StandardScaler object
scaler = pickle.load(open('C:/Users/ahmed/OneDrive/Master/Máster - IMF/TMF/python/scaler.sav', 'rb'))

# Load the saved XGBRegressor model
model = pickle.load(open('C:/Users/ahmed/OneDrive/Master/Máster - IMF/TMF/python/trained_model.sav', 'rb'))

def predict(area_worst, concavepoints_mean, concavity_worst, area_se, perimeter_se):
    # Transform the input values using the StandardScaler
    input_data = np.array([[area_worst, concavepoints_mean, concavity_worst, area_se, perimeter_se]])
    scaled_input = scaler.transform(input_data)
    
    # Make the prediction using the XGBRegressor model
    prediction = model.predict(scaled_input)
    
    # Return the prediction
    return prediction[0]
    
# Define the Streamlit app

st.title('Tumor Size Prediction App')
st.write('Enter the following values to predict the Tumor Size:')
   
# getting the input data from the user
col1, col2, col3 = st.columns(3)
    
with col1:
    area_worst = st.text_input('Enter the Worst Area')
        
with col2:
    area_se = st.text_input('Enter the Area SE')
    
with col3:
    concavity_worst = st.text_input('Enter the Worst Concavity')
    
with col1:
    concavepoints_mean = st.text_input('Enter the Concave Points Mean')
    
with col2:
    perimeter_se = st.text_input('Enter Perimeter SE')
    
# code for Prediction
tumor_size = ''
 
 
# Make predictions and display the result
if st.button('Predict'):
    tumor_size = predict(area_worst, concavepoints_mean, concavity_worst, area_se, perimeter_se)
    #st.write(f'The predicted Tumor Size is {tumor_size}')
    st.success(f'The predicted Tumor Size is {tumor_size}')