import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Load the saved StandardScaler object
scaler = pickle.load(open('scaler.sav', 'rb'))

# Load the saved XGBRegressor model
model = pickle.load(open('trained_model.sav', 'rb'))

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
col1, col2, col3, col4, col5 = st.columns(5)
    
with col1:
    area_worst = st.text_input('Area Worst')
        
with col2:
    concavepoints_mean = st.text_input('Concave Points Mean')
    
with col3:
    concavity_worst = st.text_input('Concavity Worst')
    
with col4:
    area_se = st.text_input('Area SE')
    
with col5:
    perimeter_se = st.text_input('Perimeter SE')
    
# code for Prediction
tumor_size = ''
 
 
# Make predictions and display the result
if st.button('Predict'):
    tumor_size = predict(area_worst, concavepoints_mean, concavity_worst, area_se, perimeter_se)
    #st.write(f'The predicted Tumor Size is {tumor_size}')
    st.success(f'The predicted Tumor Size is {tumor_size}')
