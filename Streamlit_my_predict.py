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
    
# Get the input data from the user
area_worst = st.number_input('Enter the worst area', format='%f',min_value=0.0, max_value=3000.0, step=0.0000001)
concavepoints_mean = st.number_input('Enter mean number of concave points', format='%f', min_value=0.0, max_value=100.0, step=0.0000001)
concavity_worst = st.number_input('Enter worst concavity', format='%f', min_value=0.0, max_value=100.0,  step=0.0000001)
area_se = st.number_input('Enter the standard error of the area', format='%f', min_value=0.0, max_value=500.0,  step=0.0000001)
perimeter_se = st.number_input('Enter the standard error of the perimeter', format='%f', min_value=0.0, max_value=100.0,  step=0.0000001)


# Make predictions and display the result
if st.button('Predict'):
    tumor_size = predict(area_worst, concavepoints_mean, concavity_worst, area_se, perimeter_se)
    st.write(f'The predicted Tumor Size is {tumor_size}')