import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import streamlit as st
from streamlit_option_menu import option_menu

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
selected_model= model
# Sidebar for navigation
with st.sidebar:
    
    selected = option_menu('MONALI®', ['Tumor Size Prediction'],default_index=0)
    
if (selected == 'Tumor Size Prediction'):
    
    # page title
    st.title('MONALI®: AI-Based Tumor Size Prediction')
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
            st.success(f'The predicted Tumor Size is {tumor_size}')

def main():
    # Add the refresh button
    refresh_button = st.button('', key='refresh')
    refresh_icon = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-arrow-clockwise" viewBox="0 0 16 16">' \
                   '<path fill-rule="evenodd" d="M8 2a6 6 0 1 0 6 6H9.5a.5.5 0 0 1-.5-.5V6a.5.5 0 0 0-1 0v3.5a.5.5 0 0 1-.5.5H2a.5.5 0 0 1-.5-.5V2a.5.5 0 0 1 .5-.5h6zm4.354 1.646a.5.5 0 0 0-.708-.708L10.5 4.793V2.5a.5.5 0 0 0-1 0v3.793a.5.5 0 0 0 .146.354l2.147 2.146a.5.5 0 0 0 .708-.708L11.207 6.5H14.5a.5.5 0 0 0 0-1h-3.793l1.647-1.646z"/>' \
                   '</svg>'
    refresh_button.markdown(refresh_icon, unsafe_allow_html=True)

    # Add your other Streamlit elements here

if __name__ == '__main__':
    main()
