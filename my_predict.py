import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

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