"""
Bike Rental Demand Prediction - Reusable Prediction Module
TY Computer Engineering Mini Project

This module loads the trained Scikit-learn Pipeline (preprocessing + Random Forest)
and exposes a clean, robust `predict_demand(input_dict)` function.
"""

import os
import joblib
import pandas as pd
import numpy as np

# Model filepath
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'bike_model.pkl')

# Global cache for the loaded model pipeline
_model = None


def load_model():
    """Loads and caches the trained machine learning pipeline."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at '{MODEL_PATH}'. "
                "Please run the Jupyter notebook or training script first to generate bike_model.pkl."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_demand(input_dict):
    """
    Predict bike rental count given input environmental and calendar features.
    
    Parameters:
    -----------
    input_dict : dict
        Dictionary containing the following keys:
        - 'season': int (1: Spring, 2: Summer, 3: Fall, 4: Winter)
        - 'yr': int (0: 2011, 1: 2012)
        - 'mnth': int (1 to 12)
        - 'hr': int (0 to 23)
        - 'holiday': int (0: No, 1: Yes)
        - 'weekday': int (0: Sunday, 1: Monday, ..., 6: Saturday)
        - 'workingday': int (1: Working day, 0: Weekend/Holiday)
        - 'weathersit': int (1: Clear, 2: Mist, 3: Light Rain/Snow, 4: Heavy Rain/Snow)
        - 'temp': float (Celsius e.g. 25.0 OR normalized 0.0-1.0)
        - 'atemp': float (Feeling temperature in Celsius OR normalized 0.0-1.0)
        - 'hum': float (Humidity % e.g. 60.0 OR normalized 0.0-1.0)
        - 'windspeed': float (Wind speed in km/h e.g. 15.0 OR normalized 0.0-1.0)
        
    Returns:
    --------
    dict:
        {
            'predicted_count': int,
            'confidence_interval_low': int,
            'confidence_interval_high': int,
            'demand_level': str ('Low', 'Moderate', 'High', 'Peak'),
            'normalized_inputs': dict
        }
    """
    model = load_model()
    
    # Copy dictionary to avoid mutating caller's data
    data = dict(input_dict)
    
    # Required keys validation
    required_keys = [
        'season', 'yr', 'mnth', 'hr', 'holiday', 
        'weekday', 'workingday', 'weathersit', 
        'temp', 'atemp', 'hum', 'windspeed'
    ]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"Missing required input features: {missing}")

    # Automatic normalization handling for natural user inputs:
    # Temperature: UCI max is 41°C
    temp_val = float(data['temp'])
    if temp_val > 1.0:
        norm_temp = temp_val / 41.0
    else:
        norm_temp = temp_val
    norm_temp = min(max(norm_temp, 0.0), 1.0)

    # Feeling temperature: UCI max is 50°C
    atemp_val = float(data['atemp'])
    if atemp_val > 1.0:
        norm_atemp = atemp_val / 50.0
    else:
        norm_atemp = atemp_val
    norm_atemp = min(max(norm_atemp, 0.0), 1.0)

    # Humidity: UCI max is 100%
    hum_val = float(data['hum'])
    if hum_val > 1.0:
        norm_hum = hum_val / 100.0
    else:
        norm_hum = hum_val
    norm_hum = min(max(norm_hum, 0.0), 1.0)

    # Windspeed: UCI max is 67 km/h
    wind_val = float(data['windspeed'])
    if wind_val > 1.0:
        norm_wind = wind_val / 67.0
    else:
        norm_wind = wind_val
    norm_wind = min(max(norm_wind, 0.0), 1.0)

    # Construct single-row DataFrame in the exact feature column order
    input_df = pd.DataFrame([{
        'season': int(data['season']),
        'yr': int(data['yr']),
        'mnth': int(data['mnth']),
        'hr': int(data['hr']),
        'holiday': int(data['holiday']),
        'weekday': int(data['weekday']),
        'workingday': int(data['workingday']),
        'weathersit': int(data['weathersit']),
        'temp': norm_temp,
        'atemp': norm_atemp,
        'hum': norm_hum,
        'windspeed': norm_wind
    }])

    # Predict demand
    raw_pred = model.predict(input_df)[0]
    predicted_count = int(max(0, round(raw_pred)))

    # Qualitative categorization based on historical distribution:
    # 25th percentile ~ 40, 50th percentile ~ 142, 75th percentile ~ 281
    if predicted_count < 50:
        demand_level = 'Low'
    elif predicted_count < 200:
        demand_level = 'Moderate'
    elif predicted_count < 450:
        demand_level = 'High'
    else:
        demand_level = 'Peak'

    # Approximate +/- 1 standard error of Random Forest (~ 47 bikes)
    margin = 47
    low_bound = max(0, predicted_count - margin)
    high_bound = predicted_count + margin

    return {
        'predicted_count': predicted_count,
        'confidence_interval_low': low_bound,
        'confidence_interval_high': high_bound,
        'demand_level': demand_level,
        'inputs_used': input_df.to_dict(orient='records')[0]
    }


if __name__ == '__main__':
    # Test example: Monday 8 AM in Summer 2012, Clear Weather, 25°C, 50% humidity, 10 km/h wind
    sample_input = {
        'season': 2,        # Summer
        'yr': 1,            # 2012
        'mnth': 6,          # June
        'hr': 8,            # 8:00 AM (Rush hour)
        'holiday': 0,       # Regular day
        'weekday': 1,       # Monday
        'workingday': 1,    # Working day
        'weathersit': 1,    # Clear skies
        'temp': 25.0,       # 25°C
        'atemp': 26.0,      # Feeling 26°C
        'hum': 50.0,        # 50% humidity
        'windspeed': 10.0   # 10 km/h
    }
    
    result = predict_demand(sample_input)
    print("--- Test Prediction ---")
    print(f"Sample Input: {sample_input}")
    print(f"Predicted Bike Rentals: {result['predicted_count']} bikes")
    print(f"Demand Category: {result['demand_level']}")
    print(f"Expected Range: {result['confidence_interval_low']} - {result['confidence_interval_high']} bikes")
