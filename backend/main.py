import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

app = FastAPI(title="Food Delivery Prediction API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'training', 'delivery_ensemble.pkl')
try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class DeliveryRequest(BaseModel):
    Distance_km: float
    Weather: str
    Traffic_Level: str
    Time_of_Day: str
    Vehicle_Type: str
    Preparation_Time_min: int 
    Courier_Experience_yrs: int

EXPECTED_COLUMNS = [
    'Distance_km', 'Traffic_Level', 'Preparation_Time_min',
    'Courier_Experience_yrs', 'Weather_Clear',
    'Weather_Foggy', 'Weather_Rainy', 'Weather_Snowy', 'Weather_Windy',
    'Time_of_Day_Afternoon', 'Time_of_Day_Evening', 'Time_of_Day_Morning',
    'Time_of_Day_Night', 'Vehicle_Type_Bike', 'Vehicle_Type_Car',
    'Vehicle_Type_Scooter']

@app.post("/predict")
def predict_delivery_time(request: DeliveryRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    input_dict = request.model_dump()
    df_input = pd.DataFrame([input_dict])
    #traffic importance
    traffic_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    df_input['Traffic_Level'] = df_input['Traffic_Level'].map(traffic_mapping)
    
    #OHE
    norml_cols = ['Weather', 'Time_of_Day', 'Vehicle_Type']
    df_encoded = pd.get_dummies(df_input, columns=norml_cols, dtype=int)
    
    for col in EXPECTED_COLUMNS:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    df_final = df_encoded[EXPECTED_COLUMNS]
    
    prediction = model.predict(df_final)[0]
    return {"estimated_delivery_minutes": round(float(prediction))}