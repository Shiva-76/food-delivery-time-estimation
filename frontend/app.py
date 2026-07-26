import streamlit as st
import requests

API_URL = "https://delivery-backend-api-gfif.onrender.com/predict"

st.set_page_config(page_title="Delivery Estimator", page_icon="🏍️")
st.title("🏍️ Food Delivery Time Estimator")

col1, col2 = st.columns(2)

with col1:
    distance = st.number_input("Distance (km)", min_value=1.0, max_value=50.0, value=5.0)
    weather = st.selectbox("Weather", ['Clear', 'Foggy', 'Rainy','Windy','Snowy'])
    time_of_day = st.selectbox("Time of Day", ['Morning', 'Afternoon', 'Evening', 'Night'])
    courier_exp = st.number_input("Courier Experience (Years)", min_value=0, max_value=20, value=2)

with col2:
    traffic_level = st.selectbox("Traffic Level", ['Low', 'Medium', 'High'])
    vehicle_type = st.selectbox("Vehicle Type", ['Bike','Scooter','Car'])
    prep_time = st.number_input("Preparation Time (min)", min_value=1, max_value=120, value=15) # Added input

if st.button("Predict Delivery Time"):
    payload = {
        "Distance_km": distance,
        "Weather": weather,
        "Traffic_Level": traffic_level,
        "Time_of_Day": time_of_day,
        "Vehicle_Type": vehicle_type,
        "Preparation_Time_min": prep_time,
        "Courier_Experience_yrs": courier_exp
    }
    
    try:
        with st.spinner("Calculating..."):
            response = requests.post(API_URL, json=payload)
            
        if response.status_code == 200:
            result = response.json()
            minutes = result["estimated_delivery_minutes"]
            st.success(f"⏱️ Estimated Delivery Time: {minutes} minutes")
        else:
            st.error(f"Error from API: {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API")