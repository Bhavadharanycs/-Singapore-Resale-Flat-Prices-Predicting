import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('resale_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit application
st.title("Singapore Flat Resale Price Prediction")
st.write("Estimate the resale value of a flat based on historical data.")

# Input fields
town = st.selectbox("Town", ["ANG MO KIO", "BEDOK", "BISHAN", ...])  # Add all town options
flat_type = st.selectbox("Flat Type", ["2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE"])
storey_range = st.selectbox("Storey Range", ["01 TO 03", "04 TO 06", ...])  # Complete as needed
floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=20, max_value=200, value=60)
lease_commence_date = st.slider("Lease Commence Date", 1960, 2024, 1990)

# Feature engineering
lease_age = 2024 - lease_commence_date

# Predict button
if st.button("Predict Resale Price"):
    # Prepare input data
    input_data = pd.DataFrame([[town, flat_type, storey_range, floor_area_sqm, lease_age]],
                              columns=['town', 'flat_type', 'storey_range', 'floor_area_sqm', 'lease_age'])
    input_data = pd.get_dummies(input_data).reindex(columns=X.columns, fill_value=0)  # Ensure columns match
    
    # Predict
    prediction = model.predict(input_data)[0]
    st.write(f"Estimated Resale Price: ${prediction:,.2f}")
