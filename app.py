import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('resale_price_model.pkl')

# Streamlit app
st.title("Singapore Resale Flat Price Prediction")
st.write("Enter flat details below:")

# User input form
towns = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG', 'BUKIT TIMAH', 
         'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST', 
         'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG', 
         'SENGKANG', 'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN']
flat_types = ['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']
storey_ranges = ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18', '19 TO 21', '22 TO 24', '25 TO 27', '28 TO 30']

# User inputs
town = st.selectbox("Town", towns)
flat_type = st.selectbox("Flat Type", flat_types)
storey_range = st.selectbox("Storey Range", storey_ranges)
floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=10.0, max_value=200.0)
lease_age = st.number_input("Lease Age (years)", min_value=0, max_value=99)

# Predict button
if st.button("Predict Price"):
    # Prepare input data
    input_data = pd.DataFrame({
        'floor_area_sqm': [floor_area_sqm],
        'lease_age': [lease_age]
    })
    # Add dummy variables for categorical inputs
    for category, value in [('town', town), ('flat_type', flat_type), ('storey_range', storey_range)]:
        for col in model.feature_names_in_:
            input_data[col] = 1 if col.endswith(value) else 0

    # Make prediction
    prediction = model.predict(input_data)[0]
    st.write(f"Predicted Resale Price: SGD {prediction:,.2f}")
