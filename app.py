import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def add_bg_image():
    st.markdown(
    """
    <style>
    .stApp {
        background: url("https://img.freepik.com/premium-photo/abstract-city-building-skyline-metropolitan-area-contemporary-color-style-futuristic-effects-real-estate-property-development-innovative-architecture-engineering-concept_31965-26878.jpg") no-repeat center fixed;
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Call the function to add the background image
add_bg_image()
# Streamlit app title
st.markdown('<h1 style="font-family:sans-serif; color: black; font-weight: bold;">Singapore Flat Resale Price Predictor</h1>', unsafe_allow_html=True)

# Upload the dataset
st.header("Upload the Resale Flat Data CSV")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the dataset
    resale_data = pd.read_csv('resale data.csv')

    # Data preprocessing
    st.write("### Preprocessing the Data...")
    
    # Extract start and end of storey range
    resale_data['storey_range_start'] = resale_data['storey_range'].str.split(' TO ').str[0].astype(int)
    resale_data['storey_range_end'] = resale_data['storey_range'].str.split(' TO ').str[1].astype(int)
    
    # Calculate flat age
    resale_data['transaction_year'] = pd.to_datetime(resale_data['month']).dt.year
    resale_data['flat_age'] = resale_data['transaction_year'] - resale_data['lease_commence_date']
    
    # Convert remaining lease to months
    def convert_lease_to_months(lease_str):
        years, months = 0, 0
        if "years" in lease_str:
            years = int(lease_str.split("years")[0].strip())
        if "months" in lease_str:
            months = int(lease_str.split("months")[0].split()[-1].strip())
        return years * 12 + months
    
    resale_data['remaining_lease_months'] = resale_data['remaining_lease'].apply(convert_lease_to_months)
    
    # Drop unnecessary columns
    columns_to_drop = ['month', 'block', 'street_name', 'storey_range', 'remaining_lease']
    resale_data_cleaned = resale_data.drop(columns=columns_to_drop)
    
     # One-hot encode categorical variables
    categorical_columns = ['town', 'flat_type', 'flat_model']
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_features = encoder.fit_transform(resale_data_cleaned[categorical_columns])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))


    
    # Combine encoded features with numerical data
    numerical_columns = ['floor_area_sqm', 'storey_range_start', 'storey_range_end', 
                         'flat_age', 'remaining_lease_months']
    X = pd.concat([resale_data_cleaned[numerical_columns], encoded_df], axis=1)
    y = resale_data_cleaned['resale_price']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest Regressor
    st.write("### Training the Model...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f"Model Performance on Test Data:")
    st.write(f"- Mean Absolute Error (MAE): {mae:,.2f}")
    st.write(f"- Mean Squared Error (MSE): {mse:,.2f}")
    st.write(f"- Root Mean Squared Error (RMSE): {rmse:,.2f}")
    st.write(f"- R² Score: {r2:.2f}")
    
    # User input for prediction
    st.header("Enter Flat Details for Prediction")
    town = st.selectbox("Town", encoder.categories_[0])
    flat_type = st.selectbox("Flat Type", encoder.categories_[1])
    flat_model = st.selectbox("Flat Model", encoder.categories_[2])
    floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=10.0, step=0.5)
    storey_range_start = st.slider("Storey Range Start", 1, 40, 1)
    storey_range_end = st.slider("Storey Range End", 1, 40, 10)
    flat_age = st.number_input("Flat Age (years)", min_value=0, step=1)
    remaining_lease_months = st.number_input("Remaining Lease (months)", min_value=0, step=1)
    
    # Predict button
    if st.button("Predict Resale Price"):
        # Prepare input for prediction
        user_data = pd.DataFrame({
            "floor_area_sqm": [floor_area_sqm],
            "storey_range_start": [storey_range_start],
            "storey_range_end": [storey_range_end],
            "flat_age": [flat_age],
            "remaining_lease_months": [remaining_lease_months]
        })
        # One-hot encode categorical inputs
        user_categorical = pd.DataFrame(encoder.transform([[town, flat_type, flat_model]]), 
                                        columns=encoder.get_feature_names_out())
        user_data_final = pd.concat([user_data, user_categorical], axis=1)
        
        # Predict resale price
        predicted_price = rf_model.predict(user_data_final)
        st.success(f"The predicted resale price is: SGD {predicted_price[0]:,.2f}")

    st.write("### Distribution of Resale Prices")
    fig, ax = plt.subplots()
    sns.histplot(resale_data['resale_price'], bins=30, ax=ax)
    st.pyplot(fig)

    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(resale_data_cleaned.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

