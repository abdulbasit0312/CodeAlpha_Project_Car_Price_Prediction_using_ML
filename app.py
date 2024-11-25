import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

# Load the model
try:
    model = pk.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file 'model.pkl' not found. Please check the file path.")

# Load the dataset
try:
    cars_data = pd.read_csv('Cardetails.csv')
except FileNotFoundError:
    st.error("Dataset file 'Cardetails.csv' not found. Please check the file path.")

# Preprocess the dataset
def get_brand_name(car_name):
    return car_name.split(' ')[0].strip()

if 'name' in cars_data.columns:
    cars_data['name'] = cars_data['name'].apply(get_brand_name)

# Streamlit UI
st.header('Car Price Prediction ML Model')

# Input widgets
name = st.selectbox('Select Car Brand', cars_data['name'].unique())
year = st.slider('Car Manufactured Year', 1994, 2024)
km_driven = st.slider('No of kms Driven', 11, 200000)
fuel = st.selectbox('Fuel type', cars_data['fuel'].unique())
seller_type = st.selectbox('Seller type', cars_data['seller_type'].unique())
transmission = st.selectbox('Transmission type', cars_data['transmission'].unique())
owner = st.selectbox('Owner type', cars_data['owner'].unique())
mileage = st.slider('Car Mileage (kmpl)', 10, 40)
engine = st.slider('Engine Capacity (CC)', 700, 5000)
max_power = st.slider('Max Power (bhp)', 0, 200)
seats = st.slider('Number of Seats', 2, 10)

# Prediction button
if st.button("Predict"):
    try:
        # Prepare input data
        input_data_model = pd.DataFrame(
            [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
            columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
        )
        
        # Replace categorical values with numeric
        input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
                                           'Fourth & Above Owner', 'Test Drive Car'], [1, 2, 3, 4, 5], inplace=True)
        input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
        input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
        input_data_model['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
        input_data_model['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
                                          'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
                                          'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
                                          'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
                                          'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
                                         list(range(1, 32)), inplace=True)

        # Make prediction
        car_price = model.predict(input_data_model)

        # Display the result
        st.success(f"Estimated Car Price: ₹{car_price[0]:,.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
