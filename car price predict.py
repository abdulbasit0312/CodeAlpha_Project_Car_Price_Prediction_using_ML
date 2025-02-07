# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1xE7TmUcLW-_DM_3joknsynpstHH2_UDj
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
file_path = '/content/car pridiction price.csv'
df = pd.read_csv(file_path)
print(df.head())

df.drop(columns=['torque'], inplace=True)
print(df.head())

df.shape

#PreProcessing

#NULL Check

df.isnull().sum()

df.dropna(inplace=True)
df.shape

#Duplicate Check

df.duplicated().sum()

df.drop_duplicates(inplace=True)
df.shape

df

df.info()

#Data Analysis

for col in df.columns:
  print('Unique valeus of ' + col)
  print(df[col].unique())
  print("===============")

def get_brand_name(car_name):
  car_name = car_name.split(' ')[0]
  return car_name.strip()
  get_brand_name('Maruti  Suzuki Dzire VDI')

def clean_data(value):
  value = value.split(' ')[0]
  value = value.strip()
  if value == '-':
    value = 0
  return float(value)
  get_brand_name('Maruti  Suzuki Dzire VDI')

df['name'] = df['name'].apply(get_brand_name)
df['name'].unique()

import pandas as pd

file_path = '/content/car pridiction price.csv'
df = pd.read_csv(file_path)

def get_brand_name(value):
    return value.split(' ')[0] if isinstance(value, str) else value
df['mileage'] = df['mileage'].apply(clean_data)
df['max_power'] = df['max_power'].apply(clean_data)
df['engine'] = df['engine'].apply(clean_data)

for col in df.columns:
    print('Unique valeus of ' + col)
    print(df[col].unique())
    print("===============")

df['name'].replace(
    ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
     'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
     'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
     'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
     'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
     21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    inplace=True
)
print("Transmission unique values before replacement:", df['transmission'].unique())
df['transmission'].replace(['manual', 'Automatic'], [1, 2], inplace=True)
print("Seller type unique values before replacement:", df['seller_type'].unique())
df['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
df.info()

df['fuel'].unique()

import pandas as pd

df['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
df.info()

df.reset_index(inplace=True)
df

df['owner'].unique()

df['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
       'Fourth & Above Owner', 'Test Drive Car'],
                    [1,2,3,4,5], inplace=True)
df.info()

df.drop(columns=['index'], inplace=True)
df

input_data = df.drop(columns=['selling_price'])
output_data = df['selling_price']
X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2)

#Model Creation

model = LinearRegression()

#Train Model

!pip install scikit-learn pandas

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

file_path = '/content/car pridiction price.csv'
df = pd.read_csv(file_path)

print("Unique mileage values before cleaning:")
print(df['mileage'].unique())

df['mileage'] = df['mileage'].str.replace(' kmpl', '')
df['mileage'] = df['mileage'].str.replace(' km/kg', '')
df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')

df['engine'] = df['engine'].str.replace(' CC', '')
df['engine'] = pd.to_numeric(df['engine'], errors='coerce')
df['max_power'] = df['max_power'].str.replace(' bhp', '')
df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')
df.dropna(subset=['mileage', 'engine', 'max_power'], inplace=True)
print("Data after cleaning:")
print(df.info())
print(df.head())

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

numerical_features = ['year', 'km_driven', 'fuel', 'seller_type',
                     'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
X = pd.get_dummies(df[numerical_features], drop_first=True)
y = df['selling_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predict = model.predict(X_test)

predict

X_train.head(1)

import pandas as pd

input_data_model = pd.DataFrame(
    [[5, 2022, 1200, 1, 1, 1, 1, 12.99, 2494.0, 100.6, 8.0, None, None, None, None, None, None, None]], # Added None values to match the number of columns
    columns=['name', 'year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats', 'fuel_Diesel', 'fuel_LPG', 'fuel_Petrol', 'seller_type_Individual', 'seller_type_Trustmark', 'Dealer', 'transmission_Manual', 'owner_Fourth & Above Owner', 'owner_Second Owner', 'owner_Test Drive Car', 'owner_Third Owner']
)

input_data_model

input_data_model_aligned = input_data_model.copy()
input_data_model_aligned = input_data_model_aligned.drop(columns=['Dealer', 'name', 'seller_type_Trustmark'], errors='ignore')
input_data_model_aligned['seller_type_Trustmark Dealer'] = 0
input_data_model_aligned = input_data_model_aligned.fillna(0)
input_data_model_aligned = input_data_model_aligned[X_train.columns]
model.predict(input_data_model_aligned)

import pickle as pk

pk.dump(model, open('model.pkl','wb'))