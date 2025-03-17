# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 19:46:58 2025

@author: harpr
"""
#%%
import pandas as pd 
import numpy as np
import re

#%%
import matplotlib.pyplot as plt
import seaborn as sns

#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error

#%% Reading the dataset
df = pd.read_csv('Mobiles Dataset (2025).csv', encoding='latin1')
#%%
df = df.rename(columns={'Launched Year':'Launched_Year'})
#%% Rename Columns

df = df.rename(columns={'Company Name':'CompanyName','Model Name':'ModelName','Mobile Weight':'MobileWeight(g)',
                        'Front Camera':'FrontCamera(MP)','Back Camera':'BackCamera(MP)','Battery Capacity':'BatteryCapacity(mah)',
                        'Screen Size':'ScreenSize(inches)','Launched Price (USA)':'Price(USD)'})



#%% phones only after the 2020
df=df.query('Launched_Year> 2020')

#%% Checking the null Values
df.isnull().sum()

#%%
df.dtypes
#%% Dropping the Columns
df = df.drop(columns=['Launched Price (Pakistan)','Launched Price (China)','Launched Price (India)','Launched Price (Dubai)'])

#%% DATA CLEANING
def clean_battery(value):
    return float(value.replace("mAh", "").replace(",", "").strip())

df["BatteryCapacity(mah)"] = df["BatteryCapacity(mah)"].apply(clean_battery)
#%%
def clean_screen_size(value):
    value = re.sub(r"\(.*?\)", "", str(value))  # Remove anything inside parentheses
    sizes = [float(s.strip().replace("inches", "")) for s in value.split(",")]  # Remove "inches" & convert
    return max(sizes) if sizes else np.nan

df["ScreenSize(inches)"] = df["ScreenSize(inches)"].apply(clean_screen_size)
#%%
df['Price(USD)'] = df['Price(USD)'].str.replace('USD ', '', regex=True)
df['Price(USD)'] = df['Price(USD)'].str.replace(',', '', regex=True)
df['Price(USD)'] = df['Price(USD)'].astype(float)

#%%
def clean_front_camera(value):
    values = [float(v) for v in re.findall(r"\d+\.?\d*", str(value))]  # Extract numeric values
    return max(values) if values else np.nan  # Take the highest MP value

df["FrontCamera(MP)"] = df["FrontCamera(MP)"].apply(clean_front_camera)
#%%
def clean_back_camera(value):
    value = re.sub(r"\(.*?\)", "", str(value))  # Remove text inside parentheses
    values = [float(v) for v in re.findall(r"\d+\.?\d*", value)]  # Extract numeric values
    return max(values) if values else np.nan  # Take the highest MP value

df["BackCamera(MP)"] = df["BackCamera(MP)"].apply(clean_back_camera)
#%%
def clean_ram(value):
    value = value.replace("GB", "").strip()
    return max(map(float, value.split("/"))) if "/" in value else float(value)

df["RAM"] = df["RAM"].apply(clean_ram)
#%%
df['MobileWeight(g)'] = df['MobileWeight(g)'].str.replace('g', '', regex=True)
df['MobileWeight(g)'] = df['MobileWeight(g)'].astype(float)
#%%

#%%
df["Battery Capacity"] = df["Battery Capacity"].astype(str)
df["RAM"] = df["RAM"].astype(str)

#%% Check for any Outliers using Scatter Plot

plt.scatter(x='MobileWeight(', y='Price(USD)', data = df)
plt.show()

#%%
df[df['Price(USD)'] > 15000]
#Drop 685
#%%
df = df.drop(index=685)

#%%
plt.scatter(x='RAM', y='Price(USD)', data = df)
plt.show()

#%%
plt.scatter(x='FrontCamera(MP)', y='Price(USD)', data = df)
plt.show()
#OK
#%%
plt.scatter(x='BackCamera(MP)', y='Price(USD)', data = df)
plt.show()
#OK
#%%
df[df['BackCamera(MP)'] > 150].count()
#%%
plt.scatter(x='BatteryCapacity(mah)', y='Price(USD)', data = df)
plt.show()
#OK
#%%
plt.scatter(x='ScreenSize(inches)', y='Price(USD)', data = df)
plt.show()
#OK
#%%
plt.scatter(x='Launched_Year', y='Price(USD)', data = df)
plt.show()
#OK
#%% Dealing with the Null Values
df.isnull().sum()
#Good no Null Values to fix
#%%
df=df.drop(columns=['CompanyName','ModelName','Processor'
    ])

#%%
df.columns
#%%
from sklearn.linear_model import LinearRegression
#%%
X = df
#%%
y = df['Price(USD)']
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=5)
#%%
lr = LinearRegression()
#%%
lr.fit(X_train, y_train)
#%%
lr.score(X_train, y_train)
#%%
lr.score(X_test, y_test)
#%%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#%%
y_pred = lr.predict(X_test)
#%%
mean_absolute_error(y_test, y_pred)
#%%
mean_squared_error(y_test, y_pred)
#%%
r2_score(y_test, y_pred)
#%%
lr.coef_
#%%
lr.intercept_

