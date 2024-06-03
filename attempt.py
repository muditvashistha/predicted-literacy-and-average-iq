# WORK FLOW FOR THIS ATTEMPT
# 1. FILTER DATA OUT FOR ASIA RECORDS AND STORE IN A NEW FILE (completed)
# 2. LEARN HOW TO TRAIN THE DATA FOR IQ RECORDS and LITERACY RATE, CREATE A PREDICTION OF IQ UPON TRAINING THE VALUES.
# 3. PLOT THE PREDICTED VALUES AGAINST EACH OTHER

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# filtering the data out
df=pd.read_csv("") #Mention your file's directory here

asia=df[(df['Continent']=='Asia')]

asia.to_csv('asia_record.csv',index=False)

selected_col=df[['Average IQ','Literacy Rate']]

selected_col.to_csv('maindata.csv',index=False)

# declaring testing and training variables and splitting data for training and testing

X=df[['Average IQ']] #features
y=df[['Average IQ','Literacy Rate']] #target variables

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# fitting the model
base_model = LinearRegression()
model = MultiOutputRegressor(base_model)

model.fit(X_train, y_train)
y_pred = model.predict(X_test) #predicting and storing the predicted values on the basis of the test data

# plotting the data
plt.figure(num='Predicted representation of Average IQ against Literacy Rates in Asia as per 2023 poplutaion')
plt.scatter(y_pred[:, 0], y_pred[:, 1])
plt.xlabel('Predicted Average IQ')
plt.ylabel('Predicted Literacy Rate')
plt.title('Predicted Values')
plt.show()


