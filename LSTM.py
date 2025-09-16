import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# Load the data
stock_data = pd.read_csv('apple_stock_data_2010.csv')

data = stock_data['Adj Close'].values.reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create the scaled training data set
train_data_len = int(np.ceil( len(data) * .8 ))
train_data = scaled_data[0:train_data_len, :]

# Split the data into x_train and y_train data sets
x_train = []
y_train = []

# Assuming you're using 60 days as your time step
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

# Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data as LSTM expects 3-D data: [samples, time steps, features]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))