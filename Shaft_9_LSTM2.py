# Import the necessary modules

import numpy as np
import pandas as pd


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler

#Read the file

df = pd.read_csv('Shaft_9_LSTM2.csv')

# Separate dates for future plotting

train_dates = pd.to_datetime(df['Date'])

# Variables for training

cols = list(df)[1:6]

df_for_training = df[cols].astype(float)

# LSTM uses sigmoid and tanh that are sensitive to magnitude, so values need to be normalised
# Normalise the data set

scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)

# As required for LSTM networks, we require to reshape an input data into n_samples x timesteps

X_train = []
y_train = []

# Number of days we want to predict into the future
n_future = 1

# Number of past days we want to use to predict the future
n_past = 250

for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    X_train.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    y_train.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

X_train, y_train = np.array(X_train), np.array(y_train)


# Define Autoencoder model

model = Sequential()
model.add(LSTM(64, activation = 'relu', input_shape = (X_train.shape[1], X_train.shape[2]), return_sequences = False))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1]))

model.compile(optimizer = 'adam', loss = 'mse')

#fit the model
history = model.fit(X_train, y_train, epochs = 15, batch_size = 32, validation_split = 0.15, verbose = 1)


#Saving the model to disk

import pickle

model.save('LSTM2.h5')
with open('LSTM2_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)






















































