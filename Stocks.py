To predict stock prices using LSTM in a Jupyter Notebook.

1. **Import Libraries:**
   Import the required libraries.

import IPython
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import trailing
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


2. **Load and Prepare Data:**
   Load the stock price data and preprocess it.


# Load your stock price data as a Pandas DataFrame
df = pd.read_csv('your_stock_data.csv')

# Extract the 'Close' price (or any other feature you want to predict)
data = df['Close'].values.reshape(-1, 1)

# Normalize the data using Min-Max scaling
scaler = MinMaxScaler()
data = scaler.fit_transform(data)


3. **Split Data:**
   Split the data into trailing and testing sets.

train_size = int(len(data) * 0.80)
train_data, test_data = data[:train_size], data[train_size:]


4. **Create Sequences:**
   Create sequences of data for training.


def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequence = data[i:i+seq_length]
        sequences.append(sequence)
    return np.array(sequences)

seq_length = 10  # You can adjust this sequence length
train_sequences = create_sequences(train_data, seq_length)
test_sequences = create_sequences(test_data, seq_length)

5. **Build LSTM Model:**
   Build and compile the LSTM model.

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')


6. **Train the Model:**
   Train the LSTM model using the training data.

model.fit(train_sequences, train_data[seq_length:], epochs=10, batch_size=32)


7. **Make Predictions:**
   Use the trained model to make predictions on the test data.

predicted_data = model.predict(test_sequences)


8. **Plot Results:**
   Plot the actual vs. predicted stock prices.


plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(test_data[seq_length:]), label='Actual Price')
plt.plot(scaler.inverse_transform(predicted_data), label='Predicted Price')
plt.legend()
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
