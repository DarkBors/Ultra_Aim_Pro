###################################################################################
##                             ULTRA-AIM-PRO                                     ##
##                                                                               ##
## Ultra96-based PYNQ AI-Managed Performance and Reliability Optimization system ##
##                                                                               ##
##                  Created by: Dark Bors version 0.0.1                          ##
##                                                                               ##
##                                                                 Final Project ##
###################################################################################


import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from math import exp
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv(r'A:\Ultra_Aim_Pro\dir\dataest_long_3_new.txt', delimiter='\t')

# Extract features and targets
X = data[['Voltage', 'Frequency', 'CPUTemperature', 'Number of dynamic logic']].values
Y_ttf = data['TTF'].values
Y_deltaT = data['CPUTemperature'].values - <ambient_temperature>  # replace with actual ambient temp

# Normalize the inputs
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train_ttf, y_test_ttf, y_train_deltaT, y_test_deltaT = train_test_split(X, Y_ttf, Y_deltaT, test_size=0.2, random_state=42)

# Define the model
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=[X.shape[1]]),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(2)  # Two outputs: TTF and DeltaT
])

model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
history = model.fit(X_train, [y_train_ttf, y_train_deltaT], validation_split=0.2, epochs=100)

# Plot training & validation loss values
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Make predictions
predictions = model.predict(X_test)
predicted_ttf, predicted_deltaT = predictions[0], predictions[1]

# Compute reliability
reliability = exp(-87660 / predicted_ttf) * 100

# Plot Predictions vs True values
plt.figure()
plt.scatter(y_test_ttf, predicted_ttf)
plt.xlabel('True Values [TTF]')
plt.ylabel('Predictions [TTF]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

# Calculate and print R^2 and MSE
r2 = r2_score(y_test_ttf, predicted_ttf)
mse = mean_squared_error(y_test_ttf, predicted_ttf)
print('R^2: ', r2)
print('MSE: ', mse)





'''
#In this updated code:

* A history object is created to store the loss and validation loss at each epoch during training.
  These are later plotted.
* Predictions are made using the test data and are plotted against the true values to visualize 
  the model's performance.
* Finally, R^2 and MSE scores are calculated and printed using sklearn's r2_score and 
  mean_squared_error methods respectively.
* Please replace <ambient_temperature> with the actual ambient temperature value.

'''
