

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Load the data
data = pd.read_csv(r'A:\Ultra_Aim_Pro\dir\dataest_long_3_new.txt', delimiter='\t')

# Check if columns have the correct data type
columns_to_check = ['V', 'f', 'T', 'N', 'ttf']
for col in columns_to_check:
    if not pd.api.types.is_numeric_dtype(data[col]):
        print(f'Column {col} is not numeric.')
        # Convert to numeric, forcing non-numeric values to NaN
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows with any NaN values
data = data.dropna()

# Add a new column for the reliability
def calculate_reliability(row):
    ttf = row['ttf']
    lambda_ = 1 / ttf
    t = 87660 * 10  # for 10 years
    reliability = np.exp(-lambda_ * t) * 100
    return reliability

data['reliability'] = data.apply(calculate_reliability, axis=1)

# Extract features and targets
X = data[['V', 'f', 'T', 'N']].values
Y_reliability = data['reliability'].values

# Normalize the inputs
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, Y_reliability, 
                                                    test_size=0.05, 
                                                    random_state=42)

# Define the model
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=[X.shape[1]]),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)  # One output: reliability
])

model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
history = model.fit(X_train, y_train, validation_split=0.001, epochs=150)

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
predicted_reliability = model.predict(X_test)

# Plot Predictions vs True values
plt.figure()
plt.scatter(y_test, predicted_reliability)
plt.xlabel('True Values [Reliability]')
plt.ylabel('Predictions [Reliability]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

# Calculate and print R^2 and MSE
r2 = r2_score(y_test, predicted_reliability)
mse = mean_squared_error(y_test, predicted_reliability)
print('R^2: ', r2)
print('MSE: ', mse)





'''
This code will load your data, compute the reliability for each row, 
split the data into a training set and a test set, train a 
neural network to predict reliability based on the input features, 
plot the loss during training, make predictions on the test set, 
plot the predictions against the true values, and finally calculate 
and print the R^2 score and mean squared error for the predictions.

Please note that the plot of predicted reliability vs true reliability 
may not make sense if the reliability values aren't in the 
range of [-100, 100]. You might need to adjust the range for the plot 
depending on the actual range of your reliability values.
'''