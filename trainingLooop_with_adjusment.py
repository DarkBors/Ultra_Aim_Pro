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
data = pd.read_csv(r'A:\Ultra_Aim_Pro\dir\data_set1.txt', delimiter='\t')

# Check if columns have the correct data type
columns_to_check = ['V', 'f', 'T', 'N', 'ttf']
for col in columns_to_check:
    if not pd.api.types.is_numeric_dtype(data[col]):
        print(f'Column {col} is not numeric.')
        # Convert to numeric, forcing non-numeric values to NaN
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows with any NaN values
data = data.dropna()

# Extract features and targets
X = data[['V', 'f', 'T', 'N']].values

# Normalize the inputs
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define the initial best hyperparameters
best_hyperparameters = {'validation_split': 0.001, 'epochs': 150, 'test_size': 0.05, 'random_state': 42}
best_r2 = -float('inf')
best_mse = float('inf')

# Define the range of hyperparameter values to explore
validation_splits = [0.03, 0.05, 0.07]
epochs = [300, 350, 400]
test_sizes = [0.05]
random_states = [123]

# Loop over the hyperparameter combinations
for val_split in validation_splits:
    for num_epochs in epochs:
        for test_size in test_sizes:
            for random_state in random_states:
                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(X, X[:, 0], test_size=test_size, random_state=random_state)

                # Define the model
                model = keras.Sequential([
                    keras.layers.Dense(32, activation='relu', input_shape=[X.shape[1]]),
                    keras.layers.Dense(16, activation='relu'),
                    keras.layers.Dense(1)  # One output: reliability
                ])

                model.compile(loss='mean_squared_error', optimizer='adam')

                # Train the model
                history = model.fit(X_train, y_train, validation_split=val_split, epochs=num_epochs, verbose=0)

                # Make predictions
                predicted_reliability = model.predict(X_test)

                # Calculate R^2 and MSE
                r2 = r2_score(y_test, predicted_reliability)
                mse = mean_squared_error(y_test, predicted_reliability)

                # Check if current hyperparameters yield better results
                if r2 > best_r2:
                    best_r2 = r2
                    best_mse = mse
                    best_hyperparameters['validation_split'] = val_split
                    best_hyperparameters['epochs'] = num_epochs
                    best_hyperparameters['test_size'] = test_size
                    best_hyperparameters['random_state'] = random_state

# Update the model with the best hyperparameters
X_train, X_test, y_train, y_test = train_test_split(X, X[:, 0], test_size=best_hyperparameters['test_size'], random_state=best_hyperparameters['random_state'])
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=[X.shape[1]]),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)  # One output: reliability
])
model.compile(loss='mean_squared_error', optimizer='adam')
print(X.shape)
print(y.shape)


# Train the model with the best hyperparameters
history = model.fit(X_train, y_train, validation_split=best_hyperparameters['validation_split'], epochs=best_hyperparameters['epochs'])

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
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

# Calculate and print R^2 and MSE
r2 = r2_score(y_test, predicted_reliability)
mse = mean_squared_error(y_test, predicted_reliability)
print('Best Hyperparameters:')
print(best_hyperparameters)
print('Best R^2:', best_r2)
print('Best MSE:', best_mse)
print('R^2:', r2)
print('MSE:', mse)
