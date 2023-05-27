

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf

from tensorflow.keras import *  # changed this line
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping


# Set the backend for matplotlib
matplotlib.use('Agg')

# Load the data
data = pd.read_excel(r'A:\Ultra_Aim_Pro\dir\DataSet_1.xlsx')

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

# Feature Engineering: Add new feature if possible, for now just square of 'N'
data['N_squared'] = data['N'] ** 2

# Extract features and targets
X = data[['V', 'f', 'T', 'N', 'N_squared']].values
Y_reliability = data['reliability'].values

# Normalize the inputs
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, Y_reliability, test_size=0.05, random_state=123)

# Define the model architecture
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=[X.shape[1]]),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Define early stopping
early_stopping = EarlyStopping(
    min_delta=0.001,  # minimum amount of change to count as an improvement
    patience=20,  # how many epochs to wait before stopping
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_split=0.07,
    epochs=700,
    batch_size=62,
    callbacks=[early_stopping]
)

# Plot training & validation loss values
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('loss_plot.png')

# Evaluate the model
predicted_reliability = model.predict(X_test)
r2 = r2_score(y_test, predicted_reliability)
mse = mean_squared_error(y_test, predicted_reliability)
print('R^2:', r2)
print('MSE:', mse)








'''
Time and date: 23/05/2023 22:16

In this modification, we have:

Added feature engineering where we added a new feature 'N_squared', which is the square of 'N'. This is a placeholder, you should replace it with a meaningful feature according to your data.

Increased model complexity by adding another dense layer and increasing the number of neurons in the existing layers.

Adjusted the learning rate in the Adam optimizer to 0.001 (this is the default value in TensorFlow). You should tune this value as needed.

Increased the number of training epochs to 500 for longer training.

Included batch size in model fitting as 32 (this is the default value in TensorFlow). This can be tuned according to your needs and system capabilities.
'''




'''
Time and date: 26/05/2023 23:41

The output shows the result of your deep learning model training and the prediction it made on the given dataset. Let's break this down:

Training Results: The first part of your output is the results of each epoch (or iteration) of your training process. Each epoch involves running through your entire dataset once. You've set this to run for 500 epochs. After each epoch, it reports the loss (the measure of error) on both the training dataset (loss) and the validation dataset (val_loss). It's generally good if these numbers are decreasing over time, as it means your model is learning to better fit your data.

Evaluation Metrics: The R^2 and MSE (Mean Squared Error) scores are reported after the training. The R^2 score is a measure of how well your model explains the variation in your data, with 1 being perfect. Your model's R^2 is 0.32, suggesting the model explains about 32% of the variance in your target variable. The MSE is another measure of error, with lower numbers being better.

Recommendation Results: The final section is a table of recommendations that your model has generated. It seems to be based on some calculated parameters (V, f, T, N, N_squared) and predicted reliability. Your model seems to suggest "No changes" for all rows, which suggests that it did not identify any conditions under which a change would be recommended. This might be expected or might suggest that your model needs further tuning or additional data.

'''


'''


'''