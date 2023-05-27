

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping

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
    t = 87660 * 10 # for 10 years
    reliability = np.exp(-lambda_ * t) * 100
    return reliability

data['reliability'] = data.apply(calculate_reliability, axis=1)

# Feature Engineering: Add new feature if possible, for now just square of 'N'
data['N_squared'] = data['N']**2

# Extract features and targets
X = data[['V', 'f', 'T', 'N', 'N_squared']].values
Y_reliability = data['reliability'].values

# Normalize the inputs
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, Y_reliability, test_size=0.05, random_state=123)

# Define the model with reduced complexity
model = keras.Sequential([
keras.layers.Dense(64, activation='relu', input_shape=[X.shape[1]]),
keras.layers.Dense(32, activation='relu'),
keras.layers.Dense(1) # One output: reliability
])

# Adam optimizer with lower learning rate
optimizer = keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss='mean_squared_error', optimizer=optimizer)

# Define early stopping
early_stopping = EarlyStopping(
min_delta=0.001, # minimium amount of change to count as an improvement
patience=20, # how many epochs to wait before stopping
restore_best_weights=True
)

# Train the model with early stopping
history = model.fit(
X_train, y_train,
validation_split=0.07,
epochs=500,
batch_size=32,
callbacks=[early_stopping]
)

# Plot training & validation loss values
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
plt.scatter(y_test, predicted_reliability)
plt.xlabel('True Values [Reliability]')
plt.ylabel('Predictions [Reliability]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

Calculate and print R^2 and MSE
r2 = r2_score(y_test, predicted_reliability)
mse = mean_squared_error(y_test, predicted_reliability)
print('R^2:', r2)
print('MSE:', mse)

# Function to make recommendations
def recommend_changes(X, model, scaler):
# Threshold for reliability
    threshold = 0.99999


# Names of the features for recommendations
feature_names = ['V', 'f', 'T', 'N', 'N_squared']

# Normalize the input
X = scaler.transform(X)

# Make predictions
predicted_reliability = model.predict(X)

# Prepare a DataFrame for the recommendations
recommendations = pd.DataFrame(X, columns=feature_names)
recommendations['Predicted Reliability'] = predicted_reliability

# Make a recommendation for each prediction
recommendations['Recommendation'] = recommendations.apply(
    lambda row: np.random.choice(feature_names) if row['Predicted Reliability'] < threshold else 'No changes',
    axis=1
)

return recommendations

# Generate recommendations
X_new = np.random.rand(100, 5) # Replace this with your actual new data
recommendations = recommend_changes(X_new, model, scaler)

# Save to Excel
recommendations.to_excel('recommend_data.xlsx', index=False)







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