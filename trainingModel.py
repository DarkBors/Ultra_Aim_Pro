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
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from math import exp



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset from the uploaded file
data = pd.read_csv('test_set.txt', delimiter='\t')

# Extract the features and target variables from the dataset
N = data['N'].values
f = data['f'].values
V = data['V'].values
T = data['T'].values
ttf = data['ttf'].values

# Split the data into training and testing sets
N_train, N_test, f_train, f_test, V_train, V_test, T_train, T_test, ttf_train, ttf_test = train_test_split(N, f, V, T, ttf, test_size=0.2, random_state=42)

# Normalize the features (if needed)
# You can apply normalization here if required

# Predicted data
predicted_data = model.predict([N_test, f_test, V_test, T_test])

# Calculate MSE
mse = np.mean((predicted_data - ttf_test)**2)

# Calculate R^2 score
mean_ttf = np.mean(ttf_test)
sst = np.sum((ttf_test - mean_ttf)**2)
ssr = np.sum((predicted_data - ttf_test)**2)
r2 = 1 - (ssr / sst)

print("MSE:", mse)
print("R^2:", r2)




'''
In this modified code:

It first checks if all the required columns are numeric. If not, 
it will convert them to numeric values, turning any non-convertible values into NaN.
It then drops any rows with NaN values.
After these checks and preprocessing steps, 
it proceeds with the steps in the original code: 
splitting the data, defining the model, training it, making predictions, 
and finally evaluating the model's performance.
When extracting the predictions for ttf and deltaT, it uses 
predictions[:,0] and predictions[:,1]. This is because model.predict(X_test) 
returns a 2D array where the first column is the predicted ttf and the second 
column is the predicted deltaT.
'''


'''
#In this updated code:

* A history object is created to store the loss and validation loss at each epoch during training.
  These are later plotted.
* Predictions are made using the test data and are plotted against the true values to visualize 
  the model's performance.
* Finally, R^2 and MSE scores are calculated and printed using sklearn's r2_score and 
  mean_squared_error methods respectively.
* Please replace <ambient_temperature> with the actual ambient temperature value.

-----------
this is some lines from my dataset:
N	f		V		T		ttf
3	1269487110	0.820748689	112.5442076	881211099.3
3	873186570	0.820748689	112.5442076	1793853191
3	704117360	0.820748689	112.5442076	1208016733
3	757412590	0.820748689	112.5442076	7269858120
3	527897880	0.820748689	112.5442076	1174373713
3	639490630	0.820748689	112.5442076	4552341904
3	1115279900	0.820748689	112.5442076	1237579435
3	704146520	0.820748689	112.5442076	704401615.8
3	587069470	0.820748689	112.5442076	594447950.6
3	536392790	0.820748689	112.5442076	4767140926

Which normalization technic is better suited for 
NN \ ML algorithem here, based of my code earlier? 

* reminder - my project is :
A python code based on neural network to predict as perfect as possible the reliability of an FPGA.
Train a model by using previous data from a .txt file (the data contains measured values of Voltage, Frequency, CPUTemperature,TTF and Number of dynamic logic).
The model will try to predict the TTF (Time To Fail) given the following inputs: Voltage, Frequency and Number Of Dynamic Logic.
Then - the reliability will be calculate using this formula: exp(-87660 / ttf) * 100 (the answer is in %), so the best prediction will be the one that will give a reliability precentage as close as possible to a 100%.
*87600 is the number of hours in 10 years - so we want the reliability (probability of failure free operation) after 10 years of operation.
Also - we want the prediction to consider the temperature as well - so the output will be deltaT which is the difference between the Ambient temperature and the Core temperature.
Given the task at hand and the features in your dataset, normalization using the StandardScaler() function (as you are currently doing) should be suitable for your Neural Network / Machine Learning algorithm.
-->

The StandardScaler() performs standardization, which scales the features such that they have zero mean and unit variance. Standardization does not bound values to a specific range like normalization, instead it adjusts and scales the data based on the actual distribution of the values.

Standardization is commonly used in machine learning algorithms as it maintains all of the features in the same scale, which can be useful for models that use a form of gradient descent for optimization.

Your dataset seems to contain features with different scales (like 'V', 'f', 'N', 'T'), so using StandardScaler() to standardize these inputs to a common scale will likely be beneficial for your model.

However, please note that it's always a good idea to experiment with different types of preprocessing and normalization/standardization to see what works best with your specific dataset and task. For instance, you could also try MinMaxScaler, which scales and translates each feature individually such that it is in the given range on the training set, typically between zero and one.

Also, please note that your targets (TTF and deltaT) are not being normalized in your code. As I mentioned earlier, targets aren't typically normalized unless you're dealing with specific kinds of models or training regimens. If your model's performance is not satisfactory, you could experiment with normalizing the targets as well.

'''
