###################################################################################
##                             ULTRA-AIM-PRO                                     ##
##                                                                               ##
## Ultra96-based PYNQ AI-Managed Performance and Reliability Optimization system ##
##                                                                               ##
##                  Created by: Dark Bors version 0.0.1                          ##
##                                                                               ##
##                                                                 Final Project ##
###################################################################################


import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf
import re

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow import keras
from sklearn import metrics



import pandas as pd

# Load the file
with open(r'A:\Ultra_Aim_Pro\dir\dataest_long_3.txt', 'r') as file:
    lines = file.read().splitlines()

# Remove the first line (headers)
data_lines = lines[1:]

# Split each line by tabs, ignore empty fields and lines
data = [line.split('\t') for line in data_lines if line.strip()]
data = [[field for field in line if field] for line in data]

# Determine the order of columns based on the header line
header = lines[0].split('\t')
desired_order = ['N', 'f', 'V', 'T', 'ttf']
indices = [header.index(col) for col in desired_order]

# Extract the desired columns based on the order
data = [[line[i] if i < len(line) else '' for i in indices] for line in data]

# Assert that each line now has 5 columns, if not print the problematic line
for line in data:
    assert len(line) == 5, f"Problematic line: {line}"

# Create a DataFrame and assign column names
df = pd.DataFrame(data, columns=desired_order)

# Format the float values in the DataFrame to 5 decimal places
df = df.applymap(lambda x: format(float(x), '.5f') if '.' in x else x)

# Manually create a string representation of the dataframe with added spacing
df_str = df.to_string(index=False, header=True, col_space=20)

# Save the dataframe to a text file
with open(r'A:\Ultra_Aim_Pro\dir\dataest_long_3_output.txt', 'w') as file:
    file.write(df_str)
