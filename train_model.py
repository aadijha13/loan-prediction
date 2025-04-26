import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load your dataset
data = pd.read_csv('/Users/aadijha/Desktop/loan/loanda.csv')

# Data preprocessing (fill missing values, encode categorical data, etc.)
# Refer to your code to handle null values, convert categorical variables to numerical, etc.

# Define X and y for model training
X = data.drop('Loan_Status', axis=1)  # Adjust according to your column names
y = data['Loan_Status']               # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model
with open('loan_model.pkl', 'wb') as file:
    pickle.dump(model, file)
