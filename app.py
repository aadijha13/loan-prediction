from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

app = Flask(__name__)

# Load or preprocess data and train model
def train_model():
    data = pd.read_csv("/Users/aadijha/Desktop/loan/loanda.csv")

    # Fill missing values and preprocess data
    data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])
    data['Married'] = data['Married'].fillna(data['Married'].mode()[0])
    data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])
    data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
    data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].median())
    data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].median())
    data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].median())

    # Encode categorical variables
    data['Gender'] = data['Gender'].replace(('Male', 'Female'), (1, 0))
    data['Married'] = data['Married'].replace(('Yes', 'No'), (1, 0))
    data['Education'] = data['Education'].replace(('Graduate', 'Not Graduate'), (1, 0))
    data['Self_Employed'] = data['Self_Employed'].replace(('Yes', 'No'), (1, 0))
    data['Loan_Status'] = data['Loan_Status'].replace(('Y', 'N'), (1, 0))
    data['Property_Area'] = data['Property_Area'].replace(('Urban', 'Semiurban', 'Rural'), (1, 1, 0))
    data['Dependents'] = data['Dependents'].replace(('0', '1', '2', '3+'), (0, 1, 1, 1))

    # Prepare features and target variable
    X = data.drop(['Loan_ID', 'Loan_Status'], axis=1)
    y = data['Loan_Status']

    # Handle class imbalance
    sm = SMOTE()
    X_res, y_res = sm.fit_resample(X, y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=0)

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Save the model
    with open('loan_model.pkl', 'wb') as file:
        pickle.dump(model, file)

# Uncomment this line to train and save the model (only do this once)
# train_model()

# Load the trained model
model = pickle.load(open('loan_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    form_data = [int(x) for x in request.form.values()]
    final_data = [np.array(form_data)]
    prediction = model.predict(final_data)

    # Interpret prediction
    output = 'Approved' if prediction[0] == 1 else 'Rejected'
    return render_template('result.html', prediction=output)

if __name__ == '__main__':
    app.run(debug=True)
