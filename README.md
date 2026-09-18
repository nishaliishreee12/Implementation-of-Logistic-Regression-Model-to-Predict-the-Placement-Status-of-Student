# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess the dataset by checking missing values and encoding categorical variables into numerical values.

2. Separate features and target, remove unnecessary columns, and split the dataset into training and testing sets.

3. Scale the features, train the Logistic Regression model using the training data, and predict placement status for the test data.

4. Evaluate the model using Accuracy, Confusion Matrix, and Classification Report.
   

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: KARANKUMAR K
RegisterNumber: 212225040171

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv("Placement_Data.csv")

print("First 5 Records:")
print(df.head())


# Check Missing Values

print("\nMissing Values:")
print(df.isnull().sum())


# Encode Categorical Columns

le = LabelEncoder()

categorical_columns = [
    'gender',
    'ssc_b',
    'hsc_b',
    'hsc_s',
    'degree_t',
    'workex',
    'specialisation',
    'status'
]

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])


# Separate Features and Target

X = df.drop(["sl_no", "status", "salary"], axis=1)
y = df["status"]


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression Model

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy =", accuracy)

# Confusion Matrix

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# Classification Report

print("\nClassification Report")
print(classification_report(y_test, y_pred))

*/
```

## Output:

<img width="338" height="422" alt="image" src="https://github.com/user-attachments/assets/777e60a6-855e-479c-86e0-063e81e3cb6d" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
