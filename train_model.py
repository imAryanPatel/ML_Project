import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from sklearn.neural_network import MLPClassifier

data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

print(f"Dataset shape: {data.shape}")
print(f"First few rows:\n{data.head()}")
print(f"Column info:\n{data.info()}")

# Data preprocessing
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce') #String to numerical

data.dropna(inplace=True)

data.drop('customerID', axis=1, inplace=True)

columns_to_remove = [
    'PhoneService', 'MultipleLines', 'InternetService', 
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'PaperlessBilling', 'PaymentMethod'
]
data.drop(columns=columns_to_remove, inplace=True)

# Convert target variable to binary
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

X = data.drop('Churn', axis=1)
y = data['Churn']

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

print(f"Categorical features: {categorical_features.tolist()}")
print(f"Numerical features: {numerical_features.tolist()}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Create dictionary to store all models and their results
models = {
    'Logistic Regression': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    
    'Random Forest': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    
    'XGBoost': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(random_state=42))
    ]),
    
    'Neural Network': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))
    ])
}

# Training and evaluation
results = {}
best_accuracy = 0
best_model_name = None

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': conf_matrix
    }
    
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{report}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    
    # Track best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name

# Print the best model
print(f"\nBest Model: {best_model_name} with accuracy {best_accuracy:.4f}")

# Save the best model to a pickle file
best_model = models[best_model_name]
with open('churn_prediction_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

# Save feature names and their types for the Streamlit app
feature_info = {
    'categorical_features': categorical_features.tolist(),
    'numerical_features': numerical_features.tolist()
}

with open('feature_info.pkl', 'wb') as file:
    pickle.dump(feature_info, file)

print("Model and feature information saved successfully!")
