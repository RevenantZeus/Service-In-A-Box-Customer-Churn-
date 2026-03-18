import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

def train_baseline():
    print("🚀 PHASE 1: Initializing Baseline Model Training...")
    
    # Load the raw dataset
    try:
        df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv') 
        
        # Selecting basic features for the Proof of Concept (POC)
        # Note: Using raw 'TotalCharges' which may require cleaning in v2
        features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        X = df[features]
        y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
        
        # Split into training and validation sets (80/20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Using a standard DecisionTree for the initial baseline artifact
        model = DecisionTreeClassifier(max_depth=10) 
        model.fit(X_train, y_train)
        
        # Calculate performance on the hold-out set
        accuracy = model.score(X_test, y_test)
        print(f"Current Model Accuracy: {accuracy:.4f}")
        
        # Exporting the model artifact for Streamlit integration
        # Note: Initial version uses default DecisionTree parameters
        joblib.dump(model, 'churn_model.pkl')
        print("✅ Model artifact 'churn_model.pkl' updated successfully.")

    except Exception as e:
        print(f"⚠️ Data Type Error Detected: {e}")
        print("Investigation: 'TotalCharges' likely contains non-numeric values. Needs preprocessing in next iteration.")

if __name__ == "__main__":
    train_baseline()