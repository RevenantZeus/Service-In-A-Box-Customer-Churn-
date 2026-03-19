import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def train_optimized():
    print("🚀 PHASE 2: Cleaning Data & Training Optimized Random Forest...")

    #Load raw data
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    #--- DATA CLEANING PIPELINE ---
    #Fix the 'Empty String' issue discovered in Phase 1
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    #Impute missing values with median
    df['Total Charges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    #Feature engineering: Select key drivers for churn
    features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X = df[features]
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    #Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Updgrade to Random Forest for better generalization
    print("🧠 Training Random Forest Classifier (100 estimators)...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    #Evaluate Performance
    accuracy = model.score(X_test, y_test)
    print(f"✅ Improved Model Accuracy: {accuracy:.4f}")

    #Save the updated "brain"
    joblib.dump(model, "churn_model.pkl")
    print("📁 Model artifact 'churn_model.pkl' has been updated.")

if __name__ == "__main__":
    train_optimized()
