import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib

def train_final_optimized():
    print("🚀 PHASE 3: Starting Hyperparameter Tuning (Grid Search)...")

    #1. Load and clean data (The Robust pipeline from Phase 2)
    df  = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], 
                                       errors='coerce').fillna(df['TotalCharges']
                                                               .median())
    
    features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X = df[features]
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Define the Hyperparameter Grid
    # We test different numbers of trees and different depths to find the best balance
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }

    print("⏳ Running GridSearchCV across 18 combinations... (This may take a moment)")
    
    # cv=5 means 5-fold Cross-Validation
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42), 
        param_grid=param_grid, 
        cv=5, 
        n_jobs=-1, # Uses all your CPU cores for speed
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)

    # 3. Results and Export
    best_model = grid_search.best_estimator_
    
    print("-" * 30)
    print(f"✅ Best Parameters: {grid_search.best_params_}")
    print(f"📈 Final Optimized Accuracy: {best_model.score(X_test, y_test):.4f}")
    print("-" * 30)

    # Save the final high-performance artifact
    joblib.dump(best_model, 'churn_model.pkl')
    print("📁 Production-ready 'churn_model.pkl' exported successfully!")

if __name__ == "__main__":
    train_final_optimized()