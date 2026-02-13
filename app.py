import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# --- 1. AUTO-TRAINER (The MLOps Logic) ---
def initial_setup():
    if not os.path.exists('churn_model.pkl'):
        # Creating a realistic dataset of 1000 customers
        np.random.seed(42)
        data = {
            'tenure': np.random.randint(1, 72, 1000),
            'monthly_charges': np.random.uniform(20, 120, 1000),
            'total_charges': np.random.uniform(100, 5000, 1000),
            'churn': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
        }
        df = pd.DataFrame(data)
        X = df.drop('churn', axis=1)
        y = df['churn']
        
        # Training the AI
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Saving the "Brain"
        joblib.dump(model, 'churn_model.pkl')
        return model
    else:
        return joblib.load('churn_model.pkl')

# --- 2. UI CONFIGURATION ---
st.set_page_config(page_title="Service-In-A-Box", layout="wide")
model = initial_setup()

st.title("🛡️ Customer Churn 'Service-In-A-Box'")
st.info("MLOps Project: Automated Pipeline & Predictive Service")

# --- 3. THE UI TABS ---
tab1, tab2 = st.tabs(["Single Prediction", "Bulk Analysis (CSV)"])

with tab1:
    st.subheader("Individual Customer Risk Assessment")
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.slider("Tenure (Months)", 1, 72, 12)
        monthly = st.number_input("Monthly Charges ($)", 20, 150, 50)
        total = st.number_input("Total Charges ($)", 20, 8000, 500)
    
    if st.button("Calculate Churn Risk"):
        features = np.array([[tenure, monthly, total]])
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]
        
        with col2:
            if prediction == 1:
                st.error(f"### High Risk: {round(prob*100, 2)}%")
                st.write("Recommendation: Offer a loyalty discount immediately.")
            else:
                st.success(f"### Low Risk: {round(prob*100, 2)}%")
                st.write("Recommendation: Target for upselling premium services.")

with tab2:
    st.subheader("Batch Processing")
    uploaded_file = st.file_uploader("Upload Customer CSV List", type="csv")
    if uploaded_file:
        batch_data = pd.read_csv(uploaded_file)
        
        # --- DATA CLEANING FOR TELCO DATASET ---
        # This renames the columns to match what the AI expects
        rename_map = {
            'MonthlyCharges': 'monthly_charges',
            'TotalCharges': 'total_charges',
            'tenure': 'tenure' 
        }
        # Apply the renaming and handle missing values
        clean_batch = batch_data.rename(columns=rename_map)
        
        # Ensure TotalCharges is numeric (it sometimes comes as text)
        clean_batch['total_charges'] = pd.to_numeric(clean_batch['total_charges'], errors='coerce').fillna(0)
        
        # Select only the features the model needs
        features_needed = ['tenure', 'monthly_charges', 'total_charges']
        
        try:
            preds = model.predict(clean_batch[features_needed])
            batch_data['Churn_Prediction'] = ["Likely to Leave" if p == 1 else "Likely to Stay" for p in preds]
            st.success("Analysis Complete!")
            st.write(batch_data)
            st.download_button("Download Report", batch_data.to_csv(index=False), "churn_report.csv")
        except KeyError as e:
            st.error(f"Missing columns! Ensure your CSV has: {features_needed}")