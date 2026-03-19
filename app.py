import streamlit as st
import pandas as pd
import joblib
import os

# -- 1. MLOps: Load the Pre-Trained Model --
# The @st.cache_resource decorator ensures the model only loads once, improving performance of the app.
@st.cache_resource
def load_model():
    if os.path.exists("churn_model.pkl"):
       return  joblib.load("churn_model.pkl")
    else:
        st.error("Model file 'churn_model.pkl' not found. Please run train_model.py first.")
        return None
    
# -- 2. UI CONFIGURATION --
st.set_page_config(page_title= "Service-In-A-Box", layout= "wide")
model = load_model()

if model: #Only build UI if model is successfully loaded
    st.title("🛡️ Customer Churn 'Service-In-A-Box")
    st.info("MLOps Project: Automated Pipeline & Predictive Service")

    # -- 3. THE UI TABS --
    tab1, tab2 = st.tabs(["Single Prediction", "Bulk Analysis (CSV)"])

    with tab1:
        st.subheader("Individual Customer Risk Assessment")
        col1, col2 = st.columns(2)
    
        with col1:
            tenure = st.slider("Tenure (months)", 1, 72, 12)
            monthly = st.number_input("Monthly Charges ($)", 20, 150, 50)
            total = st.number_input("Total Charges ($)", 20, 8000, 500)

        if st.button("Calculate Churn Risk"):
            #Fix: We must pass the data as a DataFrame with exact column names to avoid Sklearn Warnings.
            features = pd.DataFrame([[tenure, monthly, total]],
                                    columns= ['tenure', 'MonthlyCharges', 'TotalCharges'])
            
            prediction = model.predict(features)[0]
            prob = model.predict_proba(features)[0][1]

            with col2:
                if prediction == 1:
                    st.error(f"### High Risk: {round(prob*100, 2)}%")
                    st.write("Recommendation: Offer a Loyalty Discount Immediately.")
                else:
                    st.success(f"### Low Risk: {round((1-prob)*100, 2)}%")
                    st.write("Recommendation: Target for Upselling Premium Services.")
                    
    with tab2:
        st.subheader("Batch Processing")
        uploaded_file = st.file_uploader("Upload Customer CSV List", type="csv")
        
        if uploaded_file:
            batch_data = pd.read_csv(uploaded_file)
            
            # Use EXACTLY the same names as the training dataset (Case-Sensitive)
            features_needed = ['tenure', 'MonthlyCharges', 'TotalCharges']
            
            try:
                clean_batch = batch_data.copy()
                # Fix the TotalCharges string issue exactly like we did in training
                clean_batch['TotalCharges'] = pd.to_numeric(clean_batch['TotalCharges'], errors='coerce').fillna(0)
                
                # Inference
                preds = model.predict(clean_batch[features_needed])
                batch_data['Churn_Prediction'] = ["Likely to Leave" if p == 1 else "Likely to Stay" for p in preds]
                
                # Calculate metrics
                total_customers = len(batch_data)
                at_risk = sum(preds)
                
                st.success("Analysis Complete!")
                
                # Display metrics
                m1, m2 = st.columns(2)
                m1.metric("Total Customers Analyzed", total_customers)
                m2.metric("Customers at Risk", at_risk, delta="-Requires Action", delta_color="inverse")
                
                # Show dataframe and download
                st.dataframe(batch_data)
                st.download_button("Download Report", batch_data.to_csv(index=False), "churn_report.csv")
                
            except KeyError as e:
                st.error(f"Missing columns! Ensure your CSV has exactly: {features_needed}")