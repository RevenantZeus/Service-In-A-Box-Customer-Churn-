# Customer Churn "Service-In-A-Box" 🛡️

A self-bootstrapping MLOps application that predicts customer churn using Machine Learning.

## 🚀 Overview
This project provides a complete end-to-end service for predicting customer retention risk. It includes an automated training pipeline and an interactive web dashboard.

## 🛠️ Tech Stack
- **Language:** Python 3.9+
- **AI Model:** Hyperparameter-Optimized Random Forest Classifier
- **UI Framework:** Streamlit
- **Data Handling:** Pandas & NumPy
- **Model Architecture & Specifications**
Algorithm: Random Forest Classifier (Ensemble Learning)
Methodology: Bagging (Bootstrap Aggregating) to reduce variance and prevent overfitting.
- **Optimization Technique:** Automated Hyperparameter Tuning via GridSearchCV with 5-Fold Cross-Validation.
- **Key Hyperparameters Tuned:**
n_estimators: Optimized number of trees in the forest (e.g., 100 or 150).
max_depth: Controlled tree depth to ensure model Regularization.
min_samples_split: Minimum samples required to split an internal node, enhancing stability.
- **Feature Scaling & Engineering:** *Robust handling of numerical features (tenure, MonthlyCharges, TotalCharges).
Automated data cleaning pipeline for handling non-numeric "whitespace" strings in financial columns 

## 📦 Installation & Setup
Follow these steps to run the project locally:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/churn-service-in-a-box.git](https://github.com/YOUR_USERNAME/churn-service-in-a-box.git)
   cd churn-service-in-a-box
