import streamlit as st
import joblib
import numpy as np
import pandas as pd
pip install joblib

# -------------------------------
# Load trained models
# -------------------------------
linreg = joblib.load("linear_regression.pkl")
ridge = joblib.load("ridge_regression.pkl")
lasso = joblib.load("lasso_regression.pkl")
DTree = joblib.load("decision_tree.pkl")
regressor = joblib.load("random_forest.pkl")

models = {
    "Linear Regression": linreg,
    "Ridge Regression": ridge,
    "Lasso Regression": lasso,
    "Decision Tree": DTree,
    "Random Forest": regressor
}

# -------------------------------
# Streamlit App UI
# -------------------------------
st.title("🏠 Airbnb Price Prediction")
st.write("Upload feature values to predict the Airbnb price using different models.")

# Model selection
model_choice = st.selectbox("Choose a model:", list(models.keys()))

# -------------------------------
# Upload CSV for Prediction
# -------------------------------
st.subheader("Upload CSV file (must contain the same features as training data, without `price`)")

uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)

    st.write("📊 Preview of uploaded data:")
    st.dataframe(input_data.head())

    if st.button("Predict Prices"):
        model = models[model_choice]
        predictions = model.predict(input_data)
        input_data["Predicted Price"] = predictions
        st.success("✅ Prediction Complete!")
        st.dataframe(input_data.head())

        # Option to download predictions
        csv = input_data.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
