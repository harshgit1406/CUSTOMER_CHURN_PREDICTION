import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import os

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Churn Prediction",
    layout="wide",
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364);
    background-size: 400% 400%;
    animation: gradientBG 10s ease infinite;
    color: white;
}

/* Animation */
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Cards */
.block-container {
    padding-top: 1rem;
}

div.stButton > button {
    background-color: #00c6ff;
    color: black;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
columns = pickle.load(open(os.path.join(BASE_DIR, "columns.pkl"), "rb"))
# -------------------- TITLE --------------------
st.title("🔮 Customer Churn Prediction Dashboard")

# -------------------- LAYOUT --------------------
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure", 0, 72, 12)

with col2:
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    InternetService = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaymentMethod = st.selectbox("Payment", [
        "Electronic check", "Mailed check", "Bank transfer", "Credit card"
    ])
    MonthlyCharges = st.slider("Monthly Charges", 0, 200, 50)

with col3:
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    TotalCharges = st.slider("Total Charges", 0, 10000, 1000)

# -------------------- PREPROCESS --------------------
input_data = pd.DataFrame([{
    'gender': gender,
    'SeniorCitizen': SeniorCitizen,
    'Partner': Partner,
    'Dependents': Dependents,
    'tenure': tenure,
    'PhoneService': PhoneService,
    'InternetService': InternetService,
    'OnlineSecurity': OnlineSecurity,
    'TechSupport': TechSupport,
    'StreamingTV': StreamingTV,
    'Contract': Contract,
    'PaperlessBilling': PaperlessBilling,
    'PaymentMethod': PaymentMethod,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges
}])

input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=model_columns, fill_value=0)
input_scaled = scaler.transform(input_data)

# -------------------- PREDICT --------------------
if st.button("🚀 Predict Churn"):
    prob = model.predict(input_scaled)[0][0]
    prediction = int(prob > 0.5)

    # RESULT
    if prediction == 1:
        st.error(f"⚠️ High Risk of Churn ({prob:.2%})")
    else:
        st.success(f"✅ Low Risk of Churn ({prob:.2%})")

    # -------------------- GAUGE CHART --------------------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Churn Probability"},
        gauge={
            'axis': {'range': [0, 100]},
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # -------------------- BAR CHART --------------------
    chart = pd.DataFrame({
        'Category': ['Churn', 'Not Churn'],
        'Probability': [prob, 1 - prob]
    })

    st.bar_chart(chart.set_index('Category'))

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
