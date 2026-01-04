import streamlit as st
import pickle
import numpy as np

# Load model & scaler
model = pickle.load(open("chocolate_sales_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🍫 Chocolate Sales Prediction App")

st.write("Predict whether a chocolate sale will be **High Sale** or **Low Sale**")

# User inputs
sales_person = st.number_input("Sales Person (Encoded)", min_value=0)
country = st.number_input("Country (Encoded)", min_value=0)
product = st.number_input("Product (Encoded)", min_value=0)
amount = st.number_input("Amount", min_value=0.0)
boxes = st.number_input("Boxes Shipped", min_value=0)

if st.button("Predict"):
    data = np.array([[sales_person, country, product, amount, boxes]])
    data_scaled = scaler.transform(data)
    result = model.predict(data_scaled)

    if result[0] == 1:
        st.success("✅ High Sale")
    else:
        st.warning("❌ Low Sale")
