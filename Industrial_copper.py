import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="Industrial Copper Modeling",
    page_icon="🏭",
    layout="wide",
)

# Function to load models and preprocessors
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Load all necessary files
try:
    # Models
    reg_model = load_pickle('saved_models/regression_model.pkl')
    cls_model = load_pickle('saved_models/classification_model.pkl')

    # Scalers
    scaler_reg = load_pickle('saved_models/scaler_reg.pkl')
    scaler_cls = load_pickle('saved_models/scaler_cls.pkl')

    # Encoders
    encoders_reg = load_pickle('saved_models/encoders_reg.pkl')
    encoders_cls = load_pickle('saved_models/encoders_cls.pkl')
    status_encoder = load_pickle('saved_models/status_encoder.pkl')
    
    # Unique values for dropdowns
    unique_values = load_pickle('saved_models/unique_values.pkl')
    
except FileNotFoundError:
    st.error("Model or preprocessing files not found. Please run the `Industrial_Copper_Modeling.ipynb` notebook first to generate these files.")
    st.stop()


# UI Layout
st.title("Industrial Copper Modeling Application")
st.write("Predict selling price or lead status for the copper industry.")

# Create tabs for different predictions
tab1, tab2 = st.tabs(["**💰 Predict Selling Price**", "**📊 Predict Status**"])


# --- Tab 1: Regression (Predict Selling Price) ---
with tab1:
    st.header("Predict Selling Price")
    
    # User inputs in columns for a better layout
    col1, col2, col3 = st.columns(3)

    with col1:
        quantity_tons_reg = st.number_input("Quantity (tons)", min_value=0.1, value=10.0, step=0.1, key="reg_quantity")
        thickness_reg = st.number_input("Thickness", min_value=0.1, value=1.0, step=0.1, key="reg_thickness")
        customer_reg = st.selectbox("Customer ID", options=unique_values['customer'], key="reg_customer")

    with col2:
        country_reg = st.selectbox("Country", options=unique_values['country'], key="reg_country")
        width_reg = st.number_input("Width", min_value=1.0, value=1000.0, step=10.0, key="reg_width")
        product_ref_reg = st.text_input("Product Reference", value="611993", key="reg_product")

    with col3:
        item_type_reg = st.selectbox("Item Type", options=unique_values['item type'], key="reg_item_type")
        application_reg = st.selectbox("Application", options=unique_values['application'], key="reg_application")

    # Predict button
    if st.button("Predict Price", key="reg_predict"):
        try:
            # Create a dataframe from user inputs
            input_data_reg = pd.DataFrame([[
                quantity_tons_reg, customer_reg, country_reg, item_type_reg,
                application_reg, thickness_reg, width_reg, int(product_ref_reg)
            ]], columns=['quantity tons', 'customer', 'country', 'item type', 'application', 'thickness', 'width', 'product_ref'])

            # Preprocess the input data
            # Encode categorical features
            input_data_reg['item type'] = encoders_reg['item type'].transform(input_data_reg['item type'])

            # Scale the features
            input_scaled_reg = scaler_reg.transform(input_data_reg)

            # Make prediction
            predicted_log_price = reg_model.predict(input_scaled_reg)
            predicted_price = np.expm1(predicted_log_price)[0] # Inverse transform log

            st.success(f"**Predicted Selling Price:** ${predicted_price:,.2f}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")


# --- Tab 2: Classification (Predict Status) ---
with tab2:
    st.header("Predict Lead Status (Won/Lost)")
    
    # User inputs
    col1_cls, col2_cls, col3_cls = st.columns(3)

    with col1_cls:
        quantity_tons_cls = st.number_input("Quantity (tons)", min_value=0.1, value=10.0, step=0.1, key="cls_quantity")
        thickness_cls = st.number_input("Thickness", min_value=0.1, value=1.0, step=0.1, key="cls_thickness")
        customer_cls = st.selectbox("Customer ID", options=unique_values['customer'], key="cls_customer")

    with col2_cls:
        country_cls = st.selectbox("Country", options=unique_values['country'], key="cls_country")
        width_cls = st.number_input("Width", min_value=1.0, value=1000.0, step=10.0, key="cls_width")
        selling_price_cls = st.number_input("Selling Price", min_value=1.0, value=500.0, step=10.0, key="cls_price")

    with col3_cls:
        item_type_cls = st.selectbox("Item Type", options=unique_values['item type'], key="cls_item_type")
        application_cls = st.selectbox("Application", options=unique_values['application'], key="cls_application")
    
    # Predict button
    if st.button("Predict Status", key="cls_predict"):
        try:
            # Create dataframe
            input_data_cls = pd.DataFrame([[
                quantity_tons_cls, customer_cls, country_cls, item_type_cls,
                application_cls, thickness_cls, width_cls, selling_price_cls
            ]], columns=['quantity tons', 'customer', 'country', 'item type', 'application', 'thickness', 'width', 'selling_price'])

            # Preprocess
            input_data_cls['item type'] = encoders_cls['item type'].transform(input_data_cls['item type'])
            input_scaled_cls = scaler_cls.transform(input_data_cls)

            # Predict
            prediction = cls_model.predict(input_scaled_cls)
            status = status_encoder.inverse_transform(prediction)[0]

            if status == 'Won':
                st.success(f"**Predicted Status:** {status} 🎉")
            else:
                st.warning(f"**Predicted Status:** {status} 😔")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
