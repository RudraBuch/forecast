import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta

# Load your trained model
with open("C:\Users\rudra\Desktop\Intel Backup\model.pkl", 'rb') as f:
    model = pickle.load(f)

# Load your dataset to get options for filters
data = pd.read_csv("C:\Users\rudra\Desktop\Intel Backup\dairy_dataset.csv")

st.set_page_config(page_title="Sales Prediction App", layout="wide")

st.title("Daily and Total Sales Prediction")

# Sidebar filters
st.sidebar.header("Filter Your Input")

# 1. Date Range Selector (7 days max.)
start_date = st.sidebar.date_input("Start Date", datetime.today())
end_date = st.sidebar.date_input("End Date", datetime.today())

# Date validation
if (end_date - start_date).days < 0:
    st.sidebar.error("End date cannot be before start date!")
elif (end_date - start_date).days > 6:
    st.sidebar.error("Maximum allowed range is 7 days!")
else:
    date_range = pd.date_range(start=start_date, end=end_date)

# 2. Brand Selector
brand_options = data['Brand'].unique()
brand = st.sidebar.selectbox("Select Brand", brand_options)

# 3. Products Multi-selector
product_options = data['Product'].unique()
products = st.sidebar.multiselect("Select Products (or select all)", product_options, default=list(product_options))

# 4. Customer Location
location_options = data['Customer_Location'].unique()
customer_location = st.sidebar.selectbox("Select Customer Location", location_options)

# 5. Sales Type
sales_type_options = ['Wholesale', 'Retail', 'Online']
sales_type = st.sidebar.selectbox("Sales Type", sales_type_options)

if st.sidebar.button("Predict Sales"):

    if (end_date - start_date).days < 0 or (end_date - start_date).days > 6:
        st.warning("Please select a valid date range (max 7 days).")
    else:
        st.subheader(f"Sales Prediction from {start_date.strftime('%d-%m-%Y')} to {end_date.strftime('%d-%m-%Y')}")

        predictions = []

        for single_date in date_range:
            # Prepare input features (example with date features)
            date_day = single_date.day
            date_month = single_date.month
            date_weekday = single_date.weekday()  # 0 = Monday

            # Create DataFrame input row
            input_df = pd.DataFrame({
                'Day': [date_day],
                'Month': [date_month],
                'Weekday': [date_weekday],
                'Brand': [brand],
                'Customer_Location': [customer_location],
                'Sales_Type': [sales_type],
                'Product_Count': [len(products)],
                # Add other necessary features here if required
            })

            # Handle encoding if necessary (for example, using get_dummies or label encoding)
            # Ensure same preprocessing as training
            # If your model pipeline handles it internally, skip this.

            # Predict sales
            day_prediction = model.predict(input_df)[0]
            predictions.append(day_prediction)

            st.write(f"Date: {single_date.strftime('%d-%m-%Y')} | Predicted Sales: {day_prediction:.2f}")

        total_sales = np.sum(predictions)
        st.success(f"Total Predicted Sales from {start_date.strftime('%d-%m-%Y')} to {end_date.strftime('%d-%m-%Y')} = {total_sales:.2f}")

