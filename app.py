import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta

# ✅ Load your trained model
with open(r"C:\Users\rudra\Desktop\Intel Backup\model.pkl", 'rb') as f:
    model = pickle.load(f)

# ✅ Load your dataset to get filter options
data = pd.read_csv(r"C:\Users\rudra\Desktop\Intel Backup\dairy_dataset.csv")

st.set_page_config(page_title="Sales Prediction App", layout="wide")

st.title("📊 Daily and Total Sales Prediction")

# ✅ Sidebar filters
st.sidebar.header("🔍 Filter Your Input")

# 1. Date Range Selector (7 days max.)
start_date = st.sidebar.date_input("Start Date", datetime.today())
end_date = st.sidebar.date_input("End Date", datetime.today())

if (end_date - start_date).days < 0:
    st.sidebar.error("End date cannot be before start date!")
    st.stop()
elif (end_date - start_date).days > 6:
    st.sidebar.error("Maximum allowed range is 7 days!")
    st.stop()

date_range = pd.date_range(start=start_date, end=end_date)

# 2. Brand Selector
brand_options = data['Brand'].unique()
brand = st.sidebar.selectbox("Select Brand", brand_options)

# 3. Product Selector with 'Select All'
product_options = data['Product'].unique()
select_all = st.sidebar.checkbox("Select All Products", value=True)

if select_all:
    products = list(product_options)
else:
    products = st.sidebar.multiselect("Select Products", product_options)

if len(products) == 0:
    st.sidebar.warning("Please select at least one product.")
    st.stop()

# 4. Customer Location
location_options = data['Customer_Location'].unique()
customer_location = st.sidebar.selectbox("Select Customer Location", location_options)

# 5. Sales Type
sales_type_options = ['Wholesale', 'Retail', 'Online']
sales_type = st.sidebar.selectbox("Sales Type", sales_type_options)

if st.sidebar.button("📈 Predict Sales"):

    st.subheader(f"Sales Prediction from {start_date.strftime('%d-%m-%Y')} to {end_date.strftime('%d-%m-%Y')}")

    predictions = []

    for single_date in date_range:
        # Date features
        date_day = single_date.day
        date_month = single_date.month
        date_weekday = single_date.weekday()

        # Create DataFrame input
        input_df = pd.DataFrame({
            'Day': [date_day],
            'Month': [date_month],
            'Weekday': [date_weekday],
            'Brand': [brand],
            'Customer_Location': [customer_location],
            'Sales_Type': [sales_type],
            'Product_Count': [len(products)],
        })

        # Predict
        day_prediction = model.predict(input_df)[0]
        predictions.append(day_prediction)

        st.write(f"📅 {single_date.strftime('%d-%m-%Y')} → Predicted Sales: **{day_prediction:.2f}**")

    total_sales = np.sum(predictions)
    st.success(f"✅ Total Predicted Sales ({len(date_range)} days): **{total_sales:.2f}**")
