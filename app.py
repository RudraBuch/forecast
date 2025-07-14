import random
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="🚀 Product Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for attractive styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .section-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }
    .section-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    .section-title {
        color: #4a5568;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    /* Gradient backgrounds for the first three cards */
    .date-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;      
      
    }
    .brand-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
    }
    .location-card{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
    }
    .products-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
    }
    .predict-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: left;
        padding: 2rem 1rem;
    }
    .predict-section .section-title {
        color: white;
        justify-content: left;
    }
    .summary-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        # padding: 1.5rem;
        # border-radius: 12px;
        # text-align: center;
    }
    .result-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-top: 2rem;
        color: white;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .success-message {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 500;
    }
    .error-message {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 500;
    }
    .warning-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 500;
    }
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    .stSelectbox > div > div > select {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        padding: 0.5rem;
    }
    .stMultiSelect > div > div > div {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
    }
    .stDateInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model, None
    except FileNotFoundError:
        return None, "model.pkl file not found. Please ensure the file is in the same directory as this script."
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def create_features(start_date, end_date, brand, products, location,variation_factor=0.2, weekend_boost=0.2, trend_factor=0.05, base_multiplier=1.0):
    """
    Create features with realistic values instead of all zeros
    """
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    features_list = []
    all_brands = ['Amul', 'Mother Dairy', 'Raj', 'Sudha']
    all_products = [
        "Butter", "Buttermilk", "Cheese", "Curd", "Ghee", "Ice Cream", "Lassi", "Milk", "Paneer", "Yogurt"
    ]
    all_locations = [
        'Bihar', 'Chandigarh', 'Delhi', 'Gujarat', 'Haryana', 'Jharkhand', 'Karnataka', 'Kerala',
        'Madhya Pradesh', 'Maharashtra', 'Rajasthan', 'Tamil Nadu', 'Telangana', 'Uttar Pradesh', 'West Bengal'
    ]
    all_channels = ['Online', 'Retail', 'Wholesale']
    
    # Define realistic value ranges for different products
    product_defaults = {
        'Milk': {
            'price_range': (45, 65),
            'shelf_life': (2, 5),
            'base_quantity_liters': (100, 500),
            'base_quantity_kg': (0, 0),
            'stock_multiplier': 1.5,
            'reorder_multiplier': 0.8
        },
        'Butter': {
            'price_range': (400, 600),
            'shelf_life': (30, 60),
            'base_quantity_liters': (0, 0),
            'base_quantity_kg': (10, 50),
            'stock_multiplier': 1.3,
            'reorder_multiplier': 0.7
        },
        'Cheese': {
            'price_range': (350, 550),
            'shelf_life': (15, 30),
            'base_quantity_liters': (0, 0),
            'base_quantity_kg': (5, 25),
            'stock_multiplier': 1.4,
            'reorder_multiplier': 0.6
        },
        'Curd': {
            'price_range': (50, 80),
            'shelf_life': (2, 4),
            'base_quantity_liters': (20, 100),
            'base_quantity_kg': (0, 0),
            'stock_multiplier': 1.2,
            'reorder_multiplier': 0.8
        },
        'Ghee': {
            'price_range': (400, 700),
            'shelf_life': (180, 365),
            'base_quantity_liters': (0, 0),
            'base_quantity_kg': (5, 30),
            'stock_multiplier': 1.8,
            'reorder_multiplier': 0.5
        },
        'Ice Cream': {
            'price_range': (150, 300),
            'shelf_life': (30, 90),
            'base_quantity_liters': (10, 50),
            'base_quantity_kg': (0, 0),
            'stock_multiplier': 1.1,
            'reorder_multiplier': 0.9
        },
        'Lassi': {
            'price_range': (25, 45),
            'shelf_life': (2, 3),
            'base_quantity_liters': (20, 80),
            'base_quantity_kg': (0, 0),
            'stock_multiplier': 1.3,
            'reorder_multiplier': 0.8
        },
        'Buttermilk': {
            'price_range': (20, 35),
            'shelf_life': (2, 4),
            'base_quantity_liters': (15, 70),
            'base_quantity_kg': (0, 0),
            'stock_multiplier': 1.2,
            'reorder_multiplier': 0.8
        },
        'Paneer': {
            'price_range': (200, 350),
            'shelf_life': (3, 7),
            'base_quantity_liters': (0, 0),
            'base_quantity_kg': (5, 25),
            'stock_multiplier': 1.4,
            'reorder_multiplier': 0.7
        },
        'Yogurt': {
            'price_range': (40, 70),
            'shelf_life': (7, 14),
            'base_quantity_liters': (10, 50),
            'base_quantity_kg': (0, 0),
            'stock_multiplier': 1.3,
            'reorder_multiplier': 0.8
        }
    }
    
    # Set random seed for reproducible results (optional)
    np.random.seed(42)
    
    for i, date in enumerate(date_range):
        for product in products:
            row = {}
            
            # Get product defaults
            defaults = product_defaults.get(product, product_defaults['Milk'])
            
            # Add some daily variation (±20% variation)
            #daily_variation = 0.8 + (0.4 * np.random.random())  # 0.8 to 1.2
            daily_variation = (1 - variation_factor) + (2 * variation_factor * np.random.random())
            # Weekend effect (higher demand on weekends)
            #weekend_multiplier = 1.2 if date.weekday() >= 5 else 1.0
            weekend_multiplier = (1 + weekend_boost) if date.weekday() >= 5 else 1.0
            # Time-based trend (slight increase over time)
            #trend_multiplier = 1.0 + (i * 0.05)  # 5% increase per day
            trend_multiplier = 1.0 + (i * trend_factor)
            # Calculate realistic values
            price_min, price_max = defaults['price_range']
            row['Price per Unit'] = np.random.uniform(price_min, price_max) * daily_variation
            
            row['Shelf Life (days)'] = np.random.randint(defaults['shelf_life'][0], defaults['shelf_life'][1] + 1)
            
            # # Quantity (liters)
            # base_liters_min, base_liters_max = defaults['base_quantity_liters']
            # if base_liters_max > 0:
            #     row['Quantity (liters)'] = np.random.uniform(base_liters_min, base_liters_max) * daily_variation * weekend_multiplier * trend_multiplier
            # else:
            #     row['Quantity (liters)'] = 0
            
            # # Quantity (kg)
            # base_kg_min, base_kg_max = defaults['base_quantity_kg']
            # if base_kg_max > 0:
            #     row['Quantity (kg)'] = np.random.uniform(base_kg_min, base_kg_max) * daily_variation * weekend_multiplier * trend_multiplier
            # else:
            #     row['Quantity (kg)'] = 0
            
            # # Stock quantities (usually higher than demand)
            # row['Quantity in Stock (liters)'] = row['Quantity (liters)'] * defaults['stock_multiplier']
            # row['Quantity in Stock (kg)'] = row['Quantity (kg)'] * defaults['stock_multiplier']
            
            # # Reorder quantities (usually lower than current stock)
            # row['Reorder Quantity (liters)'] = row['Quantity (liters)'] * defaults['reorder_multiplier']
            # row['Reorder Quantity (kg)'] = row['Quantity (kg)'] * defaults['reorder_multiplier']
            base_liters_min, base_liters_max = defaults['base_quantity_liters']
            if base_liters_max > 0:
                row['Quantity (liters)'] = np.random.uniform(base_liters_min, base_liters_max) * daily_variation * weekend_multiplier * trend_multiplier * base_multiplier
            else:
                row['Quantity (liters)'] = 0

            # Quantity (kg)
            base_kg_min, base_kg_max = defaults['base_quantity_kg']
            if base_kg_max > 0:
                row['Quantity (kg)'] = np.random.uniform(base_kg_min, base_kg_max) * daily_variation * weekend_multiplier * trend_multiplier * base_multiplier
            else:
                row['Quantity (kg)'] = 0

            # Stock quantities (usually higher than demand)
            row['Quantity in Stock (liters)'] = row['Quantity (liters)'] * defaults['stock_multiplier']
            row['Quantity in Stock (kg)'] = row['Quantity (kg)'] * defaults['stock_multiplier']

            # Reorder quantities (usually lower than current stock)
            row['Reorder Quantity (liters)'] = row['Quantity (liters)'] * defaults['reorder_multiplier']
            row['Reorder Quantity (kg)'] = row['Quantity (kg)'] * defaults['reorder_multiplier']
            
            # One-hot encoding for brands
            for b in all_brands:
                col = f'Brand_{b}'
                row[col] = 1 if brand == b else 0
            
            # One-hot encoding for products
            for p in all_products:
                col = f'Product Name_{p}'
                row[col] = 1 if product == p else 0
            
            # One-hot encoding for locations
            for loc in all_locations:
                col = f'Customer Location_{loc}'
                row[col] = 1 if loc == location else 0
            
            # One-hot encoding for sales channels (default to Online)
            for ch in all_channels:
                col = f'Sales Channel_{ch}'
                row[col] = 1 if ch == 'Online' else 0
            
            # Store metadata
            row['date'] = date
            row['product'] = product
            row['brand'] = brand
            
            features_list.append(row)
    
    # Create DataFrame
    prediction_features = [
        'Price per Unit', 'Shelf Life (days)', 'Quantity (liters)', 'Quantity (kg)',
        'Quantity in Stock (liters)', 'Quantity in Stock (kg)', 'Reorder Quantity (liters)', 'Reorder Quantity (kg)',
        'Brand_Amul', 'Brand_Mother Dairy', 'Brand_Raj', 'Brand_Sudha',
        'Product Name_Butter', 'Product Name_Buttermilk', 'Product Name_Cheese', 'Product Name_Curd',
        'Product Name_Ghee', 'Product Name_Ice Cream', 'Product Name_Lassi', 'Product Name_Milk',
        'Product Name_Paneer', 'Product Name_Yogurt',
        'Customer Location_Bihar', 'Customer Location_Chandigarh', 'Customer Location_Delhi',
        'Customer Location_Gujarat', 'Customer Location_Haryana', 'Customer Location_Jharkhand',
        'Customer Location_Karnataka', 'Customer Location_Kerala', 'Customer Location_Madhya Pradesh',
        'Customer Location_Maharashtra', 'Customer Location_Rajasthan', 'Customer Location_Tamil Nadu',
        'Customer Location_Telangana', 'Customer Location_Uttar Pradesh', 'Customer Location_West Bengal',
        'Sales Channel_Online', 'Sales Channel_Retail', 'Sales Channel_Wholesale',
        'date', 'product', 'brand'
    ]
    
    df = pd.DataFrame(features_list)
    
    # Ensure all required columns exist
    for col in prediction_features:
        if col not in df.columns:
            df[col] = 0
    
    df = df[prediction_features]
    return df
def make_predictions(model, features_df):
    try:
        prediction_features = [
            'Price per Unit', 'Shelf Life (days)', 'Quantity (liters)', 'Quantity (kg)', 
            'Quantity in Stock (liters)', 'Quantity in Stock (kg)', 'Reorder Quantity (liters)', 'Reorder Quantity (kg)',
            'Brand_Amul', 'Brand_Mother Dairy', 'Brand_Raj', 'Brand_Sudha',
            'Product Name_Butter', 'Product Name_Buttermilk', 'Product Name_Cheese', 'Product Name_Curd', 
            'Product Name_Ghee', 'Product Name_Ice Cream', 'Product Name_Lassi', 'Product Name_Milk', 
            'Product Name_Paneer', 'Product Name_Yogurt',
            'Customer Location_Bihar', 'Customer Location_Chandigarh', 'Customer Location_Delhi', 
            'Customer Location_Gujarat', 'Customer Location_Haryana', 'Customer Location_Jharkhand', 
            'Customer Location_Karnataka', 'Customer Location_Kerala', 'Customer Location_Madhya Pradesh', 
            'Customer Location_Maharashtra', 'Customer Location_Rajasthan', 'Customer Location_Tamil Nadu', 
            'Customer Location_Telangana', 'Customer Location_Uttar Pradesh', 'Customer Location_West Bengal',
            'Sales Channel_Online', 'Sales Channel_Retail', 'Sales Channel_Wholesale'
        ]
        for feature in prediction_features:
            if feature not in features_df.columns:
                st.error(f"Missing feature: {feature}")
                return None
        X = features_df[prediction_features]
        predictions = model.predict(X)
        if predictions.ndim == 2 and predictions.shape[1] == 2:
            features_df['predicted_liters'] = predictions[:, 0]
            features_df['predicted_kg'] = predictions[:, 1]
        else:
            features_df['predicted_demand'] = predictions
        return features_df
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return None

def calculate_metrics(predictions_df):
    if predictions_df is None or predictions_df.empty:
        return None
    if 'predicted_liters' in predictions_df.columns:
        col = 'predicted_liters'
    elif 'predicted_kg' in predictions_df.columns:
        col = 'predicted_kg'
    else:
        col = 'predicted_demand'
    total_demand = predictions_df[col].sum()
    avg_daily_demand = predictions_df[col].mean()
    peak_day = predictions_df.loc[predictions_df[col].idxmax(), 'date']
    peak_demand = predictions_df[col].max()
    if len(predictions_df) > 1:
        first_half = predictions_df.iloc[:len(predictions_df)//2][col].mean()
        second_half = predictions_df.iloc[len(predictions_df)//2:][col].mean()
        growth_rate = ((second_half - first_half) / first_half) * 100 if first_half != 0 else 0
    else:
        growth_rate = 0
    volatility = (predictions_df[col].std() / predictions_df[col].mean()) * 100 if predictions_df[col].mean() != 0 else 0
    return {
        'total_demand': total_demand,
        'avg_daily_demand': avg_daily_demand,
        'peak_day': peak_day,
        'peak_demand': peak_demand,
        'growth_rate': growth_rate,
        'volatility': volatility,
        'metric_col': col
    }

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Product Prediction Dashboard</h1>
        <p>Intelligent forecasting for dairy products using ML model</p>
    </div>
    """, unsafe_allow_html=True)
    model, error_msg = load_model()
    if model is None:
        st.error(f"❌ **Model Loading Error:** {error_msg}")
        st.info("📝 **Instructions:**\n1. Place your `model.pkl` file in the same directory as this script\n2. Ensure the model was saved using pickle\n3. Restart the application")
        return
    st.success("✅ **Model loaded successfully!**")
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None
    col1, col2, col3, col4, col5 = st.columns([2.5, 2, 2.5, 2, 2])
    # with col1:
    #     st.markdown('<div class="section-card">', unsafe_allow_html=True)
    #     st.markdown('<div class="section-title">📅 Date Range</div>', unsafe_allow_html=True)
    #     st.markdown(f"<h2 style='color:#667eea;'>🗓️</h2>", unsafe_allow_html=True)
    #     st.markdown("Select the period for prediction.")
    #     st.markdown('</div>', unsafe_allow_html=True)
    with col1:
        st.markdown('''
    <div class="section-card date-card">
        <div class="section-title">📅 Date Range</div>
        <h2 style='color:#fff;'>🗓️</h2>
        <div style="font-size:1rem; color:#fff; margin-bottom:0.5rem;">
            Select the period for prediction.
        </div>
    </div>
    ''', unsafe_allow_html=True)
        # st.markdown('<div class="section-card">', unsafe_allow_html=True)
        # st.markdown('<div class="section-title">📅 Select Date Range</div>', unsafe_allow_html=True)
        today = datetime.now().date()
        start_date = st.date_input(
            "From Date",
            value=today - timedelta(days=6),
            max_value=today + timedelta(days=30),
            key="start_date",
            help="Select the starting date for prediction"
        )
        end_date = st.date_input(
            "To Date",
            value=today,
            min_value=start_date if start_date else today - timedelta(days=30),
            max_value=today + timedelta(days=30),
            key="end_date",
            help="Select the ending date for prediction"
        )
        date_range_valid = True
        if start_date and end_date:
            date_diff = (end_date - start_date).days
            if date_diff > 6:
                st.markdown(f'<div class="error-message">⚠️ Maximum date range is 7 days! Current range: {date_diff + 1} days</div>', unsafe_allow_html=True)
                date_range_valid = False
            elif date_diff < 0:
                st.markdown('<div class="error-message">❌ End date must be after start date!</div>', unsafe_allow_html=True)
                date_range_valid = False
            else:
                st.markdown(f'<div class="success-message">✅ Perfect! Selected range: {date_diff + 1} days</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    # with col2:
    #     st.markdown('<div class="section-card">', unsafe_allow_html=True)
    #     st.markdown('<div class="section-title">🏢 Brand Info</div>', unsafe_allow_html=True)
    #     st.markdown(f"<h2 style='color:#764ba2;'>🏢</h2>", unsafe_allow_html=True)
    #     st.markdown("Choose your dairy brand.")
    #     st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('''
    <div class="section-card brand-card">
        <div class="section-title">🏢 Brand Info</div>
        <h2 style='color:#764ba2;'>🏢</h2>
        <div style="font-size:1rem; color:#fff; margin-bottom:0.5rem;">
            Choose your dairy brand.
        </div>
    </div>
    ''', unsafe_allow_html=True)

        #st.markdown('<div class="section-card">', unsafe_allow_html=True)
        #st.markdown('<div class="section-title">🏢 Select Brand</div>', unsafe_allow_html=True)
        brands = ["Amul", "Mother Dairy", "Raj", "Sudha"]
        selected_brand = st.selectbox(
            "Choose Brand",
            options=brands,
            index=0,
            key="brand_selection"
        )
        brand_info = {
            "Amul": "🥛 Leading cooperative brand",
            "Mother Dairy": "🏪 Premium dairy products",
            "Raj": "🌟 Regional favorite",
            "Sudha": "🥄 Traditional quality"
        }
        st.markdown(f'<div style="background: #f0f2f6; padding: 0.5rem; border-radius: 6px; margin-top: 0.5rem; font-size: 0.9rem;">{brand_info[selected_brand]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    # with col3:
    #     st.markdown('<div class="section-card">', unsafe_allow_html=True)
    #     st.markdown('<div class="section-title">🥛 Products</div>', unsafe_allow_html=True)
    #     st.markdown(f"<h2 style='color:#f093fb;'>🥛</h2>", unsafe_allow_html=True)
    #     st.markdown("Select products to forecast.")
    #     st.markdown('</div>', unsafe_allow_html=True)   
# --- Add this for Customer Location ---
    with col3:
        st.markdown('''
    <div class="section-card location-card">
        <div class="section-title">📍 Customer Location</div>
        <h2 style='color:#f093fb;'>📍</h2>
        <div style="font-size:1rem; color:#fff; margin-bottom:0.5rem;">
            Select your customer location.
        </div>
    </div>
        ''', unsafe_allow_html=True)
        locations = [
            "Bihar","Chandigarh","Delhi","Gujarat","Haryana","Jharkhand","Karnataka",
            "Kerala","Madhya Pradesh","Maharashtra","Rajasthan","Tamil Nadu",
            "Telangana","Uttar Pradesh","West Bengal"
        ]
        selected_location = st.selectbox(
            "Choose Customer Location",
            options=locations,
            key="location_selection"
        )
    st.markdown('</div>', unsafe_allow_html=True)
# ...then shift your Products code to col4, Predict to col5, and Summary below the row... 
    with col4:
        st.markdown('''
    <div class="section-card products-card">
        <div class="section-title">🥛 Products</div>
        <h2 style='color:#f093fb;'>🥛</h2>
        <div style="font-size:1rem; color:#fff; margin-bottom:0.5rem;">
            Select products to forecast.
        </div>
    </div>
    ''', unsafe_allow_html=True)
        #st.markdown('<div class="section-card">', unsafe_allow_html=True)
        #st.markdown('<div class="section-title">🥛 Select Products</div>', unsafe_allow_html=True)
        products = [
            "Cheese", "Curd", "Butter", "Ghee", "Milk", 
            "Paneer", "Lassi", "Buttermilk", "Ice Cream", "Yogurt"
        ]
        selected_products = st.multiselect(
            "Choose Products",
            options=products,
            default=["Milk"],
            key="product_selection"
        )
        if not selected_products:
            st.markdown('<div class="warning-message">⚠️ Please select at least one product</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="success-message">✅ {len(selected_products)} product(s) selected</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    # with col4:
    #     st.markdown('<div class="section-card predict-section">', unsafe_allow_html=True)
    #     st.markdown('<div class="section-title">🔮 Prediction</div>', unsafe_allow_html=True)
    #     st.markdown(f"<h2 style='color:#4facfe;'>🔮</h2>", unsafe_allow_html=True)
    #     st.markdown("Generate demand forecast.")
    #     st.markdown('</div>', unsafe_allow_html=True)
    with col5:
        st.markdown('''
    <div class="section-card predict-section">
        <div class="section-title">🔮 Prediction</div>
        <h2 style='color:#4facfe;'>🔮</h2>
        <div style="font-size:1rem; color:#fff; margin-bottom:0.5rem;">
            Generate demand forecast.
        </div>
    </div>
    ''', unsafe_allow_html=True)
        #st.markdown('<div class="section-card predict-section">', unsafe_allow_html=True)
        #st.markdown('<div class="section-title">🔮 Generate Prediction</div>', unsafe_allow_html=True)
        can_predict = (
            start_date and end_date and 
            date_range_valid and 
            selected_products and 
            selected_brand
        )
        if can_predict:
            st.markdown('✅ **Ready to predict!**')
        else:
            st.markdown('⏳ **Complete the form**')
        
        # Add this after your existing columns, before the prediction button
    st.markdown("### 🎛️ Advanced Settings")
    with st.expander("Adjust Prediction Parameters"):
        col_a, col_b = st.columns(2)
    
        with     col_a:
            variation_factor = st.slider(
                "Daily Variation (%)", 
                min_value=0, 
                max_value=50, 
                value=20,
                help="Amount of daily variation in demand"
            )
        
            weekend_boost = st.slider(
                "Weekend Boost (%)", 
                min_value=0, 
                max_value=100, 
                value=20,
                help="Increase in demand during weekends"
            )
    
        with col_b:
            trend_factor = st.slider(
                "Growth Trend (%/day)", 
                min_value=-10, 
                max_value=10, 
                value=5,
                help="Daily growth/decline trend"
            )
        
            base_multiplier = st.slider(
                "Base Demand Multiplier", 
                min_value=0.5, 
                max_value=2.0, 
                value=1.0,
                help="Overall demand level adjustment"
            )
        if st.button("🚀 Generate Prediction", type="primary", disabled=not can_predict, use_container_width=True):
            with st.spinner("🔄 Generating predictions using your trained model..."):
                try:
                    #features_df = create_features(start_date, end_date, selected_brand, selected_products, selected_location)
                    features_df = create_features(
                        start_date, end_date, selected_brand, selected_products, selected_location,
                        variation_factor=variation_factor/100,  # Convert percentage to decimal
                        weekend_boost=weekend_boost/100,
                        trend_factor=trend_factor/100,
                        base_multiplier=base_multiplier
                    )
                    predictions_df = make_predictions(model, features_df)
                    if predictions_df is not None:
                        metrics = calculate_metrics(predictions_df)
                        st.session_state.prediction_results = {
                            'predictions_df': predictions_df,
                            'metrics': metrics,
                            'params': {
                                'start_date': start_date,
                                'end_date': end_date,
                                'brand': selected_brand,
                                'location': selected_location,
                                'products': selected_products,
                                'timestamp': datetime.now()
                            }
                        }
                        st.success("✅ Predictions generated successfully!")
                    else:
                        st.error("❌ Failed to generate predictions")
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)
    # with col5:
    #     st.markdown('<div class="section-card summary-card">', unsafe_allow_html=True)
    #     st.markdown('<div class="section-title">📋 Summary</div>', unsafe_allow_html=True)
    #     st.markdown(f"<h2 style='color:#f5576c;'>📋</h2>", unsafe_allow_html=True)
    #     st.markdown("See your selections and quick stats.")
    #     st.markdown('</div>', unsafe_allow_html=True)
    # with col5:
    #     st.markdown('''
    # <div class="section-card summary-card">
    #     <div class="section-title">📋 Your Selection</div>
    #     <h2 style='color:#f5576c;'>📋</h2>
    #     <div style="font-size:1rem; color:#fff; margin-bottom:0.5rem;">
    #         See your selections and quick stats.
    #     </div>
    # </div>
    # ''', unsafe_allow_html=True)
    # st.markdown('''
    # <div class="section-card summary-card" style="margin-top: 1.5rem;">
    #     <div class="section-title">📋 Your Selection</div>
    #     <h2 style='color:#f5576c;'>📋</h2>
    #     <div style="font-size:1rem; color:#fff; margin-bottom:0.5rem;">
    #     See your selections and quick stats.
    # </div>
    # ''', unsafe_allow_html=True)
    #     #st.markdown('<div class="section-card summary-card">', unsafe_allow_html=True)
    #     #st.markdown('<div class="section-title">📋 Current Selection</div>', unsafe_allow_html=True)
    # if start_date and end_date and date_range_valid:
    #         st.markdown(f"**📅 Date Range**  \n{start_date.strftime('%d %b %Y')} → {end_date.strftime('%d %b %Y')}")
    # else:
    #         st.markdown("**📅 Date Range**  \n⚠️ Invalid range")
    # st.markdown(f"**🏢 Brand**  \n{selected_brand}")
    # st.markdown(f"**📍 Location**  \n{selected_location}")
    # if selected_products:
    #     product_text = ", ".join(selected_products[:2])
    #     if len(selected_products) > 2:
    #         product_text += f" +{len(selected_products)-2} more"
    #     st.markdown(f"**🥛 Products**  \n{product_text}")
    # else:
    #     st.markdown("**🥛 Products**  \n⚠️ None selected")
    # st.markdown('</div>', unsafe_allow_html=True)
    # if st.session_state.prediction_results:
    #     results = st.session_state.prediction_results
    #     predictions_df = results['predictions_df']
    #     metrics = results['metrics']
    #     params = results['params']
    #     metric_col = metrics.get('metric_col', 'predicted_demand')
    #     st.markdown('<div class="result-card">', unsafe_allow_html=True)
    #     st.markdown("## 🎯 Prediction Results")
    #     result_col1, result_col2, result_col3 = st.columns(3)
    st.markdown(f'''
    <div class="section-card summary-card" style="margin-top: 1.5rem;">
        <div class="section-title">📋 Your Selection</div>
        <h2 style='color:#f5576c;'>📋</h2>
        <div style="font-size:1rem; color:#fff; margin-bottom:0.5rem;">
            See your selections and quick stats.<br><br>
            <b>📅 Date Range:</b> {'%s → %s' % (start_date.strftime('%d %b %Y'), end_date.strftime('%d %b %Y')) if start_date and end_date and date_range_valid else '⚠️ Invalid range'}<br>
            <b>🏢 Brand:</b> {selected_brand}<br>
            <b>📍 Location:</b> {selected_location}<br>
            <b>🥛 Products:</b> {", ".join(selected_products[:2]) + (" +%d more" % (len(selected_products)-2) if len(selected_products) > 2 else "") if selected_products else "⚠️ None selected"}
        </div>
    </div>
    ''', unsafe_allow_html=True)
    if st.session_state.prediction_results:
        results = st.session_state.prediction_results
        predictions_df = results['predictions_df']
        metrics = results['metrics']
        params = results['params']
        metric_col = metrics.get('metric_col', 'predicted_demand')
        result_col1, result_col2, result_col3 = st.columns(3)
        with result_col1:
            st.markdown(f'''
            <div class="metric-card">
                <h3>📊 Demand Summary</h3>
                <p><strong>Total Demand:</strong> {metrics['total_demand']:.0f} units</p>
                <p><strong>Avg Daily:</strong> {metrics['avg_daily_demand']:.1f} units</p>
                <p><strong>Peak Day:</strong> {metrics['peak_day'].strftime('%d %b')}</p>
                <p><strong>Peak Demand:</strong> {metrics['peak_demand']:.0f} units</p>
            </div>
            ''', unsafe_allow_html=True)
        with result_col2:
            st.markdown(f'''
            <div class="metric-card">
                <h3>📈 Trend Analysis</h3>
                <p><strong>Growth Rate:</strong> {metrics['growth_rate']:+.1f}%</p>
                <p><strong>Volatility:</strong> {metrics['volatility']:.1f}%</p>
                <p><strong>Trend:</strong> {'📈 Growing' if metrics['growth_rate'] > 0 else '📉 Declining' if metrics['growth_rate'] < 0 else '➡️ Stable'}</p>
                <p><strong>Stability:</strong> {'High' if metrics['volatility'] < 20 else 'Medium' if metrics['volatility'] < 50 else 'Low'}</p>
            </div>
            ''', unsafe_allow_html=True)
        with result_col3:
            st.markdown(f'''
            <div class="metric-card">
                <h3>💡 Recommendations</h3>
                <p><strong>Stock Strategy:</strong> {'Increase' if metrics['growth_rate'] > 5 else 'Maintain' if metrics['growth_rate'] > -5 else 'Reduce'}</p>
                <p><strong>Peak Preparation:</strong> {metrics['peak_day'].strftime('%d %b')}</p>
                <p><strong>Risk Level:</strong> {'Low' if metrics['volatility'] < 20 else 'Medium' if metrics['volatility'] < 50 else 'High'}</p>
                <p><strong>Focus:</strong> {'Expansion' if metrics['growth_rate'] > 0 else 'Optimization'}</p>
            </div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("### 📊 Detailed Predictions")
        unique_products = predictions_df['product'].unique()
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        # Create subplots
        fig_subplots = make_subplots(
            rows=len(unique_products), 
            cols=1,
            subplot_titles=[f'Predicted Demand for {product}' for product in unique_products],
            vertical_spacing=0.1
        )
        
        
        product_colors = {
            "Milk": "#1f77b4",
            "Curd": "#ff7f0e",
            "Butter": "#2ca02c",
            "Ghee": "#d62728",
            "Paneer": "#9467bd",
            "Lassi": "#8c564b",
            "Buttermilk": "#e377c2",
            "Cheese": "#7f7f7f",
            "Ice Cream": "#bcbd22",
            "Yogurt": "#17becf"
        }

# Add traces for each product
        for i, product in enumerate(unique_products):
            product_data = predictions_df[predictions_df['product'] == product]
            fig_subplots.add_trace(
                go.Bar(
                    x=product_data['date'],
                    y=product_data[metric_col],
                    name=product,
                    marker_color=product_colors.get(product, '#667eea')
                ),
                row=i+1, col=1
            )
        
        # Update layout
        fig_subplots.update_layout(
            height=300 * len(unique_products),
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        # 
        # fig = px.bar(
        #     predictions_df, 
        #     x='date', 
        #     y=metric_col,
        #     color='product',
        #     title='Predicted Demand Over Time',
        #     #markers=True
        #     )
        # fig.update_layout(
        #     plot_bgcolor='rgba(0,0,0,0)',
        #     paper_bgcolor='rgba(0,0,0,0)',
        #     font=dict(color='white'),
        #     xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.2)'),
        #     yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.2)')
        #     )
        st.plotly_chart(fig_subplots, use_container_width=True)
        display_df = predictions_df[['date', 'product', 'brand', metric_col]].copy()
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        display_df[metric_col] = display_df[metric_col].round(2)
        st.dataframe(display_df, use_container_width=True)
        st.markdown("### 📋 Prediction Parameters")
        st.markdown(f"""
            - **Date Range:** {params['start_date'].strftime('%d %b %Y')} to {params['end_date'].strftime('%d %b %Y')}
            - **Brand:** {params['brand']}
            - **Location:** {params['location']}
            - **Products:** {', '.join(params['products'])}
            - **Model:** Loaded from model.pkl
            - **Generated:** {params['timestamp'].strftime('%d %b %Y at %I:%M %p')}
            """)

        st.markdown("### 📥 Download Report")
        from io import BytesIO
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        import textwrap

        if st.button("📄 Download PDF Report"):
            pdf_buffer = BytesIO()
            with PdfPages(pdf_buffer) as pdf:
                # Page 1 - Styled Summary
                fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 size
                ax.axis("off")
                fig.patch.set_facecolor('#f0f2f6')  # Light background

                # Wrap the product list
                product_list = ", ".join(params['products'])
                wrapped_products = textwrap.fill(product_list, width=60)

                text_lines = [
                    "📄 SALES PREDICTION REPORT",
                    "-----------------------------",
                    f"📅  Date Range         :  {params['start_date'].strftime('%d %b %Y')} → {params['end_date'].strftime('%d %b %Y')}",
                    f"🏢  Brand              :  {params['brand']}",
                    f"📍  State Name         :  {params['location']}",
                    f"🥛  Selected Products  :  {wrapped_products}",
                    "",
                    "📈 MODEL INFORMATION",
                    "-----------------------------",
                    f"Model Used         :  model.pkl",
                    f"Generated On       :  {params['timestamp'].strftime('%d %b %Y at %I:%M %p')}",
                    "",
                    "✔️ Powered by Streamlit + Machine Learning",
                ]
                full_text = "\n".join(text_lines)
                ax.text(0.05, 0.95, full_text, verticalalignment='top', fontsize=12, color='#333', fontfamily='monospace')
                pdf.savefig(fig)
                plt.close()

                # Page(s) for graphs (1 per product)
                for product in unique_products:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    data = predictions_df[predictions_df["product"] == product]
                    ax.bar(data["date"].dt.strftime("%d-%b"), data[metric_col], color="#764ba2")
                    ax.set_title(f"Demand Forecast - {product}", fontsize=14, fontweight='bold', color='#333')
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Predicted Demand")
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(True, linestyle="--", alpha=0.4)
                    pdf.savefig(fig)
                    plt.close()

            st.download_button(
                label="📥 Download Report as PDF",
                data=pdf_buffer.getvalue(),
                file_name="Sales_Prediction_Report.pdf",
                mime="application/pdf"
            )

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🚀 Built with ❤️ using Streamlit | Product Prediction Dashboard</p>
    <p>🤖 Powered by your trained ML model (model.pkl)</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
