import random
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

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
    .gradient-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .prediction-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        padding: 2rem 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    .prediction-section .section-title {
        color: white;
        justify-content: center;
    }
    .summary-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
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
    .expandable-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration constants
PREDICTION_FEATURES = [
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

BRANDS = ["Amul", "Mother Dairy", "Raj", "Sudha"]
PRODUCTS = ["Cheese", "Curd", "Butter", "Ghee", "Milk", "Paneer", "Lassi", "Buttermilk", "Ice Cream", "Yogurt"]
LOCATIONS = [
    "Bihar", "Chandigarh", "Delhi", "Gujarat", "Haryana", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Rajasthan", "Tamil Nadu",
    "Telangana", "Uttar Pradesh", "West Bengal"
]
SALES_CHANNELS = ['Online', 'Retail', 'Wholesale']

# Product configuration with realistic defaults
PRODUCT_DEFAULTS = {
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

PRODUCT_COLORS = {
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

@st.cache_resource
def load_model():
    """Load the trained model with better error handling"""
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        st.success("✅ Model loaded successfully!")
        return model, None
    except FileNotFoundError:
        return None, "model.pkl file not found. Please ensure the file is in the same directory as this script."
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def validate_inputs(start_date, end_date, selected_products, selected_brand, selected_location):
    """Validate user inputs and return validation status and messages"""
    errors = []
    warnings = []
    
    # Date validation
    if not start_date or not end_date:
        errors.append("Please select both start and end dates")
    elif start_date > end_date:
        errors.append("End date must be after start date")
    elif (end_date - start_date).days > 6:
        errors.append("Maximum date range is 7 days")
    
    # Product validation
    if not selected_products:
        errors.append("Please select at least one product")
    elif len(selected_products) > 5:
        warnings.append("Consider selecting fewer products for better performance")
    
    # Brand and location validation
    if not selected_brand:
        errors.append("Please select a brand")
    if not selected_location:
        errors.append("Please select a location")
    
    return len(errors) == 0, errors, warnings

def create_features(start_date, end_date, brand, products, location, variation_factor=0.2, weekend_boost=0.2, trend_factor=0.05, base_multiplier=1.0):
    """Create features with realistic values and better organization"""
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    features_list = []
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    for i, date in enumerate(date_range):
        for product in products:
            row = {}
            
            # Get product defaults
            defaults = PRODUCT_DEFAULTS.get(product, PRODUCT_DEFAULTS['Milk'])
            
            # Calculate multipliers
            daily_variation = (1 - variation_factor) + (2 * variation_factor * np.random.random())
            weekend_multiplier = (1 + weekend_boost) if date.weekday() >= 5 else 1.0
            trend_multiplier = 1.0 + (i * trend_factor)
            
            # Calculate realistic values
            price_min, price_max = defaults['price_range']
            row['Price per Unit'] = np.random.uniform(price_min, price_max) * daily_variation
            
            row['Shelf Life (days)'] = np.random.randint(defaults['shelf_life'][0], defaults['shelf_life'][1] + 1)
            
            # Quantity calculations
            base_liters_min, base_liters_max = defaults['base_quantity_liters']
            if base_liters_max > 0:
                row['Quantity (liters)'] = np.random.uniform(base_liters_min, base_liters_max) * daily_variation * weekend_multiplier * trend_multiplier * base_multiplier
            else:
                row['Quantity (liters)'] = 0

            base_kg_min, base_kg_max = defaults['base_quantity_kg']
            if base_kg_max > 0:
                row['Quantity (kg)'] = np.random.uniform(base_kg_min, base_kg_max) * daily_variation * weekend_multiplier * trend_multiplier * base_multiplier
            else:
                row['Quantity (kg)'] = 0

            # Stock quantities
            row['Quantity in Stock (liters)'] = row['Quantity (liters)'] * defaults['stock_multiplier']
            row['Quantity in Stock (kg)'] = row['Quantity (kg)'] * defaults['stock_multiplier']
            
            # Reorder quantities
            row['Reorder Quantity (liters)'] = row['Quantity (liters)'] * defaults['reorder_multiplier']
            row['Reorder Quantity (kg)'] = row['Quantity (kg)'] * defaults['reorder_multiplier']
            
            # One-hot encoding
            for b in BRANDS:
                row[f'Brand_{b}'] = 1 if brand == b else 0
            
            for p in PRODUCTS:
                row[f'Product Name_{p}'] = 1 if product == p else 0
            
            for loc in LOCATIONS:
                row[f'Customer Location_{loc}'] = 1 if loc == location else 0
            
            for ch in SALES_CHANNELS:
                row[f'Sales Channel_{ch}'] = 1 if ch == 'Online' else 0
            
            # Store metadata
            row['date'] = date
            row['product'] = product
            row['brand'] = brand
            
            features_list.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(features_list)
    
    # Ensure all required columns exist
    all_columns = PREDICTION_FEATURES + ['date', 'product', 'brand']
    for col in all_columns:
        if col not in df.columns:
            df[col] = 0
    
    return df[all_columns]

def make_predictions(model, features_df):
    """Make predictions with better error handling"""
    try:
        X = features_df[PREDICTION_FEATURES]
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
    """Calculate comprehensive metrics from predictions"""
    if predictions_df is None or predictions_df.empty:
        return None
    
    # Determine the prediction column
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
    
    # Calculate growth rate
    if len(predictions_df) > 1:
        first_half = predictions_df.iloc[:len(predictions_df)//2][col].mean()
        second_half = predictions_df.iloc[len(predictions_df)//2:][col].mean()
        growth_rate = ((second_half - first_half) / first_half) * 100 if first_half != 0 else 0
    else:
        growth_rate = 0
    
    # Calculate volatility
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

def create_visualization(predictions_df, metric_col):
    """Create enhanced visualization with better formatting"""
    unique_products = predictions_df['product'].unique()
    
    if len(unique_products) == 1:
        # Single product - line chart
        fig = px.line(
            predictions_df,
            x='date',
            y=metric_col,
            title=f'Predicted Demand for {unique_products[0]}',
            markers=True,
            color_discrete_sequence=[PRODUCT_COLORS.get(unique_products[0], '#667eea')]
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.2)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.2)')
        )
        return fig
    else:
        # Multiple products - subplots
        fig = make_subplots(
            rows=len(unique_products),
            cols=1,
            subplot_titles=[f'Predicted Demand for {product}' for product in unique_products],
            vertical_spacing=0.08
        )
        
        for i, product in enumerate(unique_products):
            product_data = predictions_df[predictions_df['product'] == product]
            fig.add_trace(
                go.Scatter(
                    x=product_data['date'],
                    y=product_data[metric_col],
                    mode='lines+markers',
                    name=product,
                    line=dict(color=PRODUCT_COLORS.get(product, '#667eea'), width=3),
                    marker=dict(size=8)
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(
            height=300 * len(unique_products),
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        return fig

'''
def generate_pdf_report(predictions_df, metrics, params):
    """Generate a comprehensive PDF report"""
    from io import BytesIO
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import textwrap
    
    pdf_buffer = BytesIO()
    metric_col = metrics['metric_col']
    
    with PdfPages(pdf_buffer) as pdf:
        # Page 1 - Executive Summary
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 size
        ax.axis("off")
        fig.patch.set_facecolor('#f8f9fa')
        
        # Header
        ax.text(0.5, 0.95, '📊 SALES PREDICTION REPORT', 
                fontsize=20, fontweight='bold', ha='center', va='top', color='#2c3e50')
        
        # Separator line
        ax.axhline(y=0.91, xmin=0.1, xmax=0.9, color='#667eea', linewidth=2)
        
        # Report details
        report_text = f"""
📋 REPORT SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📅 Date Range: {params['start_date'].strftime('%d %B %Y')} → {params['end_date'].strftime('%d %B %Y')}
🏢 Brand: {params['brand']}
📍 Location: {params['location']}
🥛 Products: {', '.join(params['products'])}

📈 KEY METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Total Predicted Demand: {metrics['total_demand']:.0f} units
• Average Daily Demand: {metrics['avg_daily_demand']:.1f} units
• Peak Demand Day: {metrics['peak_day'].strftime('%d %B %Y')}
• Peak Demand Value: {metrics['peak_demand']:.0f} units
• Growth Rate: {metrics['growth_rate']:+.1f}%
• Volatility Index: {metrics['volatility']:.1f}%

💡 INSIGHTS & RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Market Trend: {'📈 Growing Market' if metrics['growth_rate'] > 0 else '📉 Declining Market' if metrics['growth_rate'] < 0 else '➡️ Stable Market'}
• Demand Stability: {'High Stability' if metrics['volatility'] < 20 else 'Medium Stability' if metrics['volatility'] < 50 else 'Low Stability'}
• Inventory Strategy: {'Increase stock levels' if metrics['growth_rate'] > 5 else 'Maintain current levels' if metrics['growth_rate'] > -5 else 'Consider stock reduction'}
• Risk Assessment: {'Low Risk' if metrics['volatility'] < 20 else 'Medium Risk' if metrics['volatility
'''
