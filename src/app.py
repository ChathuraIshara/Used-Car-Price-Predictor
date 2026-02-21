
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# Import the custom functions/classes needed for the pipeline
# (Must be imported so joblib can find them)
from train_inference_pipeline import load_and_clean_data, feature_engineering, CategoricalGrouper

# ─────────────────────────────────────────────
# Page Config & Styling
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Car Price Predictor | Sri Lanka",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS for "Beautiful" UI
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        height: 3em;
        border-radius: 10px;
    }
    .price-card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        box_shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border: 2px solid #e0e0e0;
    }
    .price-title {
        color: #555;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    .price-value {
        color: #2c3e50;
        font-size: 3rem;
        font-weight: 800;
    }
    .metric-row {
        display: flex;
        justify-content: space-around;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Data & Model Loading
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "car_price_pipeline.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "riyasewana_cars.csv")

@st.cache_data
def get_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def get_dropdown_options():
    # Load raw data only to get unique lists of makes/models
    df = pd.read_csv(DATA_PATH)
    # Standardise
    for col in ["make", "model", "location", "fuel_type", "transmission"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    
    # Filter out known scraping artifacts and invalid entries
    if "transmission" in df.columns:
        df = df[df["transmission"].isin(["Manual", "Automatic"])]
    
    if "fuel_type" in df.columns:
        valid_fuels = ["Petrol", "Diesel", "Hybrid", "Electric"]
        df = df[df["fuel_type"].isin(valid_fuels)]
        
    return df

try:
    pipeline = get_model()
    df_options = get_dropdown_options()
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# ─────────────────────────────────────────────
# Sidebar: Inputs
# ─────────────────────────────────────────────
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3202/3202926.png", width=100)
st.sidebar.title("🚗 Car Details")
st.sidebar.write("Enter vehicle specifications to get an estimated market price.")

# 1. Make
# Extract top makes from pipeline to ensure sync
try:
    # Try to find the 'make' transformer
    if 'make' in pipeline.named_steps['preprocessor'].named_transformers_:
        make_transformer = pipeline.named_steps['preprocessor'].named_transformers_['make']
        # If it's a Pipeline, we need to go deeper
        if hasattr(make_transformer, 'named_steps'):
            grouper = make_transformer.named_steps['group']
        else:
            grouper = make_transformer
        
        valid_makes = grouper.top_categories_.get('make')
        if valid_makes is None:
            valid_makes = list(grouper.top_categories_.values())[0]
            
        makes = sorted(valid_makes) + ["Other"]
    else:
        st.sidebar.error("⚠️ Model mismatch! Please Clear Cache in Streamlit menu (Top Right -> Settings -> Clear Cache) or restart the app.")
        makes = sorted(df_options["make"].unique())
except Exception as e:
    st.sidebar.warning(f"Note: Could not sync 'Make' list with model. ({e})")
    makes = sorted(df_options["make"].unique())

selected_make = st.sidebar.selectbox("Make", makes, index=makes.index("Toyota") if "Toyota" in makes else 0)

# 3. Year
min_year = int(df_options["year"].min())
max_year = 2025
selected_year = st.sidebar.slider("Year of Manufacture", 1980, max_year, 2018)

# 4. Mileage
selected_mileage = st.sidebar.number_input("Mileage (km)", min_value=0, max_value=500000, value=50000, step=1000)

# 5. Engine Capacity
selected_engine = st.sidebar.number_input("Engine Capacity (cc)", min_value=600, max_value=6000, value=1500, step=100)

# 6. Fuel Type
fuel_types = sorted(df_options["fuel_type"].unique())
selected_fuel = st.sidebar.selectbox("Fuel Type", fuel_types, index=0)

# 7. Transmission
trans_types = sorted(df_options["transmission"].unique())
selected_trans = st.sidebar.radio("Transmission", trans_types, horizontal=True)

# 7. Location
try:
    if 'loc' in pipeline.named_steps['preprocessor'].named_transformers_:
        loc_transformer = pipeline.named_steps['preprocessor'].named_transformers_['loc']
        if hasattr(loc_transformer, 'named_steps'):
            grouper = loc_transformer.named_steps['group']
        else:
            grouper = loc_transformer
            
        valid_locs = grouper.top_categories_.get('location')
        if valid_locs is None:
            valid_locs = list(grouper.top_categories_.values())[0]
            
        locations = sorted(valid_locs) + ["Other"]
    else:
        locations = sorted(df_options["location"].unique())
except Exception as e:
    st.sidebar.warning(f"Note: Could not sync 'Location' list with model. ({e})")
    locations = sorted(df_options["location"].unique())

selected_location = st.sidebar.selectbox("Location", locations, index=locations.index("Colombo") if "Colombo" in locations else 0)

# ─────────────────────────────────────────────
# Main Content: Prediction
# ─────────────────────────────────────────────
st.title("Sri Lanka Used Car Price Predictor 🇱🇰")
st.markdown("---")

if st.sidebar.button("Predict Price"):
    # Construct DataFrame for pipeline
    input_data = pd.DataFrame([{
        "make": selected_make,
        "year": selected_year,
        "mileage_km": selected_mileage,
        "engine_cc": selected_engine,
        "fuel_type": selected_fuel,
        "transmission": selected_trans,
        "location": selected_location
    }])
    
    with st.spinner("Calculating valuation..."):
        try:
            prediction = pipeline.predict(input_data)[0]
            
            # Display Result
            col1, col2 = st.columns([1, 1.5])
            
            with col1:
                st.markdown(f"""
                <div class="price-card">
                    <div class="price-title">Estimated Market Price</div>
                    <div class="price-value">LKR {prediction:,.0f}</div>
                    <div style="margin-top:10px; color:#888;">
                        ± 10% estimation range
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                # Comparison logic
                st.subheader("Vehicle Summary")
                st.write(f"**{selected_year} {selected_make}**")
                
                # Simple visual metric (Mileage vs Average)
                avg_mileage = df_options[(df_options["make"]==selected_make) & (df_options["year"]==selected_year)]["mileage_km"].median()
                if pd.isna(avg_mileage): avg_mileage = selected_mileage
                
                delta_mileage = selected_mileage - avg_mileage
                
                st.metric(
                    label="Mileage vs Market Average", 
                    value=f"{selected_mileage:,.0f} km",
                    delta=f"{delta_mileage:,.0f} km" if abs(delta_mileage) > 1000 else "Average",
                    delta_color="inverse"
                )
                
                st.info("This prediction is based on over 4,000 listings from Riyasewana.com.")
                
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.error("Please check if the input values are reasonable.")

else:
    st.info("👈 Adjust the vehicle details in the sidebar to get a price estimate.")
    
    # Show some market insights on landing
    st.subheader("Market Insights")
    st.write("Average price trends for popular brands:")
    
    # Simple bar chart of top brands
    top_brands = df_options["make"].value_counts().head(10).index
    avg_prices = df_options[df_options["make"].isin(top_brands)].groupby("make")["price_lkr"].mean().sort_values(ascending=False)
    
    st.bar_chart(avg_prices)
