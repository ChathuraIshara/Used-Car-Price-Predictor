
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt


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
    .stButton>button:hover {
        background-color: #d63031 !important;
        color: white !important;
        border-color: #d63031 !important;
    }
    .stButton>button:focus,
    .stButton>button:active {
        background-color: #c0392b !important;
        color: white !important;
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
                
            # ─────────────────────────────────────────────
            # 9. SHAP Explainability Graph
            # ─────────────────────────────────────────────
            st.markdown("---")
            with st.expander("🔍 **Explain This Prediction (How features affected price)**"):
                import shap
                import re
                import matplotlib.pyplot as plt
                
                # 1. Prepare data for model
                feat_eng = pipeline.named_steps['feat_eng']
                preprocessor = pipeline.named_steps['preprocessor']
                model = pipeline.named_steps['regressor']
                
                # We need a background sample for robust explanation (sampled from cached options)
                @st.cache_data
                def get_background_data():
                    sample_size = 50
                    bg_df = df_options.sample(sample_size, random_state=42)
                    bg_eng = feat_eng.transform(bg_df.drop(columns=["price_lkr"], errors="ignore"))
                    bg_trans = preprocessor.transform(bg_eng)
                    return bg_trans

                try:
                    X_bg = get_background_data()
                    X_eng = feat_eng.transform(input_data)
                    X_trans = preprocessor.transform(X_eng)
                    feature_names = [name.split("__")[-1] for name in preprocessor.get_feature_names_out()]
                    X_df_single = pd.DataFrame(X_trans, columns=feature_names)
                    
                    # 2. Robust Explainer Initialization
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer(X_df_single)
                    except Exception as e:
                        error_msg = str(e)
                        # Extract the base_score from error like "[6.16E6]"
                        nums = re.findall(r"[\d\.E\+]+", error_msg)
                        if nums:
                            try:
                                fixed_score = float(nums[0])
                                model.set_params(base_score=fixed_score)
                                explainer = shap.TreeExplainer(model)
                                shap_values = explainer(X_df_single)
                            except:
                                explainer = shap.Explainer(model.predict, X_bg)
                                shap_values = explainer(X_df_single)
                        else:
                            explainer = shap.Explainer(model.predict, X_bg)
                            shap_values = explainer(X_df_single)

                    if shap_values is not None:
                        st.write("This chart shows how features pushed the price up or down (in **LKR**) from the average.")
                        
                        # Clean feature names
                        clean_names = []
                        for name in feature_names:
                            clean = name.split("__")[-1]
                            clean = clean.replace("location_", "").replace("make_", "").replace("fuel_type_", "").replace("transmission_", "")
                            clean_names.append(clean)
                        
                        # Get raw display values
                        X_display = X_eng.copy()
                        raw_vals = {}
                        for i, feat in enumerate(feature_names):
                            if feat in X_display.columns:
                                try:
                                    raw_vals[clean_names[i]] = int(X_display[feat].iloc[0])
                                except:
                                    raw_vals[clean_names[i]] = X_display[feat].iloc[0]
                            else:
                                raw_vals[clean_names[i]] = int(X_df_single.iloc[0, i])
                        
                        # Extract top features by absolute impact
                        sv = shap_values[0]
                        impacts = sv.values.flatten()
                        top_n = 10
                        top_idx = np.argsort(np.abs(impacts))[-top_n:][::-1]
                        
                        top_names = [clean_names[i] for i in top_idx]
                        top_impacts = [impacts[i] for i in top_idx]
                        top_raw = [raw_vals.get(clean_names[i], "") for i in top_idx]
                        
                        # Calculate "other features" combined impact
                        all_idx = set(range(len(impacts)))
                        other_idx = all_idx - set(top_idx)
                        other_impact = sum(impacts[i] for i in other_idx)
                        other_count = len(other_idx)
                        
                        # Append "other features" as the last entry
                        top_names.append(f"{other_count} other features")
                        top_impacts.append(other_impact)
                        top_raw.append("")
                        
                        # Build labels: "value = feature_name"
                        y_labels = []
                        for raw, name in zip(top_raw, top_names):
                            if raw != "":
                                y_labels.append(f"{raw} = {name}")
                            else:
                                y_labels.append(name)
                        
                        # Custom horizontal bar chart (full control over labels)
                        fig, ax = plt.subplots(figsize=(12, 8))
                        colors = ['#ff0051' if v > 0 else '#1e88e5' for v in top_impacts]
                        
                        # Reverse so biggest impact is on top
                        y_pos = range(len(top_impacts))
                        bars = ax.barh(
                            y_pos,
                            list(reversed(top_impacts)),
                            color=list(reversed(colors)),
                            height=0.6,
                            edgecolor='white',
                            linewidth=0.5
                        )
                        ax.set_yticks(list(y_pos))
                        ax.set_yticklabels(list(reversed(y_labels)), fontsize=11)
                        
                        # Add value labels on the RIGHT side of each bar (never on the left)
                        for bar, val in zip(bars, reversed(top_impacts)):
                            width = bar.get_width()
                            label = f"{'+' if val > 0 else ''}{int(val):,}"
                            # Always place label at the end of the bar, on the outside
                            if width >= 0:
                                ax.text(width + abs(ax.get_xlim()[1]) * 0.01, bar.get_y() + bar.get_height()/2,
                                        label, ha='left', va='center', fontsize=10, fontweight='bold',
                                        color='#ff0051')
                            else:
                                # BLUE bars: place label on the RIGHT side (positive x direction)
                                ax.text(abs(ax.get_xlim()[1]) * 0.01, bar.get_y() + bar.get_height()/2,
                                        label, ha='left', va='center', fontsize=10, fontweight='bold',
                                        color='#1e88e5')
                        
                        # Base value annotation
                        # Calculate E[f(X)] and f(x)
                        base_val = sv.base_values if np.isscalar(sv.base_values) else sv.base_values[0]
                        fx_val = base_val + impacts.sum()  # f(x) = base + sum of all SHAP values
                        
                        ax.set_xlabel(
                            f"SHAP Impact on Price (LKR)\n"
                            f"E[f(X)] = {int(base_val):,} LKR  →  f(x) = {int(fx_val):,} LKR",
                            fontsize=11
                        )
                        ax.set_title("Price Impact Analysis (Actual LKR)", fontsize=14, pad=15)
                        ax.axvline(x=0, color='grey', linewidth=0.8, linestyle='--')
                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)
                        plt.tight_layout()
                        st.pyplot(fig)
                        st.caption(
                            f"E[f(X)] = {int(base_val):,} LKR (avg price)  →  "
                            f"f(x) = {int(fx_val):,} LKR (this prediction). "
                            "🔴 Increases price | 🔵 Decreases price."
                        )
                except Exception as shap_err:
                    st.warning(f"Could not generate explanation: {shap_err}")

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
