import joblib
import pandas as pd
import os
import shap
import re
import numpy as np

# Crucial: Import custom classes so pickle can find them
from train_inference_pipeline import CategoricalGrouper, feature_engineering

# Load pipeline
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'car_price_pipeline.pkl')
pipeline = joblib.load(MODEL_PATH)
model = pipeline.named_steps['regressor']
preprocessor = pipeline.named_steps['preprocessor']

print(f"Model Parameters: {model.get_params()}")

# Create a realistic sample
data = {
    'make': ['Toyota'],
    'mileage_km': [120000],
    'engine_cc': [1500],
    'fuel_type': ['Hybrid'],
    'transmission': ['Automatic'],
    'location': ['Colombo'],
    'year': [2015]
}
df = pd.DataFrame(data)

# Transform
X_trans = pipeline[:-1].transform(df)
feature_names = preprocessor.get_feature_names_out()
X_df = pd.DataFrame(X_trans, columns=feature_names)

print(f"Prediction: {pipeline.predict(df)[0]:,.2f}")

# SHAP
print("Initializing SHAP...")
try:
    explainer = shap.TreeExplainer(model)
    shap_val = explainer(X_df)
except Exception as e:
    error_msg = str(e)
    print(f"Initial TreeExplainer failed: {error_msg}")
    
    # Robust regex for [6.43E6]
    nums = re.findall(r"[\d\.E\+]+", error_msg)
    if nums:
        try:
            fixed_score = float(nums[0])
            print(f"Setting base_score to {fixed_score}")
            model.set_params(base_score=fixed_score)
            explainer = shap.TreeExplainer(model)
            shap_val = explainer(X_df)
        except Exception as e2:
            print(f"Failed to fix base_score: {e2}")
            explainer = shap.Explainer(model.predict, X_df)
            shap_val = explainer(X_df)
    else:
        explainer = shap.Explainer(model.predict, X_df)
        shap_val = explainer(X_df)

# Handle different SHAP versions/explainers
if hasattr(explainer, 'expected_value'):
    base_val = explainer.expected_value
    if isinstance(base_val, np.ndarray): base_val = base_val[0]
else:
    base_val = shap_val.base_values[0]

print(f"Base Value: {base_val:,.2f}")
print("SHAP Values for Sample 0 (Top 10):")
vals = shap_val.values[0]
top_indices = np.argsort(np.abs(vals))[-10:][::-1]

for idx in top_indices:
    print(f"  {feature_names[idx]}: {vals[idx]:,.2f}")

total_shap = np.sum(vals)
print(f"Total SHAP: {total_shap:,.2f}")
print(f"Final Prediction: {base_val + total_shap:,.2f}")
