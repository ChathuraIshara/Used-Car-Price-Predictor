"""
Builds an end-to-end inference pipeline for the Streamlit app.
Accepts raw inputs -> Returns price prediction.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
TOP_MAKES_COUNT = 15
TOP_LOCATIONS_COUNT = 20

# ─────────────────────────────────────────────
# Path Setup
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA = os.path.join(BASE_DIR, "data", "raw", "riyasewana_cars.csv")
MODEL_OUT = os.path.join(BASE_DIR, "models", "car_price_pipeline.pkl")
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

# ─────────────────────────────────────────────
# 1. Custom Preprocessing Logic (as Functions)
# ─────────────────────────────────────────────
def load_and_clean_data(filepath):
    """Loads raw data and performs basic cleaning (outliers, formatting)."""
    df = pd.read_csv(filepath)
    
    # Drop useless cols
    cols_to_drop = ["title", "listing_url", "options", "details", "date_posted", "condition"]
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    
    # Fix typos
    invalid_trans = df["transmission"].isin(["100", "2000"])
    df.loc[invalid_trans, "transmission"] = np.nan
    df.loc[df["fuel_type"] == "Kick", "fuel_type"] = np.nan
    
    for col in ["fuel_type", "transmission", "make", "model", "location"]:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()
            
    # Outliers (Year, Mileage, Price)
    current_year = 2025
    df = df[(df["year"] >= 1970) & (df["year"] <= current_year)]
    
    # Mileage cap
    Q3 = df["mileage_km"].quantile(0.75)
    IQR = Q3 - df["mileage_km"].quantile(0.25)
    upper = Q3 + 3 * IQR
    df = df[df["mileage_km"] <= upper]
    
    # Price
    df = df[df["price_lkr"] > 50000]
    df.dropna(subset=["price_lkr"], inplace=True)
    
    return df

def feature_engineering(X):
    """Adds car_age and mileage_per_year. Handles column removal."""
    X = X.copy()
    current_year = 2025
    
    # Create car_age
    X["car_age"] = current_year - X["year"]
    
    # Create mileage_per_year
    age = X["car_age"].replace(0, 1) 
    X["mileage_per_year"] = X["mileage_km"] / age
    
    # Drop columns that are definitely not used in the model
    # 'model' is high cardinality and was dropped in preprocess.py
    # 'year' is replaced by 'car_age'
    cols_to_drop = ["model", "year"]
    X.drop(columns=[c for c in cols_to_drop if c in X.columns], inplace=True)
    
    return X

class CategoricalGrouper(BaseEstimator, TransformerMixin):
    """Groups rare categories into 'Other' based on training frequency."""
    def __init__(self, top_n=10):
        self.top_n = top_n
        self.top_categories_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            self.top_categories_[col] = X[col].value_counts().nlargest(self.top_n).index.tolist()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = X[col].where(X[col].isin(self.top_categories_[col]), other="Other")
        return X

# ─────────────────────────────────────────────
# 2. Build Pipeline
# ─────────────────────────────────────────────
def build_pipeline():
    # Define features
    numeric_features = ["mileage_km", "engine_cc", "car_age", "mileage_per_year"]
    categorical_features = ["make", "fuel_type", "transmission", "location"]
    
    # Numeric Transformer: Impute median -> Scale
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ])
    
    # Categorical Transformer: Group -> Impute -> OneHot
    categorical_transformer = Pipeline(steps=[
        ("grouper", CategoricalGrouper(top_n=TOP_MAKES_COUNT)), # Placeholder top_n, refined per column below if needed
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # Specific Column Transformer to handle different Top-N values
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("make", Pipeline([
                ("group", CategoricalGrouper(top_n=TOP_MAKES_COUNT)),
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), ["make"]),
            ("loc", Pipeline([
                ("group", CategoricalGrouper(top_n=TOP_LOCATIONS_COUNT)),
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), ["location"]),
            ("cat_general", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), ["fuel_type", "transmission"]),
        ],
        remainder="drop"
    )
    
    # Full Pipeline
    # 1. Feature Engineering
    # 2. Preprocessing
    # 3. Model
    pipeline = Pipeline(steps=[
        ("feat_eng", FunctionTransformer(feature_engineering, validate=False)),
        ("preprocessor", preprocessor),
        ("regressor", xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=500,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.6,
            min_child_weight=7,
            gamma=0.2,
            reg_lambda=5,
            n_jobs=-1,
            random_state=42
        ))
    ])
    
    return pipeline

# ─────────────────────────────────────────────
# 3. Main Execution
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading raw data...")
    df = load_and_clean_data(RAW_DATA)
    print(f"Data shape after cleaning: {df.shape}")
    
    X = df.drop(columns=["price_lkr"])
    y = df["price_lkr"]
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    print("Building pipeline...")
    model_pipeline = build_pipeline()
    
    print("Training pipeline...")
    model_pipeline.fit(X_train, y_train)
    
    print("Evaluating...")
    score = model_pipeline.score(X_test, y_test)
    print(f"Test R2 Score: {score:.4f}")
    
    print(f"Saving pipeline to {MODEL_OUT}...")
    joblib.dump(model_pipeline, MODEL_OUT)
    print("Done.")
