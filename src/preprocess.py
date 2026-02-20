"""
=============================================================
  Preprocessing Pipeline – Sri Lanka Used-Car Price Dataset
  Source: Riyasewana.com (scraped via scraper.py)
  Target variable: price_lkr (car listing price in LKR)
=============================================================

Steps performed
---------------
1.  Load raw CSV
2.  Drop irrelevant / non-feature columns
3.  Drop fully-null columns (condition)
4.  Fix data-entry errors in categorical columns
5.  Filter outliers / invalid values
6.  Handle missing values
7.  Feature engineering (car_age, mileage_per_year)
8.  Label encoding for ordinal features
9.  One-hot encoding for nominal features
10. Min-Max normalisation for numeric features
11. Save preprocessed CSV
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ─────────────────────────────────────────────
# 1. Load raw data
# ─────────────────────────────────────────────
import os

# ─────────────────────────────────────────────
# 1. Load raw data
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_FILE = os.path.join(BASE_DIR, "data", "raw", "riyasewana_cars.csv")
OUT_FILE = os.path.join(BASE_DIR, "data", "processed", "riyasewana_cars_preprocessed.csv")
CURRENT_YEAR = 2025

print("=" * 60)
print("STEP 1 – Load raw dataset")
print("=" * 60)
df = pd.read_csv(RAW_FILE)
print(f"Raw shape: {df.shape}")
print(f"Columns : {df.columns.tolist()}")
print(f"\nMissing values per column:\n{df.isnull().sum()}")

# ─────────────────────────────────────────────
# 2. Drop columns not useful for ML
#    - title      : free-text, redundant with make/model/year
#    - listing_url: unique identifier, not a feature
#    - options    : free-text blob
#    - details    : free-text blob
#    - date_posted: listing date, not a car attribute
#    - condition  : 100 % missing
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 – Drop irrelevant / unusable columns")
print("=" * 60)
cols_to_drop = ["title", "listing_url", "options", "details", "date_posted", "condition"]
df.drop(columns=cols_to_drop, inplace=True)
print(f"Dropped : {cols_to_drop}")
print(f"Remaining columns: {df.columns.tolist()}")

# ─────────────────────────────────────────────
# 3. Fix data-entry errors in categoricals
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 – Fix categorical data-entry errors")
print("=" * 60)

# transmission: values '100' and '2000' are scraping artefacts → NaN
invalid_trans = df["transmission"].isin(["100", "2000"])
print(f"  transmission – invalid entries replaced with NaN: {invalid_trans.sum()}")
df.loc[invalid_trans, "transmission"] = np.nan

# fuel_type: 'Kick' is not a valid fuel type → NaN
invalid_fuel = df["fuel_type"] == "Kick"
print(f"  fuel_type    – 'Kick' entries replaced with NaN : {invalid_fuel.sum()}")
df.loc[invalid_fuel, "fuel_type"] = np.nan

# Standardise capitalisation (belt-and-braces)
for col in ["fuel_type", "transmission", "make", "model", "location"]:
    df[col] = df[col].str.strip().str.title()

# ─────────────────────────────────────────────
# 4. Filter invalid / extreme numeric values
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 – Filter outliers & invalid numeric values")
print("=" * 60)

original_len = len(df)

# year: cars made before 1970 or in the future are suspicious
invalid_year = (df["year"] < 1970) | (df["year"] > CURRENT_YEAR)
print(f"  Removing {invalid_year.sum()} rows with year outside [1970, {CURRENT_YEAR}]")
df = df[~invalid_year]

# mileage_km: IQR-based upper fence  (the max 2.5 billion km is clearly an error)
Q1 = df["mileage_km"].quantile(0.25)
Q3 = df["mileage_km"].quantile(0.75)
IQR = Q3 - Q1
mileage_upper = Q3 + 3 * IQR          # lenient 3×IQR fence
invalid_mileage = df["mileage_km"] > mileage_upper
print(f"  Removing {invalid_mileage.sum()} rows where mileage_km > {mileage_upper:,.0f} km "
      f"(3×IQR upper fence)")
df = df[~invalid_mileage]

# price_lkr: keep rows where price is present; drop obvious junk (< 50,000 LKR)
low_price = df["price_lkr"] < 50_000
print(f"  Removing {low_price.sum()} rows with price_lkr < 50,000")
df = df[~low_price | df["price_lkr"].isna()]   # NaN rows handled later

print(f"  Rows after outlier removal: {len(df)}  (removed {original_len - len(df)})")

# ─────────────────────────────────────────────
# 5. Handle missing values
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 – Handle missing values")
print("=" * 60)

# price_lkr – TARGET: drop rows where target is missing (cannot impute target)
before = len(df)
df = df.dropna(subset=["price_lkr"])
print(f"  Dropped {before - len(df)} rows where price_lkr (target) is NaN")

# mileage_km – impute with median grouped by make+fuel_type for better accuracy
df["mileage_km"] = df.groupby(["make", "fuel_type"])["mileage_km"] \
                     .transform(lambda x: x.fillna(x.median()))
# fallback: overall median
df["mileage_km"].fillna(df["mileage_km"].median(), inplace=True)
print(f"  mileage_km  – imputed with grouped median (make + fuel_type)")

# engine_cc – impute with median grouped by make+model
df["engine_cc"] = df.groupby(["make", "model"])["engine_cc"] \
                    .transform(lambda x: x.fillna(x.median()))
df["engine_cc"].fillna(df["engine_cc"].median(), inplace=True)
print(f"  engine_cc   – imputed with grouped median (make + model)")

# transmission – impute with mode per make+model
def fill_mode(s):
    m = s.mode()
    return s.fillna(m[0] if not m.empty else np.nan)

df["transmission"] = df.groupby(["make", "model"])["transmission"].transform(fill_mode)
df["transmission"].fillna(df["transmission"].mode()[0], inplace=True)
print(f"  transmission – imputed with grouped mode (make + model)")

# fuel_type – impute with mode per make
df["fuel_type"] = df.groupby("make")["fuel_type"].transform(fill_mode)
df["fuel_type"].fillna(df["fuel_type"].mode()[0], inplace=True)
print(f"  fuel_type   – imputed with grouped mode (make)")

print(f"\nMissing after imputation:\n{df.isnull().sum()}")

# ─────────────────────────────────────────────
# 6. Feature engineering
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 – Feature engineering")
print("=" * 60)

# car_age: how old the car is in years
df["car_age"] = CURRENT_YEAR - df["year"]
print(f"  Created 'car_age'  (range {df['car_age'].min()}–{df['car_age'].max()} years)")

# mileage_per_year: driving intensity proxy (avoid divide-by-zero for new cars)
df["mileage_per_year"] = df["mileage_km"] / df["car_age"].replace(0, 1)
print(f"  Created 'mileage_per_year'")

# Drop 'year' as it is now encoded by car_age
df.drop(columns=["year"], inplace=True)
print(f"  Dropped original 'year' column (encoded as car_age)")

# ─────────────────────────────────────────────
# 7. Encoding categorical variables
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 – Encoding categorical variables")
print("=" * 60)

# ── 7a. Label-encode ORDINAL features ──────────────────────────
# transmission: Manual < Automatic (ordinal sense for pricing)
le = LabelEncoder()
df["transmission_enc"] = le.fit_transform(df["transmission"])
print(f"  LabelEncoded 'transmission': mapping -> {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ── 7b. One-hot encode NOMINAL features ───────────────────────
# fuel_type, make, location
# For make and location use top-N to avoid explosion; rest → 'Other'
TOP_MAKES = 15
TOP_LOCATIONS = 20

make_counts = df["make"].value_counts()
df["make_grouped"] = df["make"].where(
    df["make"].isin(make_counts.nlargest(TOP_MAKES).index), other="Other"
)

loc_counts = df["location"].value_counts()
df["location_grouped"] = df["location"].where(
    df["location"].isin(loc_counts.nlargest(TOP_LOCATIONS).index), other="Other"
)

# One-hot encode
ohe_cols = ["fuel_type", "make_grouped", "location_grouped"]
df = pd.get_dummies(df, columns=ohe_cols, drop_first=False)
print(f"  One-hot encoded: {ohe_cols}")
print(f"  (make grouped to top {TOP_MAKES}; location grouped to top {TOP_LOCATIONS})")

# Drop original high-cardinality columns now replaced
df.drop(columns=["make", "model", "location", "transmission"], inplace=True, errors="ignore")
print(f"  Dropped: make, model, location, transmission (replaced by encoded columns)")

# ─────────────────────────────────────────────
# 8. Normalisation (Min-Max Scaling)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8 – Min-Max normalisation of numeric features")
print("=" * 60)

numeric_cols = ["mileage_km", "engine_cc", "car_age", "mileage_per_year"]
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print(f"  Scaled (0–1): {numeric_cols}")
print(f"  NOTE: price_lkr (target) is intentionally NOT scaled")

# ─────────────────────────────────────────────
# 9. Final summary & save
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9 – Final dataset summary")
print("=" * 60)
print(f"Final shape  : {df.shape}")
print(f"Columns      : {df.columns.tolist()}")
print(f"\nFirst 3 rows :\n{df.head(3)}")
print(f"\nDescriptive statistics:\n{df.describe()}")

df.to_csv(OUT_FILE, index=False)
print(f"\n[DONE]  Preprocessed dataset saved -> '{OUT_FILE}'")
print("=" * 60)
