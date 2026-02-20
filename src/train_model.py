"""
=============================================================
  Model Training & Evaluation
  Algorithm: XGBoost Regressor (Gradient Boosting)
  Problem  : Used-Car Price Prediction (Sri Lanka)
  Dataset  : riyasewana_cars_preprocessed.csv
=============================================================

Steps
-----
1.  Load preprocessed data
2.  Train / Validation / Test split  (70 / 15 / 15)
3.  Baseline XGBoost with default params
4.  Hyperparameter tuning via RandomizedSearchCV (on train+val)
5.  Final evaluation on held-out test set
6.  Feature importance
7.  Save trained model
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. Load preprocessed data
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1 – Load preprocessed dataset")
print("=" * 60)

import os

# ─────────────────────────────────────────────
# 1. Load preprocessed data
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1 – Load preprocessed dataset")
print("=" * 60)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "processed", "riyasewana_cars_preprocessed.csv")
MODEL_FILE = os.path.join(BASE_DIR, "models", "xgboost_car_price_model.pkl")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)  # Ensure output directory exists

df = pd.read_csv(DATA_FILE)
print(f"Shape: {df.shape}")

TARGET = "price_lkr"
X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"Features : {X.shape[1]}")
print(f"Target   : {TARGET}  (mean={y.mean():,.0f} LKR, std={y.std():,.0f} LKR)")

# ─────────────────────────────────────────────
# 2. Train / Validation / Test split  (70/15/15)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 – Train / Validation / Test split  (70 / 15 / 15)")
print("=" * 60)

# First: carve out 15% test set
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# Then: split remaining 85% into 70% train / 15% val
# 15/85 = 0.1765 gives us exactly 15% of total data as val
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1765, random_state=42
)

total = len(df)
print(f"  Total samples : {total}")
print(f"  Train         : {len(X_train)}  ({len(X_train)/total*100:.1f}%)")
print(f"  Validation    : {len(X_val)}   ({len(X_val)/total*100:.1f}%)")
print(f"  Test          : {len(X_test)}   ({len(X_test)/total*100:.1f}%)")

# ─────────────────────────────────────────────
# 3. Baseline XGBoost (default hyperparameters)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 – Baseline XGBoost (default hyperparameters)")
print("=" * 60)

baseline_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)
baseline_model.fit(X_train, y_train)
y_val_pred_base = baseline_model.predict(X_val)

def print_metrics(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    print(f"  {label}")
    print(f"    MAE   (Mean Absolute Error)       : {mae:>15,.0f} LKR")
    print(f"    RMSE  (Root Mean Squared Error)   : {rmse:>15,.0f} LKR")
    print(f"    R2    (R-squared)                  :    {r2:.4f}")
    print(f"    MAPE  (Mean Abs Percentage Error) :    {mape:.2f}%")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

base_metrics = print_metrics(y_val, y_val_pred_base, "Validation – Baseline")

# ─────────────────────────────────────────────
# 4. Hyperparameter tuning – RandomizedSearchCV
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 – Hyperparameter tuning (RandomizedSearchCV, 5-fold CV)")
print("=" * 60)

param_dist = {
    "n_estimators"     : [100, 200, 300, 400, 500],
    "max_depth"        : [3, 4, 5, 6, 7, 8],
    "learning_rate"    : [0.01, 0.05, 0.1, 0.2],
    "subsample"        : [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree" : [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight" : [1, 3, 5, 7],
    "gamma"            : [0, 0.1, 0.2, 0.3, 0.5],
    "reg_alpha"        : [0, 0.01, 0.1, 1],       # L1 regularisation
    "reg_lambda"       : [1, 1.5, 2, 5],           # L2 regularisation
}

print("  Parameter grid:")
for k, v in param_dist.items():
    print(f"    {k:<22}: {v}")

xgb_base = xgb.XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist,
    n_iter=50,                  # 50 random combinations
    scoring="neg_root_mean_squared_error",
    cv=5,                       # 5-fold cross-validation
    verbose=1,
    random_state=42,
    n_jobs=-1,
    refit=True
)

print("\n  Running RandomizedSearchCV (50 iterations × 5-fold) ...")
random_search.fit(X_trainval, y_trainval)   # tune on full train+val combined

best_params = random_search.best_params_
best_cv_rmse = -random_search.best_score_

print(f"\n  Best CV RMSE : {best_cv_rmse:,.0f} LKR")
print(f"  Best params  :")
for k, v in best_params.items():
    print(f"    {k:<22}: {v}")

# ─────────────────────────────────────────────
# 5. Final model – retrain with best params & evaluate on test set
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 – Final evaluation on held-out Test set")
print("=" * 60)

best_model = random_search.best_estimator_

# Validation set metrics with best model
y_val_pred_tuned = best_model.predict(X_val)
tuned_val_metrics = print_metrics(y_val, y_val_pred_tuned, "Validation – Tuned model")

# Test set metrics (true generalisation performance)
y_test_pred = best_model.predict(X_test)
test_metrics = print_metrics(y_test, y_test_pred, "Test       – Tuned model")

# 5-fold CV score on full trainval set for stability check
cv_scores = cross_val_score(
    best_model, X_trainval, y_trainval,
    scoring="neg_root_mean_squared_error", cv=5
)
print(f"\n  5-Fold CV RMSE : {-cv_scores.mean():,.0f} LKR  (+/- {cv_scores.std():,.0f})")

# ─────────────────────────────────────────────
# 6. Feature importance plot
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 – Feature importance")
print("=" * 60)

importances = pd.Series(
    best_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

top_n = 20
print(f"  Top {top_n} most important features:")
print(importances.head(top_n).to_string())

fig, ax = plt.subplots(figsize=(10, 7))
importances.head(top_n).sort_values().plot(kind="barh", ax=ax, color="#2196F3")
ax.set_title(f"XGBoost – Top {top_n} Feature Importances", fontsize=14)
ax.set_xlabel("Importance Score (F-score)", fontsize=11)
ax.set_ylabel("Feature", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"), dpi=150)
print(f"\n  Feature importance chart saved -> '{os.path.join(FIGURES_DIR, 'feature_importance.png')}'")

# ─────────────────────────────────────────────
# 7. Actual vs Predicted scatter plot
# ─────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.scatter(y_test / 1e6, y_test_pred / 1e6, alpha=0.4, s=15, color="#4CAF50")
lims = [0, max(y_test.max(), y_test_pred.max()) / 1e6]
ax2.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
ax2.set_xlabel("Actual Price (Million LKR)", fontsize=11)
ax2.set_ylabel("Predicted Price (Million LKR)", fontsize=11)
ax2.set_title("Actual vs Predicted Car Prices (Test Set)", fontsize=13)
ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "actual_vs_predicted.png"), dpi=150)
print(f"  Actual vs Predicted chart saved -> '{os.path.join(FIGURES_DIR, 'actual_vs_predicted.png')}'")

# ─────────────────────────────────────────────
# 8. Save trained model
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 – Save model")
print("=" * 60)
joblib.dump(best_model, MODEL_FILE)
print(f"  Model saved -> '{MODEL_FILE}'")

# ─────────────────────────────────────────────
# 9. Summary report
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"  Algorithm         : XGBoost Regressor (Gradient Boosting)")
print(f"  Dataset size      : {total} rows, {X.shape[1]} features")
print(f"  Train/Val/Test    : {len(X_train)}/{len(X_val)}/{len(X_test)}")
print(f"  Tuning strategy   : RandomizedSearchCV (50 iter, 5-fold CV)")
print(f"")
print(f"  --- Baseline (default params) ---")
print(f"  Val MAE           : {base_metrics['MAE']:>12,.0f} LKR")
print(f"  Val RMSE          : {base_metrics['RMSE']:>12,.0f} LKR")
print(f"  Val R2            : {base_metrics['R2']:.4f}")
print(f"  Val MAPE          : {base_metrics['MAPE']:.2f}%")
print(f"")
print(f"  --- Tuned XGBoost ---")
print(f"  Test MAE          : {test_metrics['MAE']:>12,.0f} LKR")
print(f"  Test RMSE         : {test_metrics['RMSE']:>12,.0f} LKR")
print(f"  Test R2           : {test_metrics['R2']:.4f}")
print(f"  Test MAPE         : {test_metrics['MAPE']:.2f}%")
print(f"  5-Fold CV RMSE    : {-cv_scores.mean():>12,.0f} LKR (+/- {cv_scores.std():,.0f})")
print("=" * 60)
print("[DONE] Model training and evaluation complete.")
