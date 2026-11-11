"""
Machine Learning Training Pipeline: Used Car Price Prediction (Indian Market)
Developer: DESAM KOWSHIK REDDY | Roll Number: 23761A0584
Domain: B.Tech AI/ML Internship Mini Project
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

CURRENT_YEAR = 2026
DATASET_PATH = "car_data.csv"
MODEL_PATH = "best_car_model.pkl"
METRICS_PATH = "metrics.json"
METADATA_PATH = "metadata.json"

CATEGORICAL_FEATURES = ["Brand", "Model", "Fuel_Type", "Seller_Type", "Transmission"]
NUMERICAL_FEATURES = ["Present_Price", "Kms_Driven", "Owner", "Car_Age"]

def load_or_generate_data(dataset_path: str = DATASET_PATH) -> pd.DataFrame:
    """Loads existing dataset or triggers generator if not found."""
    if not os.path.exists(dataset_path):
        print(f"'{dataset_path}' not found. Triggering automated dataset generator...")
        from generate_dataset import generate_car_dataset
        df = generate_car_dataset(1500, dataset_path)
    else:
        df = pd.read_csv(dataset_path)
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Performs feature engineering: creates Car_Age and removes raw Year."""
    df_engineered = df.copy()
    if "Year" in df_engineered.columns:
        df_engineered["Car_Age"] = CURRENT_YEAR - df_engineered["Year"]
        df_engineered.drop(columns=["Year"], inplace=True)
    return df_engineered

def build_preprocessor() -> ColumnTransformer:
    """Constructs ColumnTransformer for encoding and scaling."""
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor

def train_and_evaluate_models():
    """Trains Linear Regression, Random Forest, and XGBoost, evaluates and saves artifacts."""
    print("=" * 70)
    print("USED CAR PRICE PREDICTION - TRAINING & BENCHMARK PIPELINE")
    print("Developer: DESAM KOWSHIK REDDY | Roll No: 23761A0584")
    print("=" * 70)

    df_raw = load_or_generate_data()
    print(f"Loaded {len(df_raw)} records successfully.")

    # Extract metadata for UI and feature mappings
    brands = sorted(df_raw["Brand"].unique().tolist())
    models_by_brand = {}
    for brand in brands:
        models_by_brand[brand] = sorted(df_raw[df_raw["Brand"] == brand]["Model"].unique().tolist())

    metadata = {
        "student_name": "DESAM KOWSHIK REDDY",
        "roll_number": "23761A0584",
        "project_title": "Used Car Price Prediction System (Indian Market)",
        "current_year": CURRENT_YEAR,
        "brands": brands,
        "models_by_brand": models_by_brand,
        "fuel_types": sorted(df_raw["Fuel_Type"].unique().tolist()),
        "seller_types": sorted(df_raw["Seller_Type"].unique().tolist()),
        "transmissions": sorted(df_raw["Transmission"].unique().tolist()),
        "owners": [0, 1, 2],
        "feature_ranges": {
            "present_price_min": float(df_raw["Present_Price"].min()),
            "present_price_max": float(df_raw["Present_Price"].max()),
            "kms_driven_min": int(df_raw["Kms_Driven"].min()),
            "kms_driven_max": int(df_raw["Kms_Driven"].max()),
            "year_min": int(df_raw["Year"].min()),
            "year_max": int(df_raw["Year"].max()),
            "selling_price_min": float(df_raw["Selling_Price"].min()),
            "selling_price_max": float(df_raw["Selling_Price"].max()),
        },
    }

    # Feature Engineering
    df = feature_engineering(df_raw)

    target_col = "Selling_Price"
    X = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Dataset split: Train = {len(X_train)} samples, Test = {len(X_test)} samples.")

    # Candidate Models: Linear Regression, Random Forest, XGBoost
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42
        ),
    }

    results = {}
    best_model_name = None
    best_pipeline = None
    best_r2 = -float("inf")
    best_y_pred = None

    print("\n--- Model Training & Comparative Evaluation ---")
    for name, model in models.items():
        preprocessor = build_preprocessor()
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        results[name] = {
            "R2_Score": round(r2, 4),
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
        }
        print(f"[{name}] -> R2: {r2:.4f} | MAE: INR {mae:.4f} Lakhs | RMSE: INR {rmse:.4f} Lakhs")

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipeline
            best_y_pred = y_pred

    print(f"\nWinning Model: '{best_model_name}' with R2 Score of {best_r2:.4f}")

    # 5-Fold Cross Validation for best model
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_pipeline, X, y, cv=kf, scoring="r2")
    cv_r2_mean = round(float(np.mean(cv_scores)), 4)
    cv_r2_std = round(float(np.std(cv_scores)), 4)
    print(f"5-Fold CV R2 Score: {cv_r2_mean:.4f} (+/- {cv_r2_std:.4f})")

    # Extract feature importance for the best model if available
    feature_importance_list = []
    try:
        preprocessor_fitted = best_pipeline.named_steps["preprocessor"]
        cat_encoder = preprocessor_fitted.named_transformers_["cat"]
        encoded_cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        all_feature_names = NUMERICAL_FEATURES + encoded_cat_names

        model_fitted = best_pipeline.named_steps["model"]
        if hasattr(model_fitted, "feature_importances_"):
            importances = model_fitted.feature_importances_
            feat_imp = sorted(
                zip(all_feature_names, [float(x) for x in importances]),
                key=lambda x: x[1],
                reverse=True,
            )
            feature_importance_list = [
                {"feature": f, "importance": round(imp, 4)} for f, imp in feat_imp[:10]
            ]
    except Exception as e:
        print(f"Notice: Feature importance extraction skipped: {e}")

    # Residuals & Actual vs Predicted evaluation sample for visualization
    sample_actual_vs_pred = [
        {"actual": round(float(a), 2), "predicted": round(float(p), 2), "residual": round(float(a - p), 2)}
        for a, p in zip(y_test.tolist()[:80], best_y_pred.tolist()[:80])
    ]

    # Save best model pipeline
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"Saved best pipeline model to '{MODEL_PATH}'.")

    # Metrics dictionary
    metrics_data = {
        "models": results,
        "best_model": best_model_name,
        "best_metrics": results[best_model_name],
        "cv_r2_mean": cv_r2_mean,
        "cv_r2_std": cv_r2_std,
        "top_features": feature_importance_list,
        "actual_vs_pred_sample": sample_actual_vs_pred,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Saved evaluation metrics to '{METRICS_PATH}'.")

    metadata["best_model"] = best_model_name
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata configurations to '{METADATA_PATH}'.")
    print("=" * 70)

if __name__ == "__main__":
    train_and_evaluate_models()
