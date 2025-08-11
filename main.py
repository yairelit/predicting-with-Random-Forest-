import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === CONFIGURATION ===
FILE_PATH = r"./data/roadworks_cbs.xls"
TOP_FEATURES = 100
RANDOM_SEED = 42

# === FUNCTIONS ===

def load_and_clean_data(filepath):
    """Load Excel file and drop irrelevant columns."""
    df = pd.read_excel(filepath)
    df = df[(df["start year"] != 0) & (df["start month"].between(1, 12))]

    drop_cols = [
        "Lamas num", "case num", "road num", 
        "semel yishuv", "shem yishuv", "shem medaveach",
        "road name"
    ]
    return df.drop(columns=drop_cols)

def add_categorical_features(df):
    """Add cyclical and one-hot encoded categorical features."""
    # Cyclical month encoding
    df["month_sin"] = np.sin(2 * np.pi * df["start month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["start month"] / 12)

    # Encode month as category
    month_labels = ['Jan_st','Feb_st','Mar_st','Apr_st','May_st','Jun_st',
                    'Jul_st','Aug_st','Sep_st','Oct_st','Nov_st','Dec_st']
    df["start month category"] = pd.cut(
        df["start month"], bins=[0.5 + i for i in range(12)] + [float('inf')],
        labels=month_labels, right=False
    )

    # Encode year as category
    year_labels = [f"{y}_st" for y in range(1983, 2023)]
    df["start year category"] = pd.cut(
        df["start year"], bins=list(range(1983, 2023)) + [float('inf')],
        labels=year_labels, right=False
    )

    # One-hot encode selected categorical columns
    return pd.get_dummies(df, columns=[
        "start month category", "start year category", 
        "sug slila", "road type", "way type"
    ])

def select_top_features(X, model, top_n):
    """Select top N features based on model's feature importance."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_columns = X.columns[indices]
    return X[top_columns]

def add_dates(df):
    """Add datetime columns for start and end dates."""
    df["start date"] = pd.to_datetime(
        df["start year"].astype(str) + '-' + df["start month"].astype(str)
    )
    df["end date"] = pd.to_datetime(
        df["end year"].astype(str) + '-' + df["end month"].astype(str), errors="coerce"
    )
    return df

def prepare_datasets(df):
    """Prepare datasets for training and prediction."""
    df = add_dates(df)
    df_known = df[df["end year"] != 0].copy()
    
    # Calculate target variable (gap in days)
    df_known["gap"] = (df_known["end date"] - df_known["start date"]).dt.days 
    df_known = df_known[df_known["gap"] >= 0] 

    df_missing = df[
        (df["end year"] == 0) & (df["end month"] == 0)
    ].copy()

    drop_cols = ["start month", "start year", "end month", "end year", "start date", "end date"]
    df_known.drop(columns=drop_cols, inplace=True, errors="ignore")
    df_missing.drop(columns=drop_cols, inplace=True, errors="ignore")

    gap = df_known.pop("gap")
    df_known["gap"] = gap

    return df_known, df_missing

def train_model(X_train, y_train):
    """Train RandomForestRegressor."""
    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    # נסה API חדש, אחרת fallback
    try:
        rmse = mean_squared_error(y_val, y_pred, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    print("Validation Metrics:")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.2f}")
    return y_pred

def plot_feature_importance(model, X, top_n=TOP_FEATURES):
    """Plot top N feature importances."""
    importances = model.feature_importances_
    feature_names = X.columns
    sorted_idx = np.argsort(importances)[::-1]

    top_n = min(top_n, len(importances))
    top_features = sorted_idx[:top_n]
    top_names = feature_names[top_features]
    top_importances = importances[top_features]

    plt.figure(figsize=(12, 6))
    plt.bar(range(top_n), top_importances)
    plt.xticks(range(top_n), top_names, rotation=90)
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()

def predict_missing_gaps(model, df_missing, original_df):
    """Predict missing 'gap' values and reconstruct end dates."""
    predicted_gaps = model.predict(df_missing)

    df_result = df_missing.copy().reset_index(drop=True)
    df_original_info = original_df[
        (original_df["end year"] == 0) & 
        (original_df["end month"] == 0) & 
        (original_df["start month"].between(1, 12))
    ].reset_index(drop=True)

    df_result["predicted gap"] = predicted_gaps
    df_result["start year"] = df_original_info["start year"].values
    df_result["start month"] = df_original_info["start month"].values
    df_result["start date"] = pd.to_datetime(df_result["start year"].astype(str) + '-' + df_result["start month"].astype(str))
    df_result["predicted end date"] = df_result["start date"] + pd.to_timedelta(df_result["predicted gap"], unit='D')
    df_result["predicted end year"] = df_result["predicted end date"].dt.year
    df_result["predicted end month"] = df_result["predicted end date"].dt.month

    print("\nPredictions for Missing End Dates:")
    print(df_result[["predicted gap", "predicted end year", "predicted end month"]])

    return df_result

# === MAIN SCRIPT ===

def main():
    # Load and preprocess data
    raw_df = load_and_clean_data(FILE_PATH)
    enriched_df = add_categorical_features(raw_df)
    df_known, df_missing = prepare_datasets(enriched_df)

    # Split features/target
    X = df_known.drop(columns=["gap"])
    y = df_known["gap"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    # First training - all features
    base_model = train_model(X_train, y_train)
    print("\n--- Evaluation with all features ---")
    evaluate_model(base_model, X_val, y_val)
    plot_feature_importance(base_model, X)

    # Select top features
    X_train_top = select_top_features(X_train, base_model, top_n=TOP_FEATURES)
    X_val_top = X_val[X_train_top.columns]

    # Train on top features
    top_model = train_model(X_train_top, y_train)
    print(f"\n--- Evaluation with top {TOP_FEATURES} features ---")
    evaluate_model(top_model, X_val_top, y_val)

    # Filter missing data with same top features
    df_missing_top = df_missing[X_train_top.columns]

    # Predict and display results
    final_predictions = predict_missing_gaps(top_model, df_missing_top, raw_df)
    # final_predictions.to_excel("F:\\predicted_end_dates.xlsx", index=False)

if __name__ == "__main__":
    main()
