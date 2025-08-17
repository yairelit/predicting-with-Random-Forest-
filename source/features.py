import pandas as pd
import numpy as np

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
