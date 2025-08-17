import pandas as pd

def predict_missing_gaps(model, df_missing, original_df):
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
    df_result["start date"] = pd.to_datetime(
        df_result["start year"].astype(str) + '-' + df_result["start month"].astype(str)
    )
    df_result["predicted end date"] = df_result["start date"] + pd.to_timedelta(df_result["predicted gap"], unit='D')
    df_result["predicted end year"] = df_result["predicted end date"].dt.year
    df_result["predicted end month"] = df_result["predicted end date"].dt.month
    print("\nPredictions for Missing End Dates:")
    print(df_result[["predicted gap", "predicted end year", "predicted end month"]])
    return df_result
