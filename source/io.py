import os
import pandas as pd

def load_and_clean_data(filepath):
    df = pd.read_excel(filepath)
    df = df[(df["start year"] != 0) & (df["start month"].between(1, 12))]
    drop_cols = ["Lamas num","case num","road num","semel yishuv","shem yishuv","shem medaveach","road name"]
    return df.drop(columns=drop_cols)
