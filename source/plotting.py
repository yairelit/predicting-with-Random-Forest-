import numpy as np
import matplotlib.pyplot as plt

def plot_feature_importance(model, X, top_n):
    importances = model.feature_importances_
    feature_names = X.columns
    sorted_idx = np.argsort(importances)[::-1]
    top_n = min(top_n, len(importances))
    top_features = sorted_idx[:top_n]
    plt.figure(figsize=(12, 6))
    plt.bar(range(top_n), importances[top_features])
    plt.xticks(range(top_n), feature_names[top_features], rotation=90)
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Feature"); plt.ylabel("Importance")
    plt.tight_layout(); plt.show()
