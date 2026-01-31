from source import (
    FILE_PATH, TOP_FEATURES, RANDOM_SEED,
    load_and_clean_data, add_categorical_features, prepare_datasets, select_top_features,
    make_splits, train_model, evaluate_model,
    plot_feature_importance, predict_missing_gaps
)

def run():
    """
    Orchestrates the end-to-end pipeline for date gap prediction:
    Data loading, feature engineering, feature selection, model training, and inference.
    """
    # Load and clean the raw dataset
    df = load_and_clean_data(FILE_PATH)
    
    # Keep a copy for final output mapping
    original_df = df.copy()
    
    # Feature Engineering: Extract categorical/time-based features
    df = add_categorical_features(df)
    
    # Split data into labeled (known gaps) and unlabeled (missing gaps) sets
    df_known, df_missing = prepare_datasets(df)

    # Separate target variable (y) from features (X)
    y = df_known.pop("gap")
    X = df_known

    # Split labeled data into training and validation sets
    X_train, X_val, y_train, y_val = make_splits(X, y, random_state=RANDOM_SEED)
    
    # Initial training to determine feature importance
    model = train_model(X_train, y_train, random_state=RANDOM_SEED)

    # Feature Selection: Identify the most predictive features based on the initial model
    X_train_sel = select_top_features(X_train, model, TOP_FEATURES)
    X_val_sel = X_val[X_train_sel.columns]

    # Retrain model using only the selected top features for better generalization
    model = train_model(X_train_sel, y_train, random_state=RANDOM_SEED)
    
    # Validate model performance on the hold-out set
    _ = evaluate_model(model, X_val_sel, y_val)

    # Visualize which features influenced the predictions the most
    plot_feature_importance(model, X_train_sel, TOP_FEATURES)

    # Inference: Predict the 'gap' values for the missing entries
    df_missing_sel = df_missing[X_train_sel.columns]
    _ = predict_missing_gaps(model, df_missing_sel, original_df)

if __name__ == "__main__":
    run()
