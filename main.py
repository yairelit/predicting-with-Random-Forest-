from source import (
    FILE_PATH, TOP_FEATURES, RANDOM_SEED,
    load_and_clean_data, add_categorical_features, prepare_datasets, select_top_features,
    make_splits, train_model, evaluate_model,
    plot_feature_importance, predict_missing_gaps
)

def run():
    df = load_and_clean_data(FILE_PATH)
    original_df = df.copy()
    df = add_categorical_features(df)
    df_known, df_missing = prepare_datasets(df)

    y = df_known.pop("gap")
    X = df_known

    X_train, X_val, y_train, y_val = make_splits(X, y, random_state=RANDOM_SEED)
    model = train_model(X_train, y_train, random_state=RANDOM_SEED)

    # בחירת פיצ’רים חשובה – עשה על בסיס סט אימון (ואז יישם על ולידציה/חיזוי)
    X_train_sel = select_top_features(X_train, model, TOP_FEATURES)
    X_val_sel = X_val[X_train_sel.columns]

    # אמן מחדש אחרי בחירה (אופציונלי אבל נקי)
    model = train_model(X_train_sel, y_train, random_state=RANDOM_SEED)
    _ = evaluate_model(model, X_val_sel, y_val)

    plot_feature_importance(model, X_train_sel, TOP_FEATURES)

    # חיזוי לשורות החסרות
    df_missing_sel = df_missing[X_train_sel.columns]
    _ = predict_missing_gaps(model, df_missing_sel, original_df)

if __name__ == "__main__":
    run()
