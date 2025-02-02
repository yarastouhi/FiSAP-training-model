import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle
from lightgbm import LGBMRegressor
from preprocess import preprocess_time

def add_time_features(df):
    """Encodes the hour using sine and cosine transformations to capture its cyclic nature."""
    df["sin_hour"] = np.sin(2 * np.pi * df["fire_hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["fire_hour"] / 24)
    return df

def train_time():
    # Ensure dataset exists
    file_path = os.path.abspath("../data/merged_wildfire_data.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    # Preprocess data
    df, feature_columns = preprocess_time(df)
    df = add_time_features(df)

    # Fix CopyWarning
    df = df.copy()
    df.loc[:, feature_columns] = df[feature_columns]

    # Remove rare fire hours with fewer than 2 occurrences
    hour_counts = df["fire_hour"].value_counts()
    valid_hours = hour_counts[hour_counts > 1].index
    df = df[df["fire_hour"].isin(valid_hours)].copy()

    # Check feature correlation (Optional: Remove uncorrelated features)
    correlation = df.corr()["fire_hour"].abs().sort_values(ascending=False)
    print("Feature Correlation with Fire Hour:\n", correlation)

    # Define features and target
    X = df[feature_columns].values
    y = df["fire_hour"].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train LightGBM model
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        min_child_samples=3,  # Allow smaller splits
        num_leaves=31,  # Control complexity
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Mean Absolute Error for Fire Hour Prediction: {mae:.2f} hours")

    # Save model
    with open("../models/wildfire_time_lgb.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Fire time prediction model saved.")

if __name__ == "__main__":
    train_time()
