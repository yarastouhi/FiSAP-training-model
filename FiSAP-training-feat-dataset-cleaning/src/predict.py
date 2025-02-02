import torch
import pandas as pd
import pickle
from .preprocess import preprocess_occurrence, preprocess_severity, preprocess_time

def run_inference():
    # Load models
    occurrence_model = torch.load("models/wildfire_occurrence_lstm.pth")
    severity_model = torch.load("models/wildfire_severity_lstm.pth")
    with open("models/wildfire_time_rf.pkl", "rb") as f:
        time_model = pickle.load(f)

    # Load test dataset
    df_test = pd.read_csv("data/future_environmental_data.csv")

    # Predict wildfire occurrence
    df_occurrence, feature_columns = preprocess_occurrence(df_test)
    X_occurrence = df_occurrence[feature_columns].values
    X_occurrence_tensor = torch.tensor(X_occurrence, dtype=torch.float32)

    occurrence_predictions = (occurrence_model(X_occurrence_tensor) > 0.5).int().numpy()

    # Filter only fire-occurring days for severity prediction
    df_test["fire_predicted"] = occurrence_predictions
    df_severity = df_test[df_test["fire_predicted"] == 1]

    # Predict severity
    df_severity, severity_features = preprocess_severity(df_severity)
    X_severity = df_severity[severity_features].values
    X_severity_tensor = torch.tensor(X_severity, dtype=torch.float32)

    severity_predictions = torch.argmax(severity_model(X_severity_tensor), axis=1).numpy()

    # Predict fire time
    df_time, time_features = preprocess_time(df_severity)
    X_time = df_time[time_features].values

    time_predictions = time_model.predict(X_time)

    # Save results
    df_results = df_test[["date"]].copy()
    df_results["Fire_Predicted"] = occurrence_predictions
    df_results.loc[df_results["Fire_Predicted"] == 1, "Severity_Predicted"] = severity_predictions
    df_results.loc[df_results["Fire_Predicted"] == 1, "Fire_Hour_Predicted"] = time_predictions

    df_results.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv")
