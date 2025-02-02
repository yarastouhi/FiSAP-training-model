import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_occurrence(df):
    """Processes dataset for wildfire occurrence prediction (binary classification)."""

    severity_mapping = {'0': 0, 'low': 1, 'medium': 2, 'high': 3}
    df['severity'] = df['severity'].map(severity_mapping)

    df['fire_occurred'] = (df['severity'] > 0).astype(int)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    fire_days = df.groupby('date')['fire_occurred'].max().reset_index()
    df.drop(columns=['fire_occurred'], inplace=True)
    df = df.merge(fire_days, on='date', how='left')

    feature_columns = ['temperature', 'humidity', 'wind_speed', 'precipitation',
                       'vegetation_index', 'human_activity_index']

    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    df.drop(columns=['timestamp', 'severity', 'latitude', 'longitude'], errors='ignore', inplace=True)
    return df, feature_columns


def preprocess_severity(df):
    """Processes dataset for wildfire severity classification (multi-class)."""

    severity_mapping = {'low': 1, 'medium': 2, 'high': 3}
    df['severity'] = df['severity'].map(severity_mapping) - 1  # Adjust to {0, 1, 2}

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    daily_severity = df.groupby('date')['severity'].max().reset_index()

    df.drop(columns=['severity'], inplace=True)
    df = df.merge(daily_severity, on='date', how='left')
    df = df[df['severity'] >= 0].copy()


    df.to_csv("severity_cleaned.csv", index=False)

    feature_columns = ['temperature', 'humidity', 'wind_speed', 'precipitation',
                       'vegetation_index', 'human_activity_index']

    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    return df[feature_columns], df['severity'].astype(int)


def preprocess_time(df):
    """Processes dataset for fire time regression (predicting hour of fire occurrence)."""

    severity_mapping = {'low': 1, 'medium': 2, 'high': 3}
    df["severity"] = df["severity"].map(severity_mapping)
    df = df[df['severity'] > 1]  # Keep only fire days

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['fire_hour'] = df['timestamp'].dt.hour

    df.drop(columns=['severity', 'latitude', 'longitude', 'timestamp'], inplace=True)

    feature_columns = ['temperature', 'humidity', 'wind_speed', 'precipitation',
                       'vegetation_index', 'human_activity_index']

    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    return df, feature_columns
