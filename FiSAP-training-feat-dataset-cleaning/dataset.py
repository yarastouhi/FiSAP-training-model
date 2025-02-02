import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("/data/merged_wildfire_data.csv")

# Select relevant features
features = ["temperature", "humidity", "wind_speed", "precipitation", "vegetation_index"]
target = "severity"

# Normalize features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Split into features (X) and labels (y)
X = df[features].values
y = df[target].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(sampling_strategy="auto", random_state=42)  # Increase fire cases to 10% of total
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)