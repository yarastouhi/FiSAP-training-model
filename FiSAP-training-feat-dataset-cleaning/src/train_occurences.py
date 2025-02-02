import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from preprocess import preprocess_occurrence
import pandas as pd
import numpy as np

class WildfireLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(WildfireLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

def create_sequences(data, feature_cols, target_col, sequence_length=24):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data.iloc[i:i + sequence_length][feature_cols].values)
        y.append(data.iloc[i + sequence_length][target_col])  # Predict next day's fire occurrence
    return np.array(X), np.array(y)

def split_data(df, feature_columns):
    """Creates sequences and splits data into training and testing sets."""
    X, y = create_sequences(df, feature_columns, 'fire_occurred', sequence_length=24)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_occurrence():
    # Load dataset
    df = pd.read_csv("../data/merged_wildfire_data.csv")
    df, feature_columns = preprocess_occurrence(df)

    X_train, X_test, y_train, y_test = split_data(df, feature_columns)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WildfireLSTM(input_size=len(feature_columns), hidden_size=64, num_layers=2, output_size=1).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "../models/wildfire_occurrence_lstm.pth")
    print("Wildfire occurrence model saved.")

    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    predictions = np.array(predictions) > 0.5
    true_labels = np.array(true_labels)

    print("LSTM Model Evaluation:")
    print(classification_report(true_labels, predictions))

if __name__ == "__main__":
    train_occurrence()