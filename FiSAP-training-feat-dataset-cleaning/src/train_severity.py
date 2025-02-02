import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from preprocess import preprocess_severity  # Import preprocessing function

# Define LSTM Model
class WildfireSeverityLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=3, dropout_rate=0.3):
        super(WildfireSeverityLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)  # Additional dropout for regularization
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Apply dropout before FC layer
        return self.fc(out)  # No softmax (handled by CrossEntropyLoss)


# Create sequences for training
def create_sequences(data, target_col, sequence_length=24):
    """Creates 24-hour sequences for severity prediction."""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data.iloc[i:i + sequence_length].values)  # Use all features over 24 hours
        y.append(data.iloc[i + sequence_length - 1][target_col])  # Predict severity for the last hour
    return np.array(X), np.array(y)


# Split data into train and test sets
def split_data(X, y):
    """Splits data into training and testing sets."""
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


# Training function
def train_severity():
    """Trains the LSTM model for wildfire severity classification."""

    # Load dataset and preprocess
    df = pd.read_csv("../data/merged_wildfire_data.csv")
    X_df, y_df = preprocess_severity(df)  # Get features and severity labels

    # Reset indices to align features and labels correctly
    X_df.reset_index(drop=True, inplace=True)
    y_df.reset_index(drop=True, inplace=True)

    # Combine features and labels into one DataFrame for sequencing
    df_processed = X_df.copy()
    df_processed['severity'] = y_df  # Now, the length should match

    # Create sequences
    X, y = create_sequences(df_processed, target_col='severity', sequence_length=24)

    # Split dataset
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32, shuffle=False)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WildfireSeverityLSTM(input_size=X.shape[2]).to(device)

    # Compute class weights to handle class imbalance
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop with early stopping
    best_loss = float('inf')
    for epoch in range(20):  # Increase epochs for better learning
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

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)  # Adjust LR based on validation loss

        print(f"Epoch {epoch+1}/20, Loss: {avg_loss:.4f}")

        # Early stopping and best model saving
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "../models/best_wildfire_severity_lstm.pth")
            print("Model improved and saved!")

    print("Final Model Training Complete.")

    # Evaluation
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted_labels = torch.argmax(outputs, dim=1)  # Get predicted class
            predictions.extend(predicted_labels.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    print("LSTM Model Evaluation:")
    print(classification_report(true_labels, predictions))
    print(confusion_matrix(true_labels, predictions))


if __name__ == "__main__":
    train_severity()
