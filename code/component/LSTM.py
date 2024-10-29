import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import torch.cuda.amp as amp
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return (self.data[idx:idx + self.seq_length],
                self.data[idx + 1:idx + self.seq_length + 1])


class LSTMTimeSeriesPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMTimeSeriesPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Additional layers for better feature extraction
        self.feature_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Output layer
        self.fc = nn.Linear(hidden_size // 2, input_size)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Pass through feature extraction layers
        features = self.feature_layers(lstm_out)

        # Final output
        predictions = self.fc(features)

        return predictions


def prepare_data(data, seq_length, train_size=0.8, batch_size=32, num_workers=4):
    # Normalize the data
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)

    # Create dataset
    dataset = TimeSeriesDataset(data_normalized, seq_length)

    # Split into train and test
    train_size = int(len(dataset) * train_size)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create data loaders with GPU pinned memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader, scaler


def train_model(model, train_loader, test_loader, epochs=50, learning_rate=0.001):
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        # Training loop with mixed precision
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()

            with autocast():
                output = model(batch_x)
                loss = criterion(output, batch_y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                with autocast():
                    output = model(batch_x)
                    val_loss += criterion(output, batch_y).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'best_lstm_model.pth')

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}')

        # Clear GPU cache periodically
        if torch.cuda.is_available() and (epoch + 1) % 10 == 0:
            torch.cuda.empty_cache()

    return train_losses, val_losses


def predict(model, data, scaler, seq_length, future_steps=30):
    model.eval()

    # Use last seq_length points as input
    input_data = data[-seq_length:]
    input_data = scaler.transform(input_data)
    input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(device)

    predictions = []

    with torch.no_grad(), autocast():
        for _ in range(future_steps):
            output = model(input_tensor)
            last_prediction = output[:, -1:]
            predictions.append(last_prediction.cpu().numpy())

            # Update input tensor for next prediction
            input_tensor = torch.cat([input_tensor[:, 1:, :], last_prediction], dim=1)

    predictions = np.concatenate(predictions, axis=0)
    predictions = scaler.inverse_transform(predictions)

    return predictions


def plot_results(train_losses, val_losses, predictions, actual_data=None):
    # Plot training curves
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot predictions
    plt.subplot(1, 2, 2)
    if actual_data is not None:
        plt.plot(actual_data[-50:], label='Actual')
    plt.plot(range(len(actual_data) - 1, len(actual_data) + len(predictions) - 1),
             predictions, label='Predictions', linestyle='--')
    plt.title('Time Series Predictions')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()

    plt.tight_layout()
    plt.savefig('lstm_results.png')
    plt.close()


def main():
    # Memory tracking
    if torch.cuda.is_available():
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
        print(f"Initial GPU memory cached: {torch.cuda.memory_reserved(0) / 1024 ** 2:.2f} MB")

    # Load data
    data = pd.read_csv('../data/weather.csv')
    data = data.iloc[:, 1:2].values

    # Hyperparameters
    seq_length = 30
    batch_size = 64
    hidden_size = 128
    num_layers = 2
    epochs = 2000

    # Prepare data
    train_loader, test_loader, scaler = prepare_data(
        data,
        seq_length,
        batch_size=batch_size,
        num_workers=4
    )

    # Initialize model
    model = LSTMTimeSeriesPredictor(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers
    )

    # Print model summary
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters())}")

    # Train model
    train_losses, val_losses = train_model(model, train_loader, test_loader, epochs=epochs)

    # Make predictions
    predictions = predict(model, data, scaler, seq_length, future_steps=30)

    # Plot results
    plot_results(train_losses, val_losses, predictions, data)

    # Final memory stats
    if torch.cuda.is_available():
        print(f"\nFinal GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
        print(f"Final GPU memory cached: {torch.cuda.memory_reserved(0) / 1024 ** 2:.2f} MB")

    return predictions


if __name__ == "__main__":
    predictions = main()