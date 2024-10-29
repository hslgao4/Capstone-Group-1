import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TimeSeriesTransformer(nn.Module):
    def __init__(self, feature_size=1, d_model=64, nhead=4, num_layers=3, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()

        self.encoder = nn.Linear(feature_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.decoder = nn.Linear(d_model, feature_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output


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

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        # Training loop with mixed precision
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()

            # Use autocast for mixed precision training
            with autocast():
                output = model(batch_x)
                loss = criterion(output, batch_y)

            # Scale gradients and optimize
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

        avg_val_loss = val_loss / len(test_loader)
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss / len(train_loader):.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}')



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


def main():
    # Your data loading
    data = pd.read_csv('../data/weather.csv')
    data = data.iloc[:, 1:2].values

    # Hyperparameters
    seq_length = 4
    batch_size = 64  # Increased for GPU
    d_model = 64
    nhead = 4
    num_layers = 3
    epochs = 10

    # Prepare data with GPU optimization
    train_loader, test_loader, scaler = prepare_data(
        data,
        seq_length,
        batch_size=batch_size,
        num_workers=4
    )

    # Initialize model
    model = TimeSeriesTransformer(
        feature_size=1,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    )

    # Train model
    train_model(model, train_loader, test_loader, epochs=epochs)

    # Make predictions
    # predictions = predict(model, data, scaler, seq_length, future_steps=30)

if __name__ == '__main__':
    main()