import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from transformers import TimeSeriesTransformerForPrediction, TimeSeriesTransformerConfig
import joblib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the data
path = '../data/weather.csv'
training_set = pd.read_csv(path, parse_dates=['date'])
training_set.set_index('date', inplace=True)

# Step 1: Scale the Data
scaler = MinMaxScaler()

training_set['temperature'] = scaler.fit_transform(training_set[['temperature']])

# Split into train and test sets
split_ratio = 0.8
split_point = int(len(training_set) * split_ratio)
train_set = training_set.iloc[:split_point]
test_set = training_set.iloc[split_point:]

# Step 2: Add Time-Based Features
train_set = train_set.copy()
train_set['hour'] = train_set.index.hour + 1
train_set['minute'] = train_set.index.minute + 10
train_set['day_of_week'] = train_set.index.dayofweek +1
train_set['month'] = train_set.index.month

test_set = test_set.copy()
test_set['hour'] = test_set.index.hour
test_set['minute'] = test_set.index.minute
test_set['day_of_week'] = test_set.index.dayofweek
test_set['month'] = test_set.index.month


# Step 3: Create Sliding Windows for Training and Test Sets
def create_samples(data, window_size, n_future_steps):
    samples = []
    for i in range(len(data) - window_size - n_future_steps):
        past_values = data['temperature'].iloc[i:i + window_size].values
        past_time_features = data[['hour', 'minute', 'day_of_week', 'month']].iloc[i:i + window_size].values
        past_observed_mask = (~data['temperature'].iloc[i:i + window_size].isna()).astype(int).values
        future_values = data['temperature'].iloc[i + window_size:i + window_size + n_future_steps].values
        samples.append((past_values, past_time_features, past_observed_mask, future_values))
    return samples


# Parameters
window_size = 144  # 24 hours if data is every 10 minutes
n_future_steps = 1  # Forecasting 2 hours ahead

# Generate samples
train_samples = create_samples(train_set, window_size, n_future_steps)
test_samples = create_samples(test_set, window_size, n_future_steps)


# Step 4: Custom Dataset Class
class TimeSeriesDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        past_values, past_time_features, past_observed_mask, future_values = self.samples[idx]
        return {
            'past_values': torch.tensor(past_values, dtype=torch.float32),
            'past_time_features': torch.tensor(past_time_features, dtype=torch.float32),
            'past_observed_mask': torch.tensor(past_observed_mask, dtype=torch.float32),
            'future_values': torch.tensor(future_values, dtype=torch.float32)
        }


# Create Datasets and DataLoaders
batch_size = 64
train_dataset = TimeSeriesDataset(train_samples)
test_dataset = TimeSeriesDataset(test_samples)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Step 5: Configure and Initialize the Model
config = TimeSeriesTransformerConfig(
    prediction_length=n_future_steps,
    context_length=window_size,
    num_encoder_layers=4,
    num_decoder_layers=4,
    d_model=64,
    num_heads=4,
    activation_function="relu",
)

model = TimeSeriesTransformerForPrediction(config).to(device)

# Step 6: Training Loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()

        # Prepare batch data
        past_values = batch['past_values'].to(device)
        past_time_features = batch['past_time_features'].to(device)
        past_observed_mask = batch['past_observed_mask'].to(device)
        future_values = batch['future_values'].to(device)

        # Forward pass
        outputs = model(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            # static_categorical_features=torch.zeros((batch_size, 1), dtype=torch.int).to(device),
            static_real_features = torch.zeros((batch_size, 1), dtype=torch.int).to(device),
            future_values=future_values,
            future_time_features=torch.zeros((batch_size, n_future_steps, 4), dtype=torch.int).to(device),
        )

        # Loss (assuming outputs and future_values are compatible)
        loss = torch.nn.functional.mse_loss(outputs, future_values)
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

# Save the trained model
# torch.save(model.state_dict(), 'time_series_transformer_model.pth')
# joblib.dump(scaler, 'temperature_scaler.pkl')