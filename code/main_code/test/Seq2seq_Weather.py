import sys
import pandas as pd
sys.path.append('../../component')  # Ensure sys is imported before using it
from utils import *
from class_seq2seq import Seq2SeqLSTM
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# prepare data
# Load and preprocess data
path = '../../data/weather.csv'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
training_set = pd.read_csv(path)
data = training_set['temperature'].values.astype(float)

# Normalize data
scaler = MinMaxScaler(feature_range=(-1, 1))
data = scaler.fit_transform(data.reshape(-1, 1))

seq_length = 5
X, y = sliding_windows(data, seq_length)
print('sequence length:', seq_length)

# Split into train/test sets
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

batch_size = 128 * 20

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=6)



# Hyperparameters
input_size = 1
hidden_size = 64
output_size = 1
num_layers = 2
learning_rate = 0.001
epochs = 10
batch_size = 64 * 2

# Initialize model, criterion, and optimizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Seq2SeqLSTM(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Sample training loop (assuming DataLoader is set up)
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:  # Assuming train_loader is already defined
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}')

# save the model
torch.save(model.state_dict(), 'seq2seq_model_weights.pt')




model = Seq2SeqLSTM(input_size, hidden_size, output_size, num_layers).to(device)
model.load_state_dict(torch.load('seq2seq_model_weights.pt'))

model.eval()
with torch.no_grad():
    total_loss = 0
    predictions = []
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        total_loss += loss.item()
        predictions.append(outputs.cpu())

print('test loss:', total_loss / len(test_loader))


# Inverse transform to get actual values
predictions = torch.cat(predictions).numpy()
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.cpu().numpy())

# Plot predictions
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()