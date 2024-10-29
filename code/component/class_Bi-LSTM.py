from utils import *
from torch.utils.data import DataLoader, TensorDataset

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply hidden size by 2 for bidirectional

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Load and preprocess data
path = '../data/weather.csv'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
training_set = pd.read_csv(path)
data = training_set['temperature'].values.astype(float)

# Normalize data
scaler = MinMaxScaler(feature_range=(-1, 1))
data = scaler.fit_transform(data.reshape(-1, 1))

result = []
for seq_length in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
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


    input_size = 1
    hidden_size = 64 * 2
    num_layers = 2 + 1
    output_size = 1
    learning_rate = 0.001

    epochs = 10
    # Initialize model and optimizer
    model = BiLSTM(input_size, hidden_size, num_layers, output_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Train the model
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.8f}')

    result.append((seq_length, loss.item()))

# # Evaluate the model
# model.eval()
# with torch.no_grad():
#     predictions = model(X_test)
#
# print('test loss:', criterion(predictions, y_test))
# # Inverse transform to get actual values
# predictions = scaler.inverse_transform(predictions.cpu().numpy())
# y_test = scaler.inverse_transform(y_test.cpu().numpy())
#
#
# # Plot predictions
# plt.plot(y_test, label='Actual')
# plt.plot(predictions, label='Predicted')
# plt.legend()
# plt.show()