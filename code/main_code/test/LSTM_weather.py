import sys
sys.path.append('../../component')
from utils import *
import torch
from class_LSTM import LSTM, BiLSTM, Seq2SeqLSTM
from torch.utils.data import DataLoader, TensorDataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = '../../data/weather.csv'


'''LSTM'''
#%% LSTM model
seq_length = 30
X_train, y_train, X_test, y_test, scaler = pre_lstm_data(path, 'temperature', seq_length)

batch_size = 128 * 20
num_epochs = 2000
learning_rate = 0.001
input_size = 1
hidden_size = 2
num_layers = 1
output_size = 1

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

# Model, Loss, Optimizer
lstm = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    lstm.train()

    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = lstm(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.8f}')

'''Save the trained model'''
torch.save(lstm.state_dict(), 'LSTM_model_weights.pt')


# Test forecast
lstm.eval()
with torch.no_grad():
    total_loss = 0
    predictions = []
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = lstm(X_batch)
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



'''BiLSTM'''
#%% BiLSTM
seq_length = 2
print('sequence length:', seq_length)
X_train, y_train, X_test, y_test, scaler = pre_lstm_data(path, 'temperature', seq_length)

batch_size = 128 * 20
input_size = 1
hidden_size = 64 * 2
num_layers = 2 + 1
output_size = 1
learning_rate = 0.001
epochs = 10

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

# Initialize model and optimizer
Bilstm = BiLSTM(input_size, hidden_size, num_layers, output_size).to(device)
optimizer = torch.optim.Adam(Bilstm.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Train the model
for epoch in range(epochs):
    Bilstm.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = Bilstm(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.8f}')


# save the model
torch.save(Bilstm.state_dict(), 'BiLSTM_model_weights.pt')


Bilstm.eval()
with torch.no_grad():
    total_loss = 0
    predictions = []
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = Bilstm(X_batch)
        loss = criterion(outputs, y_batch)
        total_loss += loss.item()
        predictions.append(outputs.cpu())

print('test loss:', total_loss / len(test_loader))

predictions = torch.cat(predictions).numpy()
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.cpu().numpy())

# Plot predictions
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()





'''Seq2seq'''
#%% Seg2seq
seq_length = 5
X_train, y_train, X_test, y_test, scaler = pre_lstm_data(path, 'temperature', seq_length)

batch_size = 128 * 20
input_size = 1
hidden_size = 64
output_size = 1
num_layers = 2
learning_rate = 0.001
epochs = 100
batch_size = 64 * 2

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

# Initialize model, criterion, and optimizer
seg2seq = Seq2SeqLSTM(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(seg2seq.parameters(), lr=learning_rate)

for epoch in range(epochs):
    seg2seq.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        predictions = seg2seq(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}')

# save the model
torch.save(seg2seq.state_dict(), 'seq2seq_model_weights.pt')

seg2seq.eval()
with torch.no_grad():
    total_loss = 0
    predictions = []
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = seg2seq(X_batch)
        loss = criterion(outputs, y_batch)
        total_loss += loss.item()
        predictions.append(outputs.cpu())

print('test loss:', total_loss / len(test_loader))

predictions = torch.cat(predictions).numpy()
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.cpu().numpy())

plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()