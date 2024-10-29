import sys
sys.path.append('../../component')
from utils import *
from class_LSTM import LSTM

# Load and preprocess data
path = '../../data/weather.csv'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
training_set = pd.read_csv(path)
training_set = training_set.iloc[:, 1:2].values

sc = MinMaxScaler()
training_data = sc.fit_transform(training_set)
seq_length = 30
print('seq_length:', seq_length)
x, y = sliding_windows(training_data, seq_length)

train_size = int(len(y) * 0.8)

trainX = torch.Tensor(x[:train_size]).to(device)
trainY = torch.Tensor(y[:train_size]).to(device)

testX = torch.Tensor(x[train_size:]).to(device)
testY = torch.Tensor(y[train_size:]).to(device)

# Hyperparameters
num_epochs = 2000
learning_rate = 0.001
input_size = 1
hidden_size = 2
num_layers = 1
output_size = 1

# Model, Loss, Optimizer
lstm = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    lstm.train()
    outputs = lstm(trainX)
    optimizer.zero_grad()
    loss = criterion(outputs, trainY)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, train Loss: {loss.item():.5f}")

'''Save the trained model'''
torch.save(lstm.state_dict(), 'LSTM_model_weights.pt')


# lstm = LSTM(num_classes, input_size, hidden_size, num_layers).to(device)
# lstm.load_state_dict(torch.load('model_weights.pt'))
lstm.eval()
with torch.no_grad():
    train_predict = lstm(trainX)

train_pred = train_predict.cpu().numpy()
trainY = trainY.cpu().numpy()
train_pred = sc.inverse_transform(train_pred)
train_actual = sc.inverse_transform(trainY)


# Test forecast
lstm.eval()
with torch.no_grad():
    test_predict = lstm(testX)

test_loss = criterion(test_predict, testY)

test_pred = test_predict.cpu().numpy()
test_actual = testY.cpu().numpy()

test_pred = sc.inverse_transform(test_pred)
test_actual = sc.inverse_transform(test_actual)

# plt.axvline(x=train_size, c='r', linestyle='--', label="Train/Test Split")
plt.plot(test_actual, label="True Data")
plt.plot(test_pred, label="Predicted Data")
plt.suptitle('Time-Series Prediction')
plt.legend()
plt.show()