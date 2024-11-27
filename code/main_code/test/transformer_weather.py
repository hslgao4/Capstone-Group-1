import sys
sys.path.append('../../component')
from utils import *
from class_transformer import TransformerModel


#
# x_train, y_train, x_test, y_test, scaler = pre_transformer_data(path, target, seq_length)
#
# train_loader = set_dataloader(x_train, y_train, shuffle=True, batch_size=32)
# test_loader = set_dataloader(x_test, y_test, shuffle=False, batch_size=32)
#
# model = TransformerModel(input_dim=1, d_model=64, nhead=4, num_layers=2, dropout=0.2).to(device)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)
#
#
# train_model = transformer_train(model, train_loader, optimizer, criterion, epochs, device)
#
# predictions = transformer_eval(train_model, test_loader, criterion, device)
# rmse = np.sqrt(np.mean((scaler.inverse_transform(np.array(predictions).reshape(-1, 1)) - scaler.inverse_transform(y_test.numpy().reshape(-1, 1)))**2))
# print(f"Score(RMSE): {rmse:.4f}")


def main(path, target, seq_length, batch_size, epochs, device, lr=0.001, d_model=64, nhead=4, num_layers=2, dropout=0.2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_train, y_train, x_test, y_test, scaler = pre_transformer_data(path, target, seq_length)

    train_loader = set_dataloader(x_train, y_train, shuffle=True, batch_size=batch_size)
    test_loader = set_dataloader(x_test, y_test, shuffle=False, batch_size=batch_size)

    model = TransformerModel(d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_model = transformer_train(model, train_loader, optimizer, criterion, epochs, device)
    predictions = transformer_eval(train_model, test_loader, criterion, device)

    rmse = np.sqrt(np.mean((scaler.inverse_transform(np.array(predictions).reshape(-1, 1)) - scaler.inverse_transform(
        y_test.numpy().reshape(-1, 1))) ** 2))
    print(f"Score(RMSE): {rmse:.4f}")

    forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    actual = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))

    torch.save(train_model.state_dict(), 'transformer.pt')

    return model, forecast, actual


path = '../../data/weather.csv'
target = 'temperature'
seq_length = 10
epochs = 20
batch_size = 32
model, forecast, actual = main(path, target, seq_length, batch_size, epochs)

# torch.save(model.state_dict(), 'transformer.pt')
# model.load_state_dict(torch.load('transformer.pt'))