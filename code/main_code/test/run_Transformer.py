import sys
sys.path.append('../../component')
from utils import *
from class_transformer import TransformerModel

##################################################################################################################

def set_data(path, target, seq_length):
    x_train, y_train, x_test, y_test, scaler = pre_transformer_data(path, target, seq_length)
    train_loader = set_dataloader(x_train, y_train, shuffle=True, batch_size=batch_size)
    test_loader = set_dataloader(x_test, y_test, shuffle=False, batch_size=batch_size)
    return train_loader, test_loader, scaler


def run_transformer(dataset, train_loader, epoches, lr=0.001, d_model=64, nhead=4, num_layers=2, dropout=0.2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TransformerModel(d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_model = transformer_train(model, train_loader, optimizer, criterion, epoches, device)
    # predictions = transformer_eval(train_model, test_loader, criterion, device)
    #
    # rmse = np.sqrt(np.mean((scaler.inverse_transform(np.array(predictions).reshape(-1, 1)) - scaler.inverse_transform(
    #     y_test.numpy().reshape(-1, 1))) ** 2))
    # print(f"Score(RMSE): {rmse:.4f}")
    #
    # forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    # actual = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))

    torch.save(train_model.state_dict(), f'{dataset}_transformer.pt')

    return model

def transformer_eval(dataset, model, test_loader, scaler):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(f'{dataset}_transformer.pt'))
    criterion = nn.MSELoss()

    model.eval()
    with torch.no_grad():
        predictions = []
        total_loss = 0
        for batch in test_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item()
            predictions.extend(outputs.squeeze().tolist())
        print(f'Test Loss: {total_loss / len(test_loader):.4f}')

    predictions = scaler.inverse_transform(torch.cat(predictions).numpy())

    return predictions

##################################################################################################################
path = '../../data/weather.csv'
dataset = 'wea'
target = 'temperature'

seq_length = 10
epoches = 20
batch_size = 32


train_loader, test_loader, scaler = set_data(path, target, seq_length)
model = run_transformer(dataset, train_loader, epoches)
predictions = transformer_eval(dataset, model, test_loader, scaler)