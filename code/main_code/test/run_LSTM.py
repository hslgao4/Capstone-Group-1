import sys
sys.path.append('../../component')
from utils import *
import torch
from class_LSTM import LSTM, BiLSTM, Seq2SeqLSTM

##########################################
def set_lstm_data(path, target, seq_length, batch_size=128):
    X_train, y_train, X_test, y_test, scaler = pre_lstm_data(path, target, seq_length)
    train_loader = set_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    test_loader = set_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)
    actual_test = scaler.inverse_transform(y_test.cpu().numpy())
    return train_loader, test_loader, scaler, actual_test


def run_lstm(dataset, train_loader, model_name, epochs=100, learning_rate=0.001, hidden_size=2, num_layers=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = {'lstm': LSTM,
              'Bilstm': BiLSTM}
    model = models.get(model_name, Seq2SeqLSTM)(hidden_size=hidden_size, num_layers=num_layers).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_model = lstm_train(model_name, model, train_loader, optimizer, criterion, epochs, device)
    torch.save(train_model.state_dict(), f'../main/{dataset}_{model_name}_weights.pt')
    torch.save(train_model.state_dict(), f'{dataset}_{model_name}_weights.pt')
    return model

def lstm_eval(model_name, dataset, model, test_loader, scaler):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(f'{dataset}_{model_name}_weights.pt'))
    criterion = nn.MSELoss()

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
    print('loss:', total_loss / len(test_loader))

    predictions = scaler.inverse_transform(torch.cat(predictions).numpy())
    return predictions

