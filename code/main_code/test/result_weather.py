import sys
import pandas as pd
sys.path.append('../../component')
from utils import *
from class_LSTM import LSTM, BiLSTM, Seq2SeqLSTM
import os
os.getcwd()

path = '../../data/weather.csv'

"""Classical Time Series Model"""
target = 'temperature'
df_train, df_test, df_val, train, test, validation = prepare_arima_data(path, target)

'''AR model'''
#%% Domain knowledge
ar_train_err_1, ar_test_err_1, ar_train_err_mse_1, ar_test_err_mse_1, ar_train_plt_1, ar_test_plt_1 = ARIMA_results(2, 0, 0, df_train, df_test, train, test, validation)

#%% Optuna
ar_train_err_2, ar_test_err_2, ar_train_err_mse_2, ar_test_err_mse_2, ar_train_plt_2, ar_test_plt_2 = ARIMA_results(4, 0, 0, df_train, df_test, train, test, validation)

ar_train_err_acf = plt_ACF(ar_train_err_2, 40)
ar_test_err_acf = plt_ACF(ar_test_err_2, 40)

'''MA model'''
#%% No MA pattern in ACF/PACF, so just use Optuna MA(10)
ma_train_err, ma_test_err, ma_train_err_mse, ma_test_err_mse, ma_train_plt_1, ma_test_plt_1 = ARIMA_results(10, 0, 0, df_train, df_test, train, test, validation)

ma_train_err_acf = plt_ACF(ma_train_err, 40)
ma_test_err_acf = plt_ACF(ma_test_err, 40)

'''ARMA model'''
#%% GPAC: ARMA(2,3)
arma_train_err_1, arma_test_err_1, arma_train_err_mse_1, arma_test_err_mse_1, arma_train_plt_1, arma_test_plt_1 = ARIMA_results(2, 3, 0, df_train, df_test, train, test, validation)

# Optuna(12,11)
arma_train_err_2, arma_test_err_2, arma_train_err_mse_2, arma_test_err_mse_2, arma_train_plt_2, arma_test_plt_2 = ARIMA_results(6, 6, 0, df_train, df_test, train, test, validation)

arma_train_err_acf = plt_ACF(arma_train_err_2, 40)
arma_test_err_acf = plt_ACF(arma_test_err_2, 40)

'''ARIMA model'''
#%% GPAC ARIMA(2,1,3)
arima_train_err_1, arima_test_err_1, arima_train_err_mse_1, arima_test_err_mse_1, arima_train_plt_1, arima_test_plt_1 = ARIMA_results(2, 3, 1, df_train, df_test, train, test, validation)

#%% Optuna ARIMA(6, 1, 0)
arima_train_err_2, arima_test_err_2, arima_train_err_mse_2, arima_test_err_mse_2, arima_train_plt_2, arima_test_plt_2 = ARIMA_results(6, 0, 1, df_train, df_test, train, test, validation)

arima_train_err_acf = plt_ACF(arima_train_err_2, 40)
arima_test_err_acf = plt_ACF(arima_test_err_2, 40)






"""LSTM, BiLSTM, Seq2seq"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''LSTM'''
#%%
seq_length = 30
X_train, y_train, X_test, y_test, scaler = pre_lstm_data(path, 'temperature', seq_length)

batch_size = 128 * 20
input_size = 1
hidden_size = 2
num_layers = 1
output_size = 1

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

lstm = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
lstm.load_state_dict(torch.load('LSTM_model_weights.pt'))

train_prediction, lstm_pred_mse = lstm_loop(lstm, train_loader, device, criterion)
test_forecast, lstm_fore_mse = lstm_loop(lstm, test_loader, device, criterion)

lstm_train_pred = scaler.inverse_transform(torch.cat(train_prediction).numpy())
lstm_test_fore = scaler.inverse_transform(torch.cat(test_forecast).numpy())

y_train = scaler.inverse_transform(y_train.cpu().numpy())
y_test = scaler.inverse_transform(y_test.cpu().numpy())



'''BiLSTM'''
#%%
seq_length = 2
X_train, y_train, X_test, y_test, scaler = pre_lstm_data(path, 'temperature', seq_length)

batch_size = 128 * 20
input_size = 1
hidden_size = 64 * 2
num_layers = 2 + 1
output_size = 1

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

Bilstm = BiLSTM(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
Bilstm.load_state_dict(torch.load('BiLSTM_model_weights.pt'))

train_prediction, Bilstm_pred_mse = lstm_loop(Bilstm, train_loader, device, criterion)
test_forecast, Bilstm_fore_mse = lstm_loop(Bilstm, test_loader, device, criterion)

Bilstm_train_pred = scaler.inverse_transform(torch.cat(train_prediction).numpy())
Bilstm_test_fore = scaler.inverse_transform(torch.cat(test_forecast).numpy())


'''Seq2seq'''
#%%
seq_length = 5
X_train, y_train, X_test, y_test, scaler = pre_lstm_data(path, 'temperature', seq_length)

batch_size = 64 * 2
input_size = 1
hidden_size = 64
output_size = 1
num_layers = 2

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

seg2seq = Seq2SeqLSTM(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss()
seg2seq.load_state_dict(torch.load('seq2seq_model_weights.pt'))

train_prediction, seq2seq_pred_mse = lstm_loop(seg2seq, train_loader, device, criterion)
test_forecast, seq2seq_fore_mse = lstm_loop(seg2seq, test_loader, device, criterion)

seq2seq_train_pred = scaler.inverse_transform(torch.cat(train_prediction).numpy())
seq2seq_test_fore = scaler.inverse_transform(torch.cat(test_forecast).numpy())



'''Single dataset model performance'''
#%%
weather = pd.DataFrame({'Models': ['AR', 'MA', 'ARMA', 'ARIMA'],
                        'Train prediction': [ar_train_plt_2, ma_train_plt_1, arma_train_plt_2, arima_train_plt_2],
                        'Train error ACF': [ar_train_err_acf, ma_train_err_acf, arma_train_err_acf, arima_train_err_acf],
                        'Test prediction': [ar_test_plt_2, ma_test_plt_1, arma_test_plt_2, arima_test_plt_2],
                        'Test error ACF': [ar_test_err_acf, ma_test_err_acf, arma_test_err_acf,arima_test_err_acf]})

figure_table(weather, 'Model performance comparison')



# Optuna vs domain knowledge
df_grid_search = pd.DataFrame({'Models': ['AR', 'MA', 'ARMA', 'ARIMA'],
                               'Optuna': [4, 10, (6,6), (6,1,0)],
                               'MSE_1': [ar_test_err_mse_2, ma_test_err_mse, arma_test_err_mse_2, arima_test_err_mse_2],
                               'GPAC & ACF/PACF': [2, 'N/A', (2,3), (2,1,3)],
                               'MSE_2': [ar_test_err_mse_1, 'N/A', arma_test_err_mse_1, arima_test_err_mse_1]})


plot_metric_table(df_grid_search, 'Optuna vs. Domain knowledge')



# Model performance between all datasets, the metric
dfs_metric = pd.DataFrame({'Dataset': ['Temperature', 'Air Pollution', 'Power Consumption', 'Traffic Volume'],
                             'AR model': [ar_test_err_mse_1, 0, 0, 0],
                             'MA model': [ma_test_err_mse, 0, 0, 0 ],
                             'ARMA model': [arma_test_err_mse_1, 0, 0, 0 ],
                             'ARIMA model': [arima_test_err_mse_2, 0, 0, 0 ]})
plot_metric_table(dfs_metric, 'Metric comparison for 4 Datasets')



test_prediction = pd.DataFrame({'Models': ['AR', 'MA', 'ARMA', 'ARIMA'],
                        'Temperature': [ar_test_plt_2, ma_test_plt_1, arma_test_plt_2, arima_test_plt_2],
                        'Air pollution': [ar_test_plt_2, ma_test_plt_1, arma_test_plt_2, arima_test_plt_2],
                        'Power Consumption': [ar_test_plt_2, ma_test_plt_1, arma_test_plt_2, arima_test_plt_2],
                        'Traffic Volume': [ar_test_plt_2, ma_test_plt_1, arma_test_plt_2, arima_test_plt_2]})
figure_table(test_prediction, 'Test forecast comparison for 4 Datasets')