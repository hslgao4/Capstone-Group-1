import sys
import pandas as pd
from scipy.special.tests.test_boost_ufuncs import test_data

sys.path.append('../component')  # Ensure sys is imported before using it
from utils import *
from class_LSTM import LSTM
import os
os.getcwd()

# prepare the data for the model
path = '../data/weather.csv'
target = 'temperature'
df_train, df_test, df_val, train, test, validation = prepare_data(path, target)
#%%
'''AR model'''
# based on ACF/PACF shows a clear AR pattern with order 1 or 2, AR(2) has low MSE
ar_train_err_1, ar_test_err_1, ar_train_err_mse_1, ar_test_err_mse_1, ar_train_plt_1, ar_test_plt_1 = ARIMA_results(2, 0, 0, df_train, df_test, train, test, validation)

# optuna give order AR(4)
ar_train_err_2, ar_test_err_2, ar_train_err_mse_2, ar_test_err_mse_2, ar_train_plt_2, ar_test_plt_2 = ARIMA_results(4, 0, 0, df_train, df_test, train, test, validation)

ar_train_err_acf = plt_ACF(ar_train_err_2, 40)
ar_test_err_acf = plt_ACF(ar_test_err_2, 40)

'''MA model'''
# no MA pattern in ACF/PACF, so just use optuna MA(10)
ma_train_err, ma_test_err, ma_train_err_mse, ma_test_err_mse, ma_train_plt_1, ma_test_plt_1 = ARIMA_results(10, 0, 0, df_train, df_test, train, test, validation)

ma_train_err_acf = plt_ACF(ma_train_err, 40)
ma_test_err_acf = plt_ACF(ma_test_err, 40)

'''ARMA model'''
# GPAC: ARMA(2,3)
arma_train_err_1, arma_test_err_1, arma_train_err_mse_1, arma_test_err_mse_1, arma_train_plt_1, arma_test_plt_1 = ARIMA_results(2, 3, 0, df_train, df_test, train, test, validation)

# Optuna(6,6)
arma_train_err_2, arma_test_err_2, arma_train_err_mse_2, arma_test_err_mse_2, arma_train_plt_2, arma_test_plt_2 = ARIMA_results(6, 6, 0, df_train, df_test, train, test, validation)

arma_train_err_acf = plt_ACF(arma_train_err_2, 40)
arma_test_err_acf = plt_ACF(arma_test_err_2, 40)

'''ARIMA model'''
# GPAC ARIMA(2,1,3)
arima_train_err_1, arima_test_err_1, arima_train_err_mse_1, arima_test_err_mse_1, arima_train_plt_1, arima_test_plt_1 = ARIMA_results(2, 3, 1, df_train, df_test, train, test, validation)

# Optuna ARIMA(6, 1, 0)
arima_train_err_2, arima_test_err_2, arima_train_err_mse_2, arima_test_err_mse_2, arima_train_plt_2, arima_test_plt_2 = ARIMA_results(6, 0, 1, df_train, df_test, train, test, validation)

arima_train_err_acf = plt_ACF(arima_train_err_2, 40)
arima_test_err_acf = plt_ACF(arima_test_err_2, 40)

'''LSTM'''
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

input_size = 1
hidden_size = 2
num_layers = 1
num_classes = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lstm = LSTM(num_classes, input_size, hidden_size, num_layers).to(device)
criterion = nn.MSELoss()
lstm.load_state_dict(torch.load('LSTM_model_weights.pt'))

lstm.eval()
with torch.no_grad():
    train_predict = lstm(trainX)
    test_predict = lstm(test_data)

lstm_train_loss = criterion(train_predict, trainY)
lstm_test_loss = criterion(test_predict, testY)

test_pred = test_predict.cpu().numpy()
test_actual = testY.cpu().numpy()
test_pred = sc.inverse_transform(test_pred)
test_actual = sc.inverse_transform(test_actual)

train_pred = train_predict.cpu().numpy()
train_actual = trainY.cpu().numpy()
train_pred = sc.inverse_transform(train_pred)
train_actual = sc.inverse_transform(train_actual)

lstm_train_plt = plt_forecast(train_actual, train_pred)
lstm_test_plt = plt_forecast(test_actual, test_pred)








'''Single dataset model performance'''
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