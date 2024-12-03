#%%
import sys
sys.path.append('../../component')
from utils import *
from run_EDA import run_eda
from run_ARIMA import run_arima
import os
os.getcwd()
#%%
'''EDA'''
path = '../../data/power_consumption.csv'
target = 'power_consumption'
ts_plt, acf, acf_pacf, rolling_mean_var, decomposition = run_eda(path, 100, 144)
#%%
df_train, df_test, train, test = prepare_arima_data(path, target)
print('Train shape', train.shape, '\nTest shape', test.shape)
#%%
'''Order determination: domain knowledge vs. Optuna'''
acf_pacf.show() # ACF tail off, PACF cut off at order = 2
pred, fore = run_arima(path, target, 2, 0, 0)
print(f'Train mse:{MSE(train, pred)}')
#%%
study_ar, best_order_list_ar = optuna_search_ARIMA(train, test,
                                                   ar_max=20, ma_max=None, integ_order=None,
                                                   objective=ARIMA_objective, n_trials=20)
#%%
study_ar, best_order_list_ar = optuna_search_ARIMA(train, test,
                                                   ar_max=30, ma_max=None, integ_order=None,
                                                   objective=ARIMA_objective, n_trials=30)
#%%
# MA order = 8
study_ar, best_order_list_ar = optuna_search_ARIMA(train, test,
                                                   ar_max=None, ma_max=20, integ_order=None,
                                                   objective=ARIMA_objective, n_trials=30)
#%%
# ARMA
GPAC_table(train, J=10, K=10) # plot the GPAC table for Raw data: ARMA(2,2)
#%%
study_ar, best_order_list_ar = optuna_search_ARIMA(train, test,
                                                   ar_max=20, ma_max=20, integ_order=None,
                                                   objective=ARIMA_objective, n_trials=30)
#%%
study_ar, best_order_list_ar = optuna_search_ARIMA(train, test,
                                                   ar_max=20, ma_max=20, integ_order=1,
                                                   objective=ARIMA_objective, n_trials=20)
#%%
study_ar, best_order_list_ar = optuna_search_ARIMA(train, test,
                                                   ar_max=20, ma_max=20, integ_order=2,
                                                   objective=ARIMA_objective, n_trials=20)
#%%
learning_rate = 0.001
batch_size = 128 * 20

##################################################################################################################
dataset = 'wea'

'''LSTM'''
model_name = 'lstm'

seq_length = 30
epochs = 350
hidden_size = 2
num_layers = 1


train_loader, test_loader, scaler, actual_test = set_data(path, target, seq_length, batch_size)
model = run_lstm(dataset, train_loader, model_name, epochs, learning_rate, hidden_size, num_layers)
predictions = lstm_eval(model_name, dataset, model, test_loader, scaler)
MSE(actual_test, predictions)