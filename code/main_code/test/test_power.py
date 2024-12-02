import sys
sys.path.append('../../component')
from utils import *
from run_EDA import run_eda
from run_ARIMA import run_arima
import os
os.getcwd()


'''EDA'''
#%%
path = '../../data/power_consumption.csv'
target = 'power_consumption'
eda = EDA(path)
df = eda.read_data()

ts_plt, acf, acf_pacf, rolling_mean_var, decomposition = run_eda(path, 100, 144)

#%% Prepare data for ARIMA model
df_train, df_test, train, test = prepare_arima_data(path, target)
print('Train shape', train.shape, '\nTest shape', test.shape)

'''Order determination: domain knowledge vs. Optuna'''
acf_pacf.show() # ACF tail off, PACF cut off at order = 2
pred, fore = run_arima(path, target, 2, 0, 0)
print(f'AR mse:{MSE(fore, pred)}')




