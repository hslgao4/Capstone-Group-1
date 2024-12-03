#%%
import sys

import pandas as pd

sys.path.append('../../component')
from utils import *
from run_EDA import run_eda
from run_ARIMA import run_arima
import os
os.getcwd()

#%%
'''EDA'''
path = '../../data/traffic.csv'
target = 'pollution'
# ts_plt, acf, acf_pacf, rolling_mean_var, decomposition = run_eda(path, 100, 24)
df = pd.read_csv(path, parse_dates=['date'])
item_list = df.columns[1]
fig = plt_rolling_mean_var(df, item_list)


#%%
diff_list = differencing(df, 1, 'vehicles')
df['diff_vehicles'] = diff_list
item_list = df.columns[2]
fig2 = plt_rolling_mean_var(df, item_list)

#%%
target = 'diff_vehicles'
train_size = int(df.shape[0] * 0.8)
df_train = df[:train_size]
df_test = df[train_size:]

df_train = df_train.copy()
df_train['date'] = pd.to_datetime(df_train['date'])
df_train.set_index('date', inplace=True)

df_test = df_test.copy()
df_test['date'] = pd.to_datetime(df_test['date'])
df_test.set_index('date', inplace=True)

train = df_train[target].values.reshape(-1, 1)
test = df_test[target].values.reshape(-1, 1)
print('Train shape', train.shape, '\nTest shape', test.shape)

#%%
study_ar, best_order_list_ar = optuna_search_ARIMA(train, test,
                                                   ar_max=20, ma_max=None, integ_order=None,
                                                   objective=ARIMA_objective, n_trials=20)
# AR=20


#%%
study_ar, best_order_list_ar = optuna_search_ARIMA(train, test,
                                                   ar_max=None, ma_max=20, integ_order=None,
                                                   objective=ARIMA_objective, n_trials=20)

# MA  = 4


#%%
study_ar, best_order_list_ar = optuna_search_ARIMA(train, test,
                                                   ar_max=20, ma_max=20, integ_order=None,
                                                   objective=ARIMA_objective, n_trials=20)

# (16, 20)

study_ar, best_order_list_ar = optuna_search_ARIMA(train, test,
                                                   ar_max=20, ma_max=20, integ_order=1,
                                                   objective=ARIMA_objective, n_trials=20)

# ARIMA(19,1,19)