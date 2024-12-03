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
path = '../../data/air_pollution.csv'
target = 'pollution'
ts_plt, acf, acf_pacf, rolling_mean_var, decomposition = run_eda(path, 100, 24)
#%%
df_train, df_test, train, test = prepare_arima_data(path, target)
print('Train shape', train.shape, '\nTest shape', test.shape)
#%%
study_ar, best_order_list_ar = optuna_search_ARIMA(train, test,
                                                   ar_max=20, ma_max=None, integ_order=None,
                                                   objective=ARIMA_objective, n_trials=30)
#%%
study_ar, best_order_list_ar = optuna_search_ARIMA(train, test,
                                                   ar_max=None, ma_max=20, integ_order=None,
                                                   objective=ARIMA_objective, n_trials=30)
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