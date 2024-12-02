#%%
import sys
sys.path.append('../../component')
import pandas as pd
from utils import *
from class_TS_model_classical import ARIMA_model, SARIMA_Model
from class_EDA import *
import os
os.getcwd()

'''EDA'''
#%%
path = '../../data/weather.csv'
eda = EDA(path)
df = eda.read_data()

ts_plt, acf, acf_pacf, rolling_mean_var, decomposition = run_eda(path, 100, 144)


'''Prepare the data for the model: 80% train, 20% test'''
#%%
df = pd.read_csv('../../data/weather.csv', parse_dates=['date'])

train_size = int(df.shape[0] * 0.8)
df_train = df[:train_size]
df_test = df[train_size:]

df_train['date'] = pd.to_datetime(df_train['date'])
df_train.set_index('date', inplace=True)

df_test['date'] = pd.to_datetime(df_test['date'])
df_test.set_index('date', inplace=True)

train = df_train['temperature'].values.reshape(-1, 1)
test = df_test['temperature'].values.reshape(-1, 1)

print('Train shape', train.shape, '\nTest shape', test.shape)

'''Order determination: domain knowledge vs. Optuna'''
#%%
acf_pacf.show() # check the ACF/PACF plot to find pattern

"AR model - domain knowledge"
# ACF/PACF plot shows clear AR patter with ACF tail off and PACF cut off, statistical significance at order 1 and 2.
#%%
# First try AR(1), calculate the MSE
ar_1_d = ARIMA_model(AR_order=1)
ar_1_d.fit(train)

prediction = ar_1_d.predict(train)
mse_pred_ar_1_d = MSE(train, prediction)
print('train prediction MSE:', round(mse_pred_ar_1_d,4))

forecast = ar_1_d.forecast(len(test))
mse_fore_ar_1_d = MSE(test, forecast)
print('test forecast MSE:', round(mse_fore_ar_1_d, 4))

#%%
# try AR(2), calculate the MSE
ar_2_d = ARIMA_model(AR_order=2)
ar_2_d.fit(train)

prediction = ar_2_d.predict(train)
mse_pred_ar_2_d = MSE(train, prediction)
print('train prediction MSE:', round(mse_pred_ar_2_d,4))

forecast = ar_2_d.forecast(len(test))
mse_fore_ar_2_d = MSE(test, forecast)
print('test forecast MSE:', round(mse_fore_ar_2_d, 4))


"AR model - Optuna"
#%%
# final_order, mse_test = cus_grid_search_ar(train, test, 1, 10)
study_ar, best_order_list_ar = optuna_search_ARIMA(train,
                                                   ar_max=20, ma_max=None, integ_order=None,
                                                   objective=ARIMA_objective, n_trials=20)

print(best_order_list_ar)

#%%
# the final order with grid search would be 4
ar_n_o = ARIMA_model(AR_order=4)
ar_n_o.fit(train)

prediction = ar_n_o.predict(train)
mse_pred_ar_n_o = MSE(train, prediction)
print('train prediction MSE:', round(mse_pred_ar_n_o,4))

forecast = ar_n_o.forecast(len(test))
mse_fore_ar_n_o = MSE(test, forecast)
print('test forecast MSE:', round(mse_fore_ar_n_o, 4))


#%% The best AR order is xxx using Optuna
AR_model = ARIMA_model(AR_order=1000)
AR_model.fit(train)
prediction_ar = AR_model.predict(train)
forecast_ar = AR_model.forecast(len(test))
#%%
plt_prediction(df_train[:100], prediction_ar[:100])   # train vs prediction
plt_forecast(df_test, forecast_ar)   # test vs forecast
plt_train_test_forecast(df_train, df_test, forecast_ar)   # train, test, forecast

#%% prediction error
error, error_mean, error_var, error_mse = cal_err(prediction_ar.tolist(), train[:, 0].tolist())
plt_ACF(error, 40)
#%% forecast error
error, error_mean, error_var, error_mse = cal_err(forecast_ar.tolist(), test[:, 0].tolist())
plt_ACF(error, 40)







"MA model - Optuna"
#%% Since the ACF/PACF shows no MA pattern, for MA order, only use Optuna, set the max MA order to be 10
study_ma, best_order_list_ma = optuna_search_ARIMA(train, test,
                                                   ar_max=None, ma_max=10, integ_order=None,
                                                   objective=ARIMA_objective, n_trials=15)
#%%
print(best_order_list_ma)
#%%
model_ma = ARIMA_model(AR_order=0, MA_order=10000)
model_ma.fit(train)

prediction_ma = model_ma.predict(train)
mse_pred_ma = MSE(train, prediction)
print('train prediction MSE:', round(mse_pred_ma,4))

forecast_ma = model_ma.forecast(len(test))
mse_fore_ma = MSE(test, forecast)
print('test forecast MSE:', round(mse_fore_ma, 4))

#%%
plt_prediction(df_train[:100], prediction_ma[:100])
plt_forecast(df_test, forecast_ma)
plt_train_test_forecast(df_train, df_test, forecast_ma)

#%% prediction error
error, error_mean, error_var, error_mse = cal_err(prediction_ma.tolist(), train[:, 0].tolist())
plt_ACF(error, 40)

#%% forecast error
error, error_mean, error_var, error_mse = cal_err(forecast_ma.tolist(), test[:, 0].tolist())
plt_ACF(error, 40)






"ARMA model - domain knowledge"
#%%
GPAC_table(train, J=10, K=10) # plot the GPAC table for Raw data
# The GPAC shows a order of ARMA(2,3)

model_arma_d = ARIMA_model(AR_order=2, MA_order=3)
model_arma_d.fit(train)

prediction = model_arma_d.predict(train)
mse_pred_arma_d = MSE(train, prediction)
print('prediction error:', round(mse_pred_arma_d,4))

forecast = model_arma_d.forecast(len(test))
mse_fore_arma_d = MSE(test, forecast)
print('test error:', round(mse_fore_arma_d,4))


"ARMA model - Optuna"
#%%
study_arma, best_order_list_arma = optuna_search_ARIMA(train, test,
                                                       ar_max=15, ma_max=15, integ_order=0,
                                                       objective=ARIMA_objective, n_trials=40)
print(best_order_list_arma)

 #%%
model_arma_o = ARIMA_model(AR_order=1000, MA_order=10000)
model_arma_o.fit(train)

prediction = model_arma_o.predict(train)
mse_pred_arma_o = MSE(train, prediction)
print('train prediction MSE:', round(mse_pred_arma_o,4))

forecast = model_arma_o.forecast(len(test))
mse_fore_arma = MSE(test, forecast)
print('test forecast MSE:', round(mse_fore_arma, 4))


#%% best ARMA
model_ARMA = ARIMA_model(AR_order=10000, MA_order=100000)
model_ARMA.fit(train)

prediction_arma = model_ARMA.predict(train)
mse_pred_ARMA = MSE(train, prediction_arma)
print('prediction error:', round(mse_pred_ARMA,4))

forecast_arma = model_ARMA.forecast(len(test))
mse_fore_arma = MSE(test, forecast_arma)
print('forecast error:', round(mse_fore_arma, 4))

#%%
plt_prediction(df_train[:100], prediction_arma[:100])
plt_forecast(df_test, forecast_arma)
plt_train_test_forecast(df_train, df_test, forecast_arma)

#%% prediction error
error, error_mean, error_var, error_mse = cal_err(prediction_arma.tolist(), train[:, 0].tolist())
plt_ACF(error, 40)

#%% forecast error
error, error_mean, error_var, error_mse = cal_err(forecast_arma.tolist(), test[:, 0].tolist())
plt_ACF(error, 40)


"ARIMA model - domain knowledge"
#%% perform first differencing
train_diff = df_train.copy()
diff_1st = differencing(train_diff, 1, 'temperature')
#%%
GPAC_table(diff_1st, J=10, K=10)
#%%
# The GPAC shows a order of AR=2, MA=3
model_arima_d_1 = ARIMA_model(AR_order=1, MA_order=2, Inte_order=1)
model_arima_d_1.fit(train)

prediction = model_arima_d_1.predict(train)
mse_pred_arima_d_1 = MSE(train, prediction)
print('prediction error:', round(mse_pred_arima_d_1,4))

forecast = model_arima_d_1.forecast(len(test))
mse_fore_arima_d_1 = MSE(test, forecast)
print('test error:', round(mse_fore_arima_d_1,4))


# try second order differencing
train_diff = df_train.copy()
diff_2nd = differencing(train_diff, 2, 'temperature')
#%%
GPAC_table(diff_2nd, J=10, K=10)
#%%
# The GPAC shows a order of AR=, MA=
model_arima_d_2 = ARIMA_model(AR_order=1, MA_order=2, Inte_order=1)
model_arima_d_2.fit(train)

prediction = model_arima_d_2.predict(train)
mse_pred_arima_d_2 = MSE(train, prediction)
print('prediction error:', round(mse_pred_arima_d_2,4))

forecast = model_arima_d_2.forecast(len(test))
mse_fore_arima_d_2 = MSE(test, forecast)
print('test error:', round(mse_fore_arima_d_2,4))

