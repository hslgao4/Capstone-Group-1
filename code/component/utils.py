import matplotlib.pyplot as plt
import statsmodels
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import pandas as pd
from statsmodels.tsa.seasonal import STL
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
import copy
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.base import BaseEstimator
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from prettytable import PrettyTable
import optuna
import sys
sys.path.append('../component')
from class_ARIMA_model import ARIMA_model
from class_SARIMA_model import SARIMA_Model

'''EDA Part function'''
# plot rolling mean & variance
def plt_rolling_mean_var(df, item_list):
    def rolling_mean_var(df, x):
        rolling_mean = []
        rolling_var = []
        for i in range(1, len(df) + 1):
            new_df = df.iloc[:i, ]
            if i == 1:
                mean = new_df[x].iloc[0]
                var = 0
            else:
                mean = new_df[x].mean()
                var = new_df[x].var()
            rolling_mean.append(mean)
            rolling_var.append(var)
        return rolling_mean, rolling_var

    # Call rolling_mean_var with the item_list as the column name
    roll_mean, roll_var = rolling_mean_var(df, item_list)

    # Plot the results
    plt.figure(figsize=(8, 7))

    # Subplot 1: Rolling Mean
    plt.subplot(2, 1, 1)
    plt.plot(roll_mean, label=item_list, lw=1.5)
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.title('Rolling Mean')
    plt.legend()

    # Subplot 2: Rolling Variance
    plt.subplot(2, 1, 2)
    plt.plot(roll_var, label=item_list, lw=1.5)
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.title('Rolling Variance')
    plt.legend()

    plt.tight_layout()
    plt.show()


# ADF test
def ADF_test(x):
    result = adfuller(x)
    print("ADF Statistic: %f" % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    if result[1] < 0.05:
        print('Pass ADF test.')
    else:
        print('Fail to pass ADF test.')


# KPSS test
def kpss_test(timeseries):
    print('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=[
                            'Test Statistic', 'p-value', 'Lags Used'])
    print('p-value: %f' % kpsstest[1])
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value

    print(kpss_output)

    if kpsstest[1] > 0.05:
        print('Pass KPSS test.')
    else:
        print('Fail to pass KPSS test.')

# Decomposition
def Decomposition(df, item, date, period):
    # df = pd.read_csv(path, parse_dates=[date])
    df.set_index(date, inplace=True)
    stl = STL(df[item], period=period)
    res = stl.fit()
    T = res.trend
    S = res.seasonal
    R = res.resid
    fig = res.plot()
    plt.show()

    df["season"] = S.tolist()
    df["trend"] = T.tolist()

    adjutsed_df = df[item] - df["season"] - df["trend"]
    adjutsed_df.index = df.index

    plt.figure(figsize=(12, 8))
    plt.plot(df[item], label="Original", lw=1.5)
    plt.plot(adjutsed_df, label="Detrend & Season_adj", lw=1.5)
    plt.title("Original vs. Detrend & Season adjusted")
    plt.ylabel(item)
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    # plt.show()

    F = np.maximum(0, 1 - np.var(np.array(R))/np.var(np.array(T+R)))
    print(f'The strength of trend for this data set is {100*F:.2f}%')

    FS = np.maximum(0, 1 - np.var(np.array(R)) / np.var(np.array(S + R)))
    print(f'The strength of seasonality for this data set is  {100*FS:.2f}%')

# differencing
def differencing(df, order, item):
    diff_list = [0] * order
    for i in range(order, len(df)):
        diff = df.loc[i, item] - df.loc[i-order, item]
        diff_list.append(diff)
    return diff_list


# reverse differencing
def rev_diff(value, forecast):
    rev_forecast = []
    for i in range(0, len(forecast)):
        value += forecast[i]
        rev_forecast.append(value)
    return rev_forecast

# ACF, PACF
def ACF_PACF_Plot(y,lags):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(2,1,2)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()

# simple line plot
def plot_time_series(df):
    plt.figure(figsize=(16, 10))
    plt.plot(df[df.columns[0]], df[df.columns[1]])
    plt.xlabel('Date')
    plt.ylabel('Magnitude')
    plt.title(f'{df.columns[1]} over Date')
    plt.xticks(rotation=45)
    plt.show()



# statistics table comparison
def plot_data_statistics(data_list, names=None):
    if names is None:
        names = [f"Data {i + 1}" for i in range(len(data_list))]

    if len(names) != len(data_list):
        raise ValueError("The number of names must match the number of datasets.")

    stats = []
    for data in data_list:
        stats.append([
            f"{np.mean(data):.2f}",
            f"{np.var(data):.2f}",
            f"{np.std(data):.2f}",
            f"{np.percentile(data, 25):.2f}",
            f"{np.median(data):.2f}",
            f"{np.percentile(data, 75):.2f}"
        ])

    fig, ax = plt.subplots(figsize=(12, len(data_list) + 2))
    ax.axis('off')

    table = ax.table(
        cellText=stats,
        rowLabels=names,
        colLabels=['Mean', 'Variance', 'Std Dev', 'Q1', 'Median', 'Q3'],
        loc='center',
        cellLoc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.title("Comparison of Dataset Statistics")
    plt.tight_layout()
    plt.show()

# ACF plot
def plt_ACF(y, lag):
    mean = np.mean(y)
    D = sum((y-mean)**2)
    R = []
    for tao in range(lag+1):
        S = 0
        for t in range(tao, len(y)):
            N = (y[t]-mean)*(y[t-tao]-mean)
            S += N
        r = S/D
        R.append(r)
    R_inv = R[::-1]
    Magnitute = R_inv + R[1:]
    # ACF = plt.figure()
    plt.figure()
    x_values = range(-lag, lag + 1)
    (markers, stemlines, baseline) = plt.stem(x_values, Magnitute, markerfmt='o')
    plt.setp(markers, color = 'red')
    m = 1.96/np.sqrt(len(y))
    plt.axhspan(-m, m, alpha = 0.2, color = 'blue')
    plt.xlabel('Lags')
    plt.ylabel('Magnitute')
    plt.title(f'ACF plot' )
    plt.show()
    # return ACF

def cal_ACF(y, lag):
    mean = np.mean(y)
    D = sum((y-mean)**2)
    R = []
    for tao in range(lag+1):
        S = 0
        for t in range(tao, len(y)):
            N = (y[t]-mean)*(y[t-tao]-mean)
            S += N
        r = S/D
        R.append(r)
    return R

def cal_fi(j, k, ACF):
    if k == 1:
        up = ACF[j+1]
        bottom = ACF[j]
        if bottom == 0:
            fi = "inf"
        else:
            fi = up / bottom
    else:
        Den = []
        for a in range(j, j + k):
            row = []
            for b in range(a - (k - 1), a + 1):
                b = abs(b)
                R = ACF[b]
                row.append(R)
            row = row[::-1]
            Den.append(row)

        Num = copy.deepcopy(Den)
        for i in range(k):
            Num[i][-1] = ACF[j + 1 + i]
        up = np.linalg.det(Num)
        bottom = np.linalg.det(Den)

        if bottom == 0:
            fi = "inf"
        else:
            fi = up / bottom
            if abs(fi) < 0.0000001:
                fi = 0
    return fi

def GPAC_table(y, J=7, K=7):
    ACF = cal_ACF(y, J+K+1)
    temp = np.zeros((J, K - 1))
    for k in range(1, K):
        for j in range(J):
            value = cal_fi(j, k, ACF)
            temp[j][k-1] = value
    table = pd.DataFrame(temp)
    table = table.round(2)
    table.columns = range(1, K)
    plt.figure()
    sns.heatmap(table, annot=True)
    plt.title("Generalized Partial Autocorrelation(GPAC) Table")
    plt.show()
    return table

# metric function
def MSE(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def RMSE(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)

def MAE(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)



# plot
def plt_forecast(df_test, rev_forecast):
    plt.figure(figsize=(16, 10))
    plt.plot(df_test.index, df_test.iloc[:, 0], label='Test Data', color='red')
    plt.plot(df_test.index, rev_forecast, label='Forecast', color='green')
    plt.xlabel('Index')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.title('Test, and Predicted Values')
    plt.show()

def plt_prediction(df_train, rev_pred):
    plt.figure(figsize=(12, 6))
    plt.plot(df_train.index, df_train.iloc[:, 0], label='Train')
    plt.plot(df_train.index, rev_pred, label='Prediction')
    plt.xlabel('Index')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.title('Train vs Predicted values')
    plt.show()

def plt_train_test_forecast(df_train, df_test, rev_forecast):
    plt.figure(figsize=(12, 6))
    plt.plot(df_train.index, df_train.iloc[:, 0], label='Train', color='blue')
    plt.plot(df_test.index, df_test.iloc[:, 0], label='Test', color='red')
    plt.plot(df_test.index, rev_forecast, label='Forecast', color='green')
    plt.xlabel('Index')
    plt.ylabel('Magnitude')
    plt.title('Train vs Test vs Forecast')
    plt.legend()
    plt.show()

def tabel_pretty(df,title):
    x = PrettyTable()
    for i in range(df.shape[0]):
        x.add_row(df.iloc[i,:])
    x.title = title
    x.field_names = df.columns
    x.float_format = '.2'
    x.hrules = 1
    print(x.get_string())


def neg_mean_squared_error(estimator, X, y=None):
    y_pred = estimator.predict(X)
    return -mean_squared_error(X.ravel(), y_pred)

# grid_search with sk-learn
def grid_sklearn(data, param_grid, model, n_splits=5):
    tscv = TimeSeriesSplit(n_splits)

    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grid,
                               cv=tscv,
                               scoring=neg_mean_squared_error,
                               verbose=1)

    grid_search.fit(data)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Lowest metric score: {-grid_search.best_score_}")

    best_order = grid_search.best_params_[list(param_grid.keys())]
    return best_order


# optuna for ARIMA
def ARIMA_objective(trial, data,
                    ar_max=None, ma_max=None, integ_order=None):

    ar_order = trial.suggest_int('AR_order', 1, ar_max) if ar_max is not None else 0
    ma_order = trial.suggest_int('MA_order', 1, ma_max) if ma_max is not None else 0
    inte_order = integ_order if integ_order is not None else 0

    tscv = TimeSeriesSplit(n_splits=5)
    mse_scores = []

    for train_index, test_index in tscv.split(data):
        train, test = data[train_index], data[test_index]
        model = ARIMA_model(AR_order=ar_order, MA_order=ma_order, Inte_order=inte_order)
        model.fit(train)
        predictions = model.predict(test)
        mse = MSE(test, predictions)
        mse_scores.append(mse)

    return np.mean(mse_scores)


def optuna_search_ARIMA(data,
                        ar_max=None, ma_max=None, integ_order=None,
                        objective=ARIMA_objective, n_trials=5):

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, data, ar_max, ma_max, integ_order), n_trials=n_trials)

    print(f"Best parameters: {study.best_params}")
    print(f"Best MSE: {study.best_value}")

    best_order_list = list(study.best_params.keys())

    return study, best_order_list


# optuna SARIMA
def SARIMA_objective(trial, data,
                    ar_max=None, ma_max=None, integ_order=None,
                    ar_s_max=None, ma_s_max=None, integ_s=None):

    ar_order = trial.suggest_int('AR_order', 1, ar_max) if ar_max is not None else 0
    ma_order = trial.suggest_int('MA_order', 1, ma_max) if ma_max is not None else 0
    inte_order = integ_order if integ_order is not None else 0
    ar_s_order = trial.suggest_int('AR_s_order', 1, ar_s_max) if ar_s_max is not None else 0
    ma_s_order = trial.suggest_int('MA_s_order', 1, ma_s_max) if ma_s_max is not None else 0
    inte_s = integ_s if integ_s is not None else 0

    tscv = TimeSeriesSplit(n_splits=5)
    mse_scores = []

    for train_index, test_index in tscv.split(data):
        train, test = data[train_index], data[test_index]
        model = SARIMA_Model(AR_order=ar_order, MA_order=ma_order, Inte_order=inte_order,
                             AR_s=ar_s_order, MA_s=ma_s_order, Seas_s=inte_s)
        model.fit(train)
        predictions = model.predict(test)
        mse = MSE(test, predictions)
        mse_scores.append(mse)

    return np.mean(mse_scores)



def optuna_search_SARIMA(data, ar_max=None, ma_max=None, integ_order=None,
                         ar_s_max=None, ma_s_max=None, integ_s=None,
                         objective=SARIMA_objective, n_trials=5):

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, data, ar_max, ma_max, integ_order,
                                           ar_s_max, ma_s_max, integ_s),
                   n_trials=n_trials)

    print(f"Best parameters: {study.best_params}")
    print(f"Best MSE: {study.best_value}")

    best_order_list = list(study.best_params.keys())

    return study, best_order_list

