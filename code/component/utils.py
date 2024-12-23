import matplotlib.pyplot as plt
import statsmodels
from matplotlib.lines import lineStyles
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import pandas as pd
from statsmodels.tsa.seasonal import STL
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf
import statsmodels.api as sm
import copy
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.base import BaseEstimator
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from prettytable import PrettyTable
import optuna
import sys
from sklearn.preprocessing import MinMaxScaler
from sympy import rotations

sys.path.append('../component')
from class_TS_model_classical import *
from class_SARIMA import *

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

    roll_mean, roll_var = rolling_mean_var(df, item_list)

    fig = plt.figure()

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
    return fig


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
    new_df = df.copy()

    new_df.set_index(date, inplace=True)
    stl = STL(new_df[item], period=period)
    res = stl.fit()
    T = res.trend
    S = res.seasonal
    R = res.resid
    fig = res.plot()
    plt.xticks(rotation=65)

    new_df["season"] = S.tolist()
    new_df["trend"] = T.tolist()

    adjutsed_df = new_df[item] - new_df["season"] - new_df["trend"]
    adjutsed_df.index = new_df.index

    plt.figure(figsize=(12, 8))
    plt.plot(new_df[item], label="Original", lw=1.5)
    plt.plot(adjutsed_df, label="Detrend & Season_adj", lw=1.5)
    plt.title("Original vs. Detrend & Season adjusted")
    plt.ylabel(item)
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()

    F = np.maximum(0, 1 - np.var(np.array(R))/np.var(np.array(T+R)))
    print(f'The strength of trend for this data set is {100*F:.2f}%')

    FS = np.maximum(0, 1 - np.var(np.array(R)) / np.var(np.array(S + R)))
    print(f'The strength of seasonality for this data set is  {100*FS:.2f}%')

    return fig

# differencing
def differencing(df, order, item):
    diff_list = [0] * order
    for i in range(order, len(df)):
        diff = df.iloc[i][item] - df.iloc[i-order][item]
        diff_list.append(diff)
    return diff_list

# metric function
def MSE(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def RMSE(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)

def MAE(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


# reverse differencing
def rev_diff(value, forecast):
    rev_forecast = []
    for i in range(0, len(forecast)):
        value += forecast[i]
        rev_forecast.append(value)
    return rev_forecast

# ACF, PACF
def ACF_PACF_Plot(y,lags):
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(2,1,2)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    return fig

# simple line plot
def plot_time_series(df):
    fig = plt.figure()
    plt.plot(df[df.columns[0]], df[df.columns[1]])
    plt.xlabel('Date')
    plt.ylabel('Magnitude')
    plt.title(f'{df.columns[1]} over Date')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


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

    fig = plt.figure()
    x_values = range(-lag, lag + 1)
    (markers, stemlines, baseline) = plt.stem(x_values, Magnitute, markerfmt='o')
    plt.setp(markers, color = 'red')
    m = 1.96/np.sqrt(len(y))
    plt.axhspan(-m, m, alpha = 0.2, color = 'blue')
    plt.xlabel('Lags')
    plt.ylabel('Magnitute')
    plt.title(f'ACF plot' )
    # plt.show()
    return fig

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
    ACF = acf(y, nlags=J + K + 1)
    temp = np.zeros((J, K - 1))
    for k in range(1, K):
        for j in range(J):
            value = cal_fi(j, k, ACF)
            temp[j][k-1] = value
    table = pd.DataFrame(temp)
    table = table.round(2)
    table.columns = range(1, K)
    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(table, annot=True, annot_kws={"size": 16})
    plt.title("Generalized Partial Autocorrelation(GPAC) Table")
    plt.show()
    return fig

def plt_forecast(prediction, actual, length, model):
    fig = plt.figure()
    plt.plot(prediction[:length], label='Forecast', marker='*')
    plt.plot(actual[:length], label='Actual data', marker='o')
    plt.xlabel('Steps')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.title(f'{model}')
    plt.tight_layout()
    return fig

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


def ARIMA_objective(trial, train, test,
                    ar_max=None, ma_max=None, integ_max=None):

    ar_order = trial.suggest_int('AR_order', 1, ar_max) if ar_max is not None else 0
    ma_order = trial.suggest_int('MA_order', 1, ma_max) if ma_max is not None else 0
    inte_order = trial.suggest_int('Inte_order', 1, integ_max) if integ_max is not None else 0

    model = ARIMA_model(AR_order=ar_order, MA_order=ma_order, Inte_order=inte_order)
    model.fit(train)
    forecast = model.forecast(len(test))
    mse = MSE(test, forecast)

    return mse

def optuna_search_ARIMA(train, test,
                        ar_max=None, ma_max=None, integ_max=None,
                        objective=ARIMA_objective, n_trials=5):

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train, test, ar_max, ma_max, integ_max), n_trials=n_trials)

    print(f"Best parameters: {study.best_params}")
    print(f"Best MSE: {study.best_value}")

    best_order_list = list(study.best_params.keys())

    return study, best_order_list


# optuna SARIMA
def SARIMA_objective(trial, data,
                    ar_max=None, ma_max=None, integ_max=None,
                    ar_s_max=None, ma_s_max=None, integ_s=None):

    ar_order = trial.suggest_int('AR_order', 1, ar_max) if ar_max is not None else 0
    ma_order = trial.suggest_int('MA_order', 1, ma_max) if ma_max is not None else 0
    inte_order = trial.suggest_int('Inte_order', 1, integ_max) if integ_max is not None else 0

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


def cus_grid_search_ar(data, test, min_order, max_order):
    mse_test = []

    final_order = min_order
    best_mse_test = float('inf')

    for order in range(min_order, max_order + 1):
        model_ar = ARIMA_model(AR_order=order)
        model_ar.fit(data)

        forecast_test = model_ar.forecast(steps=len(test))
        current_mse_test = round(MSE(test, forecast_test), 4)
        mse_test.append((order, current_mse_test))


        if current_mse_test < best_mse_test :
            best_mse_test = current_mse_test
            final_order = order

    return final_order, mse_test

def cus_grid_search_ma(data, test, min_order, max_order):
    mse_test = []
    final_order = min_order
    best_mse_test = float('inf')


    for order in range(min_order, max_order + 1):
        model_ar = ARIMA_model(AR_order=0, MA_order=order)
        model_ar.fit(data)

        forecast_test = model_ar.forecast(steps=len(test))
        current_mse_test = MSE(test, forecast_test)
        mse_test.append((order, current_mse_test))


        if current_mse_test < best_mse_test:
            best_mse_test = current_mse_test
            final_order = order

    return final_order, mse_test


def cal_err(y_pred, Y_test):
    error = []
    error_se = []
    for i in range(len(y_pred)):
        e = Y_test[i] - y_pred[i]
        error.append(e)
        error_se.append(e**2)
    error_mse = np.mean(error_se)
    return error, error_mse


def figure_table(df):
    n_rows = df.shape[0]
    n_cols = df.shape[1]

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 4))


    for row in range(n_rows):
        for col in range(n_cols):
            axs[row, col].axis('off')

            if col == 0:
                axs[row, col].text(0.5, 0.5, df.iloc[row, col], fontsize=14, fontweight='bold', va='center',
                                   ha='center',
                                   transform=axs[row, col].transAxes)
            else:
                if row == 0:
                    axs[row, col].set_title(df.columns[col], fontsize=12, fontweight='bold')

                fig_to_display = df.iloc[row, col]
                fig_to_display.canvas.draw()

                axs[row, col].imshow(fig_to_display.canvas.buffer_rgba())
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def prepare_arima_data(path, target):
    df = pd.read_csv(path, parse_dates=['date'])

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

    return df_train, df_test, train, test

def plot_metric_table(df, title):
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.show()


def sliding_windows(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        _x = data[i:(i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)
    return x, y

'''Transformer functions'''
# prepare transformer data
def pre_transformer_data(path, target, seq_length):
    df = pd.read_csv(path)
    train_size = int(df.shape[0] * 0.8)
    df_train = df[:train_size]
    df_test = df[train_size:]

    data_train = df_train[target].to_numpy().reshape(-1, 1)
    data_test = df_test[target].to_numpy().reshape(-1, 1)

    scaler = StandardScaler()
    data_train_t = scaler.fit_transform(data_train).flatten().tolist()
    data_test_t = scaler.transform(data_test).flatten().tolist()

    x, y = sliding_windows(data_train_t, seq_length)
    x_train, y_train = torch.tensor(x, dtype=torch.float32).view(-1, seq_length, 1), torch.tensor(y, dtype=torch.float32).view(-1, 1)

    x, y = sliding_windows(data_test_t, seq_length)
    x_test, y_test = torch.tensor(x, dtype=torch.float32).view(-1, seq_length, 1), torch.tensor(y, dtype=torch.float32).view(-1, 1)

    return x_train, y_train, x_test, y_test, scaler

def set_dataloader(x_train, y_train, shuffle=True, batch_size=32, num_workers=6):
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return train_loader

def transformer_train(model, train_loader, optimizer, criterion, epochs, device):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.8f}')

    return model


'''LSTM functions'''
def pre_lstm_data(path, target, seq_length):
    training_set = pd.read_csv(path)

    data = training_set[target].values.astype(float)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data.reshape(-1, 1))

    print('seq_length:', seq_length)

    X, y = sliding_windows(data, seq_length)
    X, y = np.array(X), np.array(y)

    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    return X_train, y_train, X_test, y_test, scaler

def lstm_train(name, model, train_loader, optimizer, criterion, epochs, device):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.8f}')
    torch.save(model.state_dict(), f'{name}_weights.pt')
    return model


def calculate_metrics(predictions, actuals):
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Normalize the data
    actuals_min, actuals_max = np.min(actuals), np.max(actuals)
    predictions = (predictions - actuals_min) / (actuals_max - actuals_min)
    actuals = (actuals - actuals_min) / (actuals_max - actuals_min)

    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    print(f'MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}')

    return mse, rmse, mae