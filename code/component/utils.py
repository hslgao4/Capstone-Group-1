import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import pandas as pd
from statsmodels.tsa.seasonal import STL
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

'''EDA Part function'''
# plot rolling mean & variance
def plt_rolling_mean_var(df, item_list):
    def rolling_mean_var(df, x):
        rolling_mean = []
        rolling_var = []
        for i in range(1, len(df) + 1):
            new_df = df.iloc[:i, ]
            if i == 1:
                mean = new_df[x]
                var = 0
            else:
                mean = new_df[x].mean()
                var = new_df[x].var()
            rolling_mean.append(mean)
            rolling_var.append(var)
        return rolling_mean, rolling_var
    plt.figure(figsize=(8, 7))
    plt.subplot(2, 1, 1)
    for i in item_list:
        roll_mean, roll_var = rolling_mean_var(df, i)

        plt.plot(roll_mean, label=i, lw=1.5)
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.title('Rolling Mean')
    plt.legend()

    plt.subplot(2, 1, 2)
    for i in item_list:
        roll_mean, roll_var = rolling_mean_var(df, i)

        plt.plot(roll_var, label=i, lw=1.5)
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
    # print('Critical Values:')
    # for key, value in result[4].items():
    #     print('\t%s: %.3f' % (key, value))
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
    # for key, value in kpsstest[3].items():
    #     kpss_output['Critical Value (%s)' % key] = value
    #
    # print(kpss_output)

    if kpsstest[1] > 0.05:
        print('Pass KPSS test.')
    else:
        print('Fail to pass KPSS test.')

# Decomposition
def Decomposition(path, item, date):
    df = pd.read_csv(path, parse_dates=[date])
    df.set_index(date, inplace=True)
    stl = STL(df[item], period=24)
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
def plot_time_series(ts):
    plt.figure(figsize=(10, 8))
    plt.plot(ts)
    plt.xlabel('Date')
    plt.ylabel('Magnitude')
    plt.title('Genrated Time Series data over Date')
    plt.xticks(rotation=45)
    plt.show()




'''Data generator function'''
# generate data using SARIMAX, which can handle AR, MA, ARMA, ARIMA, SARIMA, and multiplicative
def generate_sarima_data(sample_size=5000,
                         ar_para=[0.6], ma_para=[-0.4],
                         sar_para=[0.5], sma_para=[-0.3],
                         d=1, D=1, seasonal_period=12, var_WN=1):
    std = np.sqrt(var_WN)

    # Set up SARIMAX model for SARIMA (no exogenous inputs)
    sarimax_model = sm.tsa.SARIMAX(endog=np.zeros(sample_size),  # Dummy endog to initialize the model
                                   order=(len(ar_para), d, len(ma_para)),  # Non-seasonal ARIMA order
                                   seasonal_order=(len(sar_para), D, len(sma_para), seasonal_period),  # Seasonal ARIMA order
                                   trend='n',  # No trend
                                   enforce_stationarity=False,  # Relax stationarity constraint for simulation
                                   enforce_invertibility=False)  # Relax invertibility constraint

    params = np.r_[ar_para, ma_para, sar_para, sma_para, std]
    simulated_data = sarimax_model.simulate(params=params, nsimulations=sample_size, measurement_shock_scale=std)

    return simulated_data



def gen_arma(sample=5000, ar_para=[0.6, -0.3], ma_para=[0.1, 0.2], var_WN=1):
    std = np.sqrt(var_WN)
    arparams = np.array(ar_para)
    maparams = np.array(ma_para)
    ar = np.r_[1, arparams]
    ma = np.r_[1, maparams]
    arma_process = sm.tsa.ArmaProcess(ar, ma)
    arma_data = arma_process.generate_sample(sample, scale=std)
    return arma_data



# deterministic approach to generate data
def gen_deterministic(samples=5000, custom_func=None, type='random', **kwargs):
    if custom_func is not None:
        return custom_func(samples, **kwargs)
    if type == 'random':
        mean = kwargs.get('mean', 0)
        std = kwargs.get('std', 1)
        return np.random.normal(mean, std, samples)
    elif type == 'sine':
        period = kwargs.get('period', 365)
        amplitude = kwargs.get('amplitude', 1)
        return amplitude * np.sin(2 * np.pi * np.arange(samples) / period)
    elif type == 'cosine':
        period = kwargs.get('period', 365)
        amplitude = kwargs.get('amplitude', 1)
        return amplitude * np.cos(2 * np.pi * np.arange(samples) / period)
    elif type == 'linear':
        slope = kwargs.get('slope', 0.1)
        return slope * np.arange(samples)
    elif type == 'exponential':
        rate = kwargs.get('rate', 0.01)
        return np.exp(rate * np.arange(samples))
    else:
        raise ValueError("Invalid input type")

# deterministic approach to add seasonality to data
def seasonality_determ(data, period_s=365, amplitude=1):
    samples = data.shape[0]
    seasonal = amplitude * np.sin(2 * np.pi * np.arange(samples) / period_s)
    seasonal_data = data + seasonal
    # seasonal_data = seasonal_data.round(2)
    return seasonal_data


# deterministic approach to add trend to data
def trend_determ(data, trend_type='linear', **kwargs):
    samples = data.shape[0]
    if trend_type == 'linear':
        slope = kwargs.get('slope', 0.1)
        trend = slope * np.arange(samples)
    elif trend_type == 'exponential':
        rate = kwargs.get('rate', 0.001)
        trend = np.exp(rate * np.arange(samples))
    else:
        raise ValueError("Invalid trend type")
    trended_data = data + trend
    # trended_data = trended_data.round(2)
    return trended_data