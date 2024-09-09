import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import pandas as pd
from statsmodels.tsa.seasonal import STL
import numpy as np

# Function_1: plot rolling mean & variance
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


# Function_2: ADF test
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


# Function_3: KPSS test
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

# Function 4: Decomposition
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
    plt.show()

    F = np.maximum(0, 1 - np.var(np.array(R))/np.var(np.array(T+R)))
    print(f'The strength of trend for this data set is {100*F:.2f}%')

    FS = np.maximum(0, 1 - np.var(np.array(R)) / np.var(np.array(S + R)))
    print(f'The strength of seasonality for this data set is  {100*FS:.2f}%')