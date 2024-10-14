from utils import *

class EDA:
    def __init__(self, path):
        self.path = path

    def read_data(self):
        df = pd.read_csv(self.path, parse_dates=['date'])
        return df

    def plots(self, df, lag): #, period):
        plot_time_series(df)  # line plot
        plt_ACF(df.iloc[:, 1], lag)
        ACF_PACF_Plot(df[df.columns[1]], 100)  # ACF, PACF
        item_list = df.columns[1]
        plt_rolling_mean_var(df, item_list)
        # Decomposition(df, item_list, 'date', period)

    def stationarity(self, df):
        ADF_test(df.iloc[:, 1])
        kpss_test(df.iloc[:, 1])