from utils import *

class EDA:
    def __init__(self, path):
        self.path = path

    def read_data(self):
        df = pd.read_csv(self.path, parse_dates=['date'])
        return df

    def ts_plots(self, df):
        fig1 = plot_time_series(df)
        return fig1

    def acf_plot(self, df, lag, Col=1):
        fig2 = plt_ACF(df.iloc[:, Col], lag)
        return fig2

    def acf_pacf_plot(self, df, lag, Col=1):
        fig3 = ACF_PACF_Plot(df[df.columns[Col]], lag)
        return fig3

    def rolling_mean_var(self, df, Col=1):
        item_list = df.columns[Col]
        fig4 = plt_rolling_mean_var(df, item_list)
        return fig4

    def decomposition(self, df, period, Col=1):
        item_list = df.columns[Col]
        fig5 = Decomposition(df, item_list, 'date', period)
        return fig5

    def stationarity(self, df):
        ADF_test(df.iloc[:, 1])
        kpss_test(df.iloc[:, 1])