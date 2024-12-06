import sys
sys.path.append('../../component')
from utils import *
from class_EDA import EDA
import os
os.getcwd()

def run_eda(path, lag, season_lag):
    eda = EDA(path)
    df = eda.read_data()
    ts_plt = eda.ts_plots(df) # the line plot of the target
    acf = eda.acf_plot(df, lag, Col=1) # the ACF plot of the target
    acf_pacf = eda.acf_pacf_plot(df, lag, Col=1) # the ACF and PACF plot
    rolling_mean_var = eda.rolling_mean_var(df, Col=1) # the rolling mean and variance
    decomposition = eda.decomposition(df, season_lag, Col=1) # the time series decomposition
    eda.stationarity(df) # ADF/KPSS tests to check stationarity
    return ts_plt, acf, acf_pacf, rolling_mean_var, decomposition

