import sys
sys.path.append('../../component')
from utils import *
from class_EDA import EDA
import os
os.getcwd()


weather_path = '../../data/weather.csv'
eda = EDA(weather_path)
df1 = eda.read_data()
seq_1 = eda.ts_plots(df1)
acf_1 = eda.acf_plot(df1, 100, Col=1)
pacf_1 = eda.acf_pacf_plot(df1, 80, Col=1)
rolling_1 = eda.rolling_mean_var(df1, Col=1)
decom_1 = eda.decomposition(df1, 144, Col=1)
eda.stationarity(df1)




air_pollution_path = '../../data/air_pollution.csv'
eda = EDA(air_pollution_path)
df2 = eda.read_data()
seq_2 = eda.ts_plots(df2)
acf_2 = eda.acf_plot(df2, 100, Col=1)
pacf_2 = eda.acf_pacf_plot(df2, 80, Col=1)
rolling_2 = eda.rolling_mean_var(df2, Col=1)
decom_2 = eda.decomposition(df2, 24, Col=1)
eda.stationarity(df2)



power_path = '../../data/power_consumption.csv'
eda = EDA(power_path)
df3 = eda.read_data()
seq_3 = eda.ts_plots(df3)
acf_3 = eda.acf_plot(df3, 100, Col=1)
pacf_3 = eda.acf_pacf_plot(df3, 80, Col=1)
rolling_3 = eda.rolling_mean_var(df3, Col=1)
decom_3 = eda.decomposition(df3, 144, Col=1)
eda.stationarity(df3)



traffic_path = '../../data/traffic.csv'
eda = EDA(traffic_path)
df_temp = eda.read_data()
temp = df_temp.sort_values('date')
df4 = temp.groupby('date', as_index=False)['vehicles'].sum()
seq_4 = eda.ts_plots(df4)
acf_4 = eda.acf_plot(df4, 100, Col=1)
pacf_4 = eda.acf_pacf_plot(df4, 80, Col=1)
rolling_4 = eda.rolling_mean_var(df4, Col=1)
decom_4 = eda.decomposition(df4, 24, Col=1)
eda.stationarity(df4)




datasets = pd.DataFrame({'Dataset': ['Temperature', 'Air Pollution', 'Power Consumption', 'Traffic Volume'],
                         'Sequence plots': [seq_1, seq_2, seq_3, seq_4],
                         'Rolling mean & var': [rolling_1, rolling_2, rolling_3, rolling_4],
                         'ACF plots': [acf_1, acf_2, acf_3, acf_4],
                         'ACF/PACF plots': [pacf_1, pacf_2, pacf_3, pacf_4],
                         'Decompostion plots': [decom_1, decom_2, decom_3, decom_4]})

figure_table(datasets, 'EDA plots of four Datasets')