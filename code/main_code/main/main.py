import sys
sys.path.append('../test')
from EDA_plot import EDA_Plot
from result_weather import df_weather_results


weather_path = '../../data/weather.csv'
air_pollution_path = '../../data/air_pollution.csv'
power_path = '../../data/power_consumption.csv'
traffic_path = '../../data/traffic.csv'

def main():
    eda_plt = EDA_Plot(weather_path, air_pollution_path, power_path, traffic_path)
    figure = eda_plt.plot()

    df_weather_results = df_weather_results(weather_path, 'temperature')
    fig1, fig2 = df_weather_results.main()


if __name__ == '__main__':
    main()

