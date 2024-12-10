import pandas as pd

'''Check shape, columns, any missing value. Save only data + target to a new dataframe'''

# process air_pollution.csv
air_pollution = pd.read_csv('./raw_data/air_pollution.csv')
print(air_pollution.shape)
print(air_pollution.columns)
print(air_pollution.isnull().sum())

df = air_pollution.iloc[:, 0:2]
df.to_csv('./air_pollution.csv', index=False)


# process power_consumption.csv
power = pd.read_csv('./raw_data/power_consumption.csv')
print(power.shape)
print(power.columns)
print(power.isnull().sum())

df = power.loc[:, ['DateTime', 'Zone 1 Power Consumption']]
df.columns = ['date', 'power_consumption']
df.to_csv('./power_consumption.csv', index=False)

# process weather.csv
weather = pd.read_csv('./raw_data/weather.csv')
print(weather.shape)
print(weather.columns)
print(weather.isnull().sum())

df = weather.loc[:, ['datetime', 'Temp_C']]
df.columns = ['date', 'temperature']
df.to_csv('./weather.csv', index=False)