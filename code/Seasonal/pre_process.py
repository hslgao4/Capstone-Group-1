import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


### load data, year 2020, every 10 minutes ###
def loaddata(start, end):
    path = 'https://www.bgc-jena.mpg.de/wetter/'
    list = []
    df = pd.DataFrame()
    for year in np.arange(start, end, 1):
        list.append(path+"mpi_roof_"+str(year)+"a.zip")
        list.append(path+"mpi_roof_"+str(year)+"b.zip")
    for url in list:
        df = df.append(pd.read_csv(url, encoding='unicode_escape',
                       parse_dates=True, index_col="Date Time"))
    df.index.name = 'datetime'
    return df

raw_data = loaddata(start=2019, end=2021)
names = ['atmos_p', 'Temp_C', "Temp_K", 'Temp_C_humi', "rel_humi%", "Vapor_p_max", "Vapor_p",
         "Vapor_p_deficit", "spe_humi", "H2O_conc", "air_density", "wind_sp", "wind_sp_max", "wind_direction",
         "rain_depth", "rain_time", "SWDR", "PAR", "max_PAR", "Tlog", "CO2"]
raw_data.columns = names


### fill missing data with Drift method ###
def null_check(df):
    null_value = df.isnull()
    row_null = null_value.any(axis=1)
    rows = df[row_null]
    return rows

def fill_data(df):
    filldf = df.groupby(pd.Grouper(freq='10T')).mean()
    df_null = null_check(filldf)
    print(f"{len(df_null)} rows have been filled")
    # Drift method
    filldf = filldf.interpolate().round(2)
    return filldf

fill_raw = fill_data(raw_data)  # fill 9 rows
fill_raw.to_csv('weather.csv', index=True)   # save to csv file


####################################################################
df = pd.read_csv("weather.csv")
print(f"Shape of raw dataset: {df.shape}")    # (105264, 22)
print(f"NA in the raw dataset: {df.isnull().sum().sum()}")   # NA in the raw dataset: 0

# change date format, remove index, change to hourly-df
date_range = pd.date_range(start="2019-01-01 00:10:00",
                           end="2021-01-01 00:00:00",
                           freq="10T")
df.insert(1, "date", date_range)
df = df.iloc[:, 1:]
df = df.set_index('date')
hourly_df = df.resample("60T").mean()
print(f"Hourly_df shape {hourly_df.shape}")
hourly_df = hourly_df.reset_index()



# Check outliers and plot
def statistics_and_plt(df):
    for i in range(1, df.shape[1]):
        print(f"{df.columns[i]} statistics: \n{df.iloc[:,i].describe()}")
        plt.figure(figsize=(8, 6))
        plt.plot(df['date'], df.iloc[:, i])
        plt.ylabel('Magnitude')
        plt.xlabel('Date')
        plt.title(df.columns[i])
        plt.grid()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

statistics_and_plt(hourly_df)

# Fix wind_speed - change minimum to mean (Average method)
mean_wind = hourly_df[hourly_df["wind_sp"] >= 0]["wind_sp"].mean()
min_wind = hourly_df["wind_sp"].min()
hourly_df["wind_sp"] = hourly_df["wind_sp"].replace(min_wind, mean_wind)

print(f"new wind_sp statistics: \n{hourly_df['wind_sp'].describe()}")
plt.figure(figsize=(8, 6))
plt.plot(hourly_df['date'], hourly_df["wind_sp"])
plt.tight_layout()
plt.ylabel('Magnitude')
plt.xlabel('Date')
plt.title('new wind_sp')
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# change negative CO2 to mean(after dropping negative) - Average method
mean_CO2 = hourly_df[hourly_df["CO2"] >= 0]["CO2"].mean()
hourly_df.loc[hourly_df["CO2"] < 0, "CO2"] = mean_CO2

print(f"new CO2 statistics: \n{hourly_df['CO2'].describe()}")
plt.figure(figsize=(8, 6))
plt.plot(hourly_df['date'], hourly_df["CO2"])
plt.tight_layout()
plt.ylabel('Magnitude')
plt.xlabel('Date')
plt.title('new CO2')
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# change negative max_PAR to the last observation - Naive method
mean_max_PAR = hourly_df[hourly_df["max_PAR"] >= 0]["max_PAR"].mean()
hourly_df.loc[hourly_df["max_PAR"] < 0, "max_PAR"] = mean_max_PAR

for i in range(1, len(hourly_df)):
    if hourly_df.at[i, "max_PAR"] < 0:
        hourly_df.at[i, "max_PAR"] = hourly_df.at[i-1, "max_PAR"]

print(f"new max_PAR statistics: {hourly_df['max_PAR'].describe()}")

plt.figure(figsize=(8, 6))
plt.plot(hourly_df['date'], hourly_df["max_PAR"])
plt.tight_layout()
plt.ylabel('Magnitude')
plt.xlabel('Date')
plt.title('new max_PAR ')
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

hourly_df.to_csv("hourly_weather.csv", index=False)


# split into train, test, validation and save to csv
seed = 6501

df = pd.read_csv('hourly_weather.csv')
train_df, remain_df = train_test_split(df, train_size=0.7, random_state=seed)
val_df, test_df = train_test_split(remain_df, test_size=0.5, random_state=seed)

train_df.to_csv("train_seasonal.csv", index=False)
test_df.to_csv("test_seasonal.csv", index=False)
val_df.to_csv("val_seasonal.csv", index=False)



