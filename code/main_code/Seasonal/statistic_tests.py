import pandas as pd
import os, sys
utils_dir = '../../component'
sys.path.insert(0, os.path.abspath(utils_dir))
from utils import *


df = pd.read_csv('./hourly_weather.csv')
item_list = ['Temp_C']

def main():
    plt_rolling_mean_var(df, item_list)
    ADF_test(df['Temp_C'])
    print("            ")
    kpss_test(df['Temp_C'])
    Decomposition('./hourly_weather.csv', 'Temp_C', 'date')

if __name__ == "__main__":
    main()