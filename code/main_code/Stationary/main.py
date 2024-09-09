import pandas as pd
import os, sys
utils_dir = '../../component'
sys.path.insert(0, os.path.abspath(utils_dir))
from utils import *


df = pd.read_csv('./AR_1_data.csv')
item_list = ['data']

def main():
    plt_rolling_mean_var(df, item_list)
    ADF_test(df['data'])
    print("            ")
    kpss_test(df['data'])

if __name__ == "__main__":
    main()
