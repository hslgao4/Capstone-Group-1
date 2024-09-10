import pandas as pd
import os, sys
utils_dir = '../../component'
sys.path.insert(0, os.path.abspath(utils_dir))
from utils import *
from sklearn.model_selection import train_test_split

df = pd.read_csv('./yahoo_stock.csv')
item_list = ['Adj Close']

# split into train, test, validation and save to csv
seed = 6501

train_df, remain_df = train_test_split(df, train_size=0.7, random_state=seed)
val_df, test_df = train_test_split(remain_df, test_size=0.5, random_state=seed)

train_df.to_csv("train_non_sta.csv", index=False)
test_df.to_csv("test_non_sta.csv", index=False)
val_df.to_csv("val_non_sta.csv", index=False)


def main():
    plt_rolling_mean_var(df, item_list)
    ADF_test(df['Adj Close'])
    print("            ")
    kpss_test(df['Adj Close'])
    Decomposition('./yahoo_stock.csv', 'Adj Close', 'Date')

if __name__ == "__main__":
    main()