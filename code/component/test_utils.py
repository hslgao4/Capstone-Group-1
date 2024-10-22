import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL

def plt_rolling_mean_var(df, item_list):
    def rolling_mean_var(df, x):
        rolling_mean = []
        rolling_var = []
        for i in range(1, len(df) + 1):
            new_df = df.iloc[:i, ]
            if i == 1:
                mean = new_df[x].iloc[0]
                var = 0
            else:
                mean = new_df[x].mean()
                var = new_df[x].var()
            rolling_mean.append(mean)
            rolling_var.append(var)
        return rolling_mean, rolling_var

    # Call rolling_mean_var with the item_list as the column name
    roll_mean, roll_var = rolling_mean_var(df, item_list)

    # Plot the results
    fig = plt.figure()

    # Subplot 1: Rolling Mean
    plt.subplot(2, 1, 1)
    plt.plot(roll_mean, label=item_list, lw=1.5)
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.title('Rolling Mean')
    plt.legend()

    # Subplot 2: Rolling Variance
    plt.subplot(2, 1, 2)
    plt.plot(roll_var, label=item_list, lw=1.5)
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.title('Rolling Variance')
    plt.legend()

    plt.tight_layout()
    # plt.show()

    return fig

def plot_time_series(df):
    fig = plt.figure()
    plt.plot(df[df.columns[0]], df[df.columns[1]])
    plt.xlabel('Date')
    plt.ylabel('Magnitude')
    plt.title(f'{df.columns[1]} over Date')
    plt.xticks(rotation=45)
    # plt.show()
    return fig

def plt_ACF(y, lag):
    mean = np.mean(y)
    D = sum((y-mean)**2)
    R = []
    for tao in range(lag+1):
        S = 0
        for t in range(tao, len(y)):
            N = (y[t]-mean)*(y[t-tao]-mean)
            S += N
        r = S/D
        R.append(r)
    R_inv = R[::-1]
    Magnitute = R_inv + R[1:]

    fig = plt.figure()
    x_values = range(-lag, lag + 1)
    (markers, stemlines, baseline) = plt.stem(x_values, Magnitute, markerfmt='o')
    plt.setp(markers, color = 'red')
    m = 1.96/np.sqrt(len(y))
    plt.axhspan(-m, m, alpha = 0.2, color = 'blue')
    plt.xlabel('Lags')
    plt.ylabel('Magnitute')
    plt.title(f'ACF plot' )
    plt.show()
    return fig

def ACF_PACF_Plot(y,lags):
    # acf = sm.tsa.stattools.acf(y, nlags=lags)
    # pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(2,1,2)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    # plt.show()
    return fig

def Decomposition(df, item, date, period):
    new_df = df.copy()

    new_df.set_index(date, inplace=True)
    stl = STL(new_df[item], period=period)
    res = stl.fit()
    # T = res.trend
    # S = res.seasonal
    # R = res.resid
    fig = res.plot()
    plt.xticks(rotation=45)
    # plt.show()
    return fig
#%%
df = pd.read_csv('./code/data/traffic.csv', parse_dates=['date'])
item_list = df.columns[1]

rolling_mean = plt_rolling_mean_var(df, item_list)
timeseries = plot_time_series(df)
acf = plt_ACF(df.iloc[:, 1], 40)
acf_pacf = ACF_PACF_Plot(df.iloc[:, 1],40)
decom = Decomposition(df, item_list, 'date', 24)

#%%
name = ['Sequence plot', 'Rolling mean & var', 'ACF', 'ACF/PACF', 'Decomposition']
figure = [timeseries, rolling_mean, acf, acf_pacf, decom]

plt_df_2 = pd.DataFrame({'Sequence plot': figure,
                       'Rolling mean & var': figure,
                       'ACF/PACF': figure,
                       'Decomposition': figure})

#%%
plt_df = pd.DataFrame({'First row': name,
                       'Second row': figure})

def plot_figures_1(df, title):
    n_rows = 1  # Only one row for the table
    n_cols = df.shape[0]  # Number of figures as columns
    # Create a figure for the table
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, 4))  # Adjust width as needed
    fig.suptitle(title, fontsize=16)
    for i in range(n_cols):
        axs[i].axis('off')  # Hide the axes
        axs[i].set_title(df.iloc[i, 0], fontsize=12)  # Set the title from the DataFrame
        # Show the figure in the respective cell
        axs[i].imshow(df.iloc[i, 1].canvas.buffer_rgba())  # Convert the figure to an image and display it
        axs[i].set_xticks([])  # Remove x ticks
        axs[i].set_yticks([])  # Remove y ticks
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make space for the title
    plt.show()
# Display the figures as a horizontal table with the title
plot_figures_1(plt_df, title='Figures for Data 1')




sequence = [timeseries, timeseries, timeseries]
rolling = [rolling_mean, rolling_mean, rolling_mean]
Acf_pacf = [acf_pacf, acf_pacf, acf_pacf]
Acf = [acf, acf, acf]
decomo = [decom, decom, decom]

data_name = ['Weather', 'Air pollution', 'Power consumption']
plt_df_3 = pd.DataFrame({'Dataset': data_name,
                        'Sequence plot': sequence,
                       'Rolling mean & var': rolling,
                         'ACF': Acf,
                       'ACF/PACF': Acf_pacf,
                       'Decomposition': decomo})




def figure_table(df, title):
    n_rows = df.shape[0]
    n_cols = df.shape[1]

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 4))  # Adjust size as needed
    fig.suptitle(title, fontsize=16)

    # Loop through the DataFrame to plot each figure and display dataset names
    for row in range(n_rows):
        for col in range(n_cols):
            axs[row, col].axis('off')

            if col == 0:
                axs[row, col].text(0.5, 0.5, df.iloc[row, col], fontsize=14, va='center', ha='center',
                                   transform=axs[row, col].transAxes)
            else:
                # Set column titles for figure plots
                if row == 0:
                    axs[row, col].set_title(df.columns[col], fontsize=12)

                axs[row, col].imshow(df.iloc[row, col].canvas.buffer_rgba())
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

figure_table(plt_df_3, 'Figures for Data 1')