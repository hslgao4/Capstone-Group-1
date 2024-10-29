from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.tsa.holtwinters as ets
import numpy as np

weather_path = '../../data/weather.csv'
df = pd.read_csv(weather_path, parse_dates=['date'])

# Raw - add
y = df["temperature"]
yt, yf = train_test_split(y, shuffle=False, test_size=0.2)

holt_t = ets.ExponentialSmoothing(yt, trend='add', seasonal='add', seasonal_periods=144).fit()
holt_f = holt_t.forecast(steps=len(yf))
holt_f = pd.DataFrame(holt_f).set_index(yf.index)
MSE = np.square(np.subtract(yf.values,np.ndarray.flatten(holt_f.values))).mean()
print("Mean square error for holt-winter method is ", MSE)

# Raw - multiplicative
y = df["temperature"] + 12
yt, yf = train_test_split(y, shuffle=False, test_size=0.2)

holt_t = ets.ExponentialSmoothing(yt, trend='mul', seasonal='mul', seasonal_periods=144).fit()
holt_f = holt_t.forecast(steps=len(yf))
holt_f = pd.DataFrame(holt_f).set_index(yf.index)
MSE = np.square(np.subtract(yf.values,np.ndarray.flatten(holt_f.values))).mean()
print("Mean square error for holt-winter method is ", MSE)
holt_f = holt_f -12

plt.figure(figsize=(8, 6))
plt.plot(yt,label= "train")
plt.plot(yf,label= "test")
plt.plot(holt_f,label= "Holt-Winter Method")
plt.legend()
plt.title('Holt-Winter Method - Raw(add)-720')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Temperature-C')
plt.show()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')


def prepare_data_for_arma(df, target_col):
    """
    Prepare the data for ARMA modeling by ensuring it's stationary and properly formatted
    """
    # Ensure the data is sorted by index
    df = df.sort_index()

    # Extract the target variable
    y = df[target_col].values
    return y


def fit_arma_model(data, p, q):
    """
    Fit ARMA(p,q) model using statsmodels
    """
    model = ARIMA(data, order=(p, 0, q))  # Setting d=0 for ARMA
    results = model.fit()
    return results


def get_in_sample_predictions(model):
    """
    Get in-sample predictions (fitted values)
    """
    return model.fittedvalues


def get_one_step_ahead_predictions(model, start_idx, end_idx):
    """
    Get one-step-ahead predictions for a specific range
    """
    predictions = model.get_prediction(start=start_idx, end=end_idx, dynamic=False)
    return predictions.predicted_mean


def implement_arma_analysis(df, target_col, p=1, q=1, test_size=0.2, random_state=42):
    """
    Main function to implement ARMA analysis with train-test split
    """
    # Prepare the data
    y = prepare_data_for_arma(df, target_col)

    # Split the data into training and testing sets
    train_size = int(len(y) * (1 - test_size))
    y_train, y_test = y[:train_size], y[train_size:]

    # Fit ARMA model on training data
    train_model = fit_arma_model(y_train, p, q)

    # Get in-sample predictions for training set
    train_predictions = get_in_sample_predictions(train_model)

    # Fit model on full training data for test predictions
    full_model = fit_arma_model(y, p, q)

    # Get one-step-ahead predictions for test set
    test_predictions = get_one_step_ahead_predictions(
        full_model,
        start_idx=train_size,
        end_idx=len(y) - 1
    )