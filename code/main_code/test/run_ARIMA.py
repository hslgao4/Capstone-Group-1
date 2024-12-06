import sys
sys.path.append('../../component')  # Ensure sys is imported before using it
from utils import *
from class_TS_model_classical import ARIMA_model, SARIMA_Model
import os
os.getcwd()

#########################################################################
def run_arima(path, target, ar_order, ma_order, inte_order):
    df_train, df_test, train, test = prepare_arima_data(path, target)

    model = ARIMA_model(AR_order=ar_order, MA_order=ma_order, Inte_order=inte_order)
    model.fit(train)

    pred = model.predict(train)
    fore = model.forecast(len(test))

    return pred, fore

