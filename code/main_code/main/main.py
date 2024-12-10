import sys
sys.path.append('../test')
sys.path.append('../../component')
from utils import *
from class_LSTM import LSTM, BiLSTM, Seq2SeqLSTM
from run_EDA import run_eda
from run_ARIMA import run_arima
from run_LSTM import set_lstm_data, lstm_eval
from run_Transformer import set_trans_data, transformer_eval
from class_transformer import TransformerModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(path, target, dataset, period, ar_order, ma_order, inte_order, seq_length = 6):
    # All EDA plots
    ts_plot, acf, acf_pacf, rol_mean_var, decom = run_eda(path, 100, period)
    ts_plot.show()
    acf.show()
    acf_pacf.show()
    rol_mean_var.show()
    decom.show()

    # Classical models
    # Find the order first, then running the Classical models
    _, _, _, test = prepare_arima_data(path, target)
    _, forecast = run_arima(path, target, ar_order, ma_order, inte_order)

    mse, rmse, mae = calculate_metrics(forecast, test)
    print(f'MSE:{mse}, RMSE:{rmse}, MAE:{mae}')
    fig = plt_forecast(forecast, test, 500, 'Classical model')
    fig.show()

    # Modern techniques: LSTM, BiSTM, Seq2Seq
    train_loader, test_loader, scaler, actual_test = set_lstm_data(path, target, seq_length)
    model = run_lstm(dataset, train_loader, model_name)
    predictions = lstm_eval(model_name, dataset, model, test_loader, scaler)

    mse, rmse, mae = calculate_metrics(predictions, actual_test)
    print(f'MSE:{mse}, RMSE:{rmse}, MAE:{mae}')
    fig = plt_forecast(predictions, actual_test, 500, 'Modern model')
    fig.show()

    # State-of-Art: Transformer
    train_loader, test_loader, scaler, actual_test = set_lstm_data(path, target, seq_length, batch_size=128)
    model = run_transformer(dataset, train_loader, epoches=100)
    predictions = transformer_eval(dataset, model, test_loader, scaler)

    mse, rmse, mae = calculate_metrics(predictions, actual_test)
    print(f'MSE:{mse}, RMSE:{rmse}, MAE:{mae}')
    fig = plt_forecast(predictions, actual_test, 500, 'Transformer')
    fig.show()