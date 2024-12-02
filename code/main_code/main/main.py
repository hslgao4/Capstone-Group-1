import sys
sys.path.append('../test')
from run_EDA import run_eda
from run_ARIMA import run_arima
from run_LSTM import *
from run_Transformer import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weather_path = '../../data/weather.csv'
air_pollution_path = '../../data/air_pollution.csv'
power_path = '../../data/power_consumption.csv'
traffic_path = '../../data/traffic.csv'

def main(weather_path):#, air_pollution_path, power_path, traffic_path):
    # EDA plots
    wea_ts, wea_acf, wea_acf_pacf, wea_rolling, wea_decom = run_eda(weather_path, 100, 144)
    air_ts, air_acf, air_acf_pacf, air_rolling, wea_decom = run_eda(air_pollution_path, 100, 24)
    pow_ts, pow_acf, pow_acf_pacf, pow_rolling, pow_decom = run_eda(power_path, 100, 144)
    tra_ts, tra_acf, tra_acf_pacf, tra_rolling, tra_decom = run_eda(traffic_path, 100, 24)

    eda_plots = pd.DataFrame({'Dataset': ['Temperature', 'Air Pollution', 'Power Consumption', 'Traffic Volume'],
                              'Sequence plots': [wea_ts, air_ts, pow_ts, tra_ts],
                              'Rolling mean & var': [wea_rolling, air_rolling, pow_rolling, tra_rolling],
                              'ACF plots': [wea_acf, air_acf, pow_acf, tra_acf],
                              'ACF/PACF plots': [wea_acf_pacf, air_acf_pacf, pow_acf_pacf, tra_acf_pacf],
                              'Decompostion plots': [wea_decom, wea_decom, pow_decom, tra_decom]})

    figure_table(eda_plots, 'EDA plots of four Datasets')

    '''Dataset: weather'''
    ### Classical models ###
    w_ar_d_pred, w_ar_d_fore = run_arima(weather_path, 'temperature', 2, 0, 0) # AR with domain knowledge - ACF/PACF
    w_ar_o_pred, w_ar_o_fore = run_arima(weather_path, 'temperature', 4, 0, 0)  # AR with Optuna

    w_ma_o_pred, w_ma_o_fore = run_arima(weather_path, 'temperature', 0, 10, 0) # No MA pattern, set max MA order to 10

    w_arma_d_pred, w_arma_d_fore = run_arima(weather_path, 'temperature', 2, 3, 0) # GPAC
    w_arma_o_pred, w_arma_o_fore = run_arima(weather_path, 'temperature', 12, 11, 0) #Optuna

    w_arima_d_pred, w_arima_d_fore = run_arima(weather_path, 'temperature', 2, 3, 1) # GPAC
    w_arima_o_pred, w_arima_o_fore = run_arima(weather_path, 'temperature', 6, 0, 1) # Optuna

    ### LSTM, Bilstm, Transformer ###
    dataset = 'wea'
    target = 'temperature'
    batch_size = 128 * 20

    model_name = 'lstm'
    seq_length = 30
    hidden_size = 2
    num_layers = 1
    train_loader, test_loader, scaler, actual_test = set_data(weather_path, target, seq_length, batch_size)
    model = LSTM(hidden_size, num_layers).to(device)
    pred_lstm = lstm_eval(model_name, dataset, model, test_loader, scaler)


    model_name = 'Bilstm'
    seq_length = 2
    hidden_size = 64 * 2
    num_layers = 2 + 1
    train_loader, test_loader, scaler, actual_test = set_data(weather_path, target, seq_length, batch_size)
    model = BiLSTM(hidden_size, num_layers).to(device)
    pred_bilstm = lstm_eval(model_name, dataset, model, test_loader, scaler)

    model_name = 'seq2seq'
    seq_length = 5
    batch_size = 64 * 2
    hidden_size = 64
    num_layers = 2
    train_loader, test_loader, scaler, actual_test = set_data(weather_path, target, seq_length, batch_size)
    model = Seq2SeqLSTM(hidden_size, num_layers).to(device)
    pred_seq2seq = lstm_eval(model_name, dataset, model, test_loader, scaler)

    '''Transformer'''
    seq_length = 10
    batch_size = 32

    train_loader, test_loader, scaler, actual_test = set_data(weather_path, target, seq_length, batch_size)
    model = TransformerModel(d_model=64, nhead=4, num_layers=2, dropout=0.2).to(device)
    pred_trans = transformer_eval(dataset, model, test_loader, scaler)


if __name__ == '__main__':
    main(weather_path)
