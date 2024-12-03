import sys
sys.path.append('../test')
sys.path.append('../../component')
from utils import *
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
    wea_ts, _, wea_acf_pacf, wea_rolling, wea_decom = run_eda(weather_path, 100, 144)
    air_ts, _, air_acf_pacf, air_rolling, air_decom = run_eda(air_pollution_path, 100, 24)
    pow_ts, _, pow_acf_pacf, pow_rolling, pow_decom = run_eda(power_path, 100, 144)
    # tra_ts, tra_acf, tra_acf_pacf, tra_rolling, tra_decom = run_eda(traffic_path, 100, 24)

    eda_plots = pd.DataFrame({'Dataset': ['Temperature', 'Air Pollution', 'Power Consumption'],#, 'Traffic Volume'],
                              'Sequence plots': [wea_ts, air_ts, pow_ts],#, tra_ts],
                              'Rolling mean & var': [wea_rolling, air_rolling, pow_rolling],#, tra_rolling],
                              # 'ACF plots': [wea_acf, air_acf, pow_acf],#, tra_acf],
                              'ACF/PACF plots': [wea_acf_pacf, air_acf_pacf, pow_acf_pacf],#, tra_acf_pacf],
                              'Decompostion plots': [wea_decom, wea_decom, pow_decom]})#, tra_decom]})

    eda_fig = figure_table(eda_plots)
    eda_fig.savefig('eda.pdf', dpi=300, bbox_inches='tight')

    _, _, _, test = prepare_arima_data(weather_path, 'temperature')
    '''Dataset: weather'''
    ### Classical models ###
    w_ar_d_pred, w_ar_d_fore = run_arima(weather_path, 'temperature', 2, 0, 0) # AR with domain knowledge - ACF/PACF
    calculate_metrics(w_ar_d_fore, test)
    fig = plt_forecast(w_ar_d_fore, test, 500, 'AR')

    w_ar_o_pred, w_ar_o_fore = run_arima(weather_path, 'temperature', 4, 0, 0)  # AR with Optuna
    calculate_metrics(w_ar_o_fore, test)
    fig = plt_forecast(w_ar_o_fore, test, 500, 'AR')

    fig.savefig('./pdf/AR.pdf', dpi=300, bbox_inches='tight')

    w_ma_o_pred, w_ma_o_fore = run_arima(weather_path, 'temperature', 0, 10, 0) # No MA pattern, set max MA order to 10
    fig = plt_forecast(w_ma_o_fore, test, 500, 'MA')

    w_arma_d_pred, w_arma_d_fore = run_arima(weather_path, 'temperature', 2, 3, 0) # GPAC
    w_arma_o_pred, w_arma_o_fore = run_arima(weather_path, 'temperature', 12, 11, 0) #Optuna

    w_arima_d_pred, w_arima_d_fore = run_arima(weather_path, 'temperature', 2, 3, 1) # GPAC
    w_arima_o_pred, w_arima_o_fore = run_arima(weather_path, 'temperature', 6, 0, 1) # Optuna

    ### LSTM, Bilstm, Transformer ###
    dataset = 'wea'
    target = 'temperature'
    seq_length = 6

    model_name = 'lstm'
    train_loader, test_loader, scaler, actual_test = set_lstm_data(weather_path, target, seq_length)
    model = LSTM().to(device)
    pred_lstm = lstm_eval(model_name, dataset, model, test_loader, scaler)

    model_name = 'Bilstm'
    train_loader, test_loader, scaler, actual_test = set_lstm_data(weather_path, target, seq_length)
    model = BiLSTM().to(device)
    pred_bilstm = lstm_eval(model_name, dataset, model, test_loader, scaler)

    model_name = 'seq2seq'
    train_loader, test_loader, scaler, actual_test = set_lstm_data(weather_path, target, seq_length)
    model = Seq2SeqLSTM().to(device)
    pred_seq2seq = lstm_eval(model_name, dataset, model, test_loader, scaler)

    '''Transformer'''
    seq_length = 10

    train_loader, test_loader, scaler, actual_test = set_trans_data(weather_path, target, seq_length)
    model = TransformerModel().to(device)
    pred_trans = transformer_eval(dataset, model, test_loader, scaler)

#
# if __name__ == '__main__':
#     main(weather_path)

'''Dataset: Power Consumption'''

power_path = '../../data/power_consumption.csv'
target = 'power_consumption'

_, _, _, test = prepare_arima_data(power_path, target)

### Classical models ###
w_ar_d_pred, w_ar_d_fore = run_arima(weather_path, target, 2, 0, 0) # AR with domain knowledge - ACF/PACF
calculate_metrics(w_ar_d_fore, test)
fig = plt_forecast(w_ar_d_fore, test, 500, 'AR')

w_ar_o_pred, w_ar_o_fore = run_arima(weather_path, target, 4, 0, 0)  # AR with Optuna
calculate_metrics(w_ar_o_fore, test)
fig = plt_forecast(w_ar_o_fore, test, 500, 'AR')

fig.savefig('./pdf/AR.pdf', dpi=300, bbox_inches='tight')

w_ma_o_pred, w_ma_o_fore = run_arima(weather_path, target, 0, 10, 0) # No MA pattern, set max MA order to 10
fig = plt_forecast(w_ma_o_fore, test, 500, 'MA')

w_arma_d_pred, w_arma_d_fore = run_arima(weather_path, target, 2, 3, 0) # GPAC
w_arma_o_pred, w_arma_o_fore = run_arima(weather_path, target, 12, 11, 0) #Optuna

w_arima_d_pred, w_arima_d_fore = run_arima(weather_path, target, 2, 3, 1) # GPAC
w_arima_o_pred, w_arima_o_fore = run_arima(weather_path, target, 6, 0, 1) # Optuna