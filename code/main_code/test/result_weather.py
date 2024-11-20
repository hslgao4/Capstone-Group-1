import sys
import pandas as pd
sys.path.append('../../component')
from utils import *
from class_LSTM import LSTM, BiLSTM, Seq2SeqLSTM
import os
os.getcwd()

path = '../../data/weather.csv'
target = 'temperature'

class df_weather_results:
    def __init__(self, path, target):
        self.path = path
        self.target = target

    """Classical Time Series Model"""
    def ar_model(self):
        df_train, df_test, train, test = prepare_arima_data(self.path, self.target)

        '''AR model'''
        # Domain knowledge
        ar_train_err_d, ar_test_err_d, ar_train_mse_d, ar_test_mse_d, ar_train_plt_d, ar_test_plt_d = ARIMA_results(2, 0, 0, df_train, df_test, train, test)
        ar_train_err_acf_d = plt_ACF(ar_train_err_d, 40)
        ar_test_err_acf_d = plt_ACF(ar_test_err_d, 40)

        # Optuna
        ar_train_err_o, ar_test_err_o, ar_train_mse_o, ar_test_mse_o, ar_train_plt_o, ar_test_plt_o = ARIMA_results(4, 0, 0, df_train, df_test, train, test)

        ar_train_err_acf_o = plt_ACF(ar_train_err_o, 40)
        ar_test_err_acf_o = plt_ACF(ar_test_err_o, 40)

        return ar_train_mse_d, ar_test_mse_d, ar_train_plt_d, ar_test_plt_d, ar_train_err_acf_d, ar_test_err_acf_d, ar_train_mse_o, ar_test_mse_o, ar_train_plt_o, ar_test_plt_o, ar_train_err_acf_o, ar_test_err_acf_o

    def ma_model(self):
        '''MA model'''
        # No MA pattern in ACF/PACF, so just use Optuna MA(10)
        df_train, df_test, train, test = prepare_arima_data(self.path, self.target)
        ma_train_err, ma_test_err, ma_train_mse, ma_test_mse, ma_train_plt, ma_test_plt = ARIMA_results(0, 10, 0, df_train, df_test, train, test)
        ma_train_err_acf = plt_ACF(ma_train_err, 40)
        ma_test_err_acf = plt_ACF(ma_test_err, 40)

        return ma_train_mse, ma_test_mse, ma_train_plt, ma_test_plt, ma_train_err_acf, ma_test_err_acf

    def arma_model(self):
        df_train, df_test, train, test = prepare_arima_data(self.path, self.target)
        '''ARMA model'''
        # GPAC: ARMA(2,3)
        arma_train_err_d, arma_test_err_d, arma_train_mse_d, arma_test_mse_d, arma_train_plt_d, arma_test_plt_d = ARIMA_results(2, 3, 0, df_train, df_test, train, test)
        arma_train_err_acf_d = plt_ACF(arma_train_err_d, 40)
        arma_test_err_acf_d = plt_ACF(arma_test_err_d, 40)

        # Optuna(12,11)
        arma_train_err_o, arma_test_err_o, arma_train_mse_o, arma_test_mse_o, arma_train_plt_o, arma_test_plt_o = ARIMA_results(6, 6, 0, df_train, df_test, train, test)

        arma_train_err_acf_o = plt_ACF(arma_train_err_o, 40)
        arma_test_err_acf_o = plt_ACF(arma_test_err_o, 40)

        return arma_train_mse_d, arma_test_mse_d, arma_train_plt_d, arma_test_plt_d, arma_train_err_acf_d, arma_test_err_acf_d, arma_train_mse_o, arma_test_mse_o, arma_train_plt_o, arma_test_plt_o, arma_train_err_acf_o, arma_test_err_acf_o

     def arima_model(self):
        df_train, df_test, train, test = prepare_arima_data(self.path, self.target)
        '''ARIMA model'''
        # GPAC ARIMA(2,1,3)
        arima_train_err_d, arima_test_err_d, arima_train_mse_d, arima_test_mse_d, arima_train_plt_d, arima_test_plt_d = ARIMA_results(2, 3, 1, df_train, df_test, train, test)

        arima_train_err_acf_d = plt_ACF(arima_train_err_d, 40)
        arima_test_err_acf_d = plt_ACF(arima_test_err_d, 40)

        # Optuna ARIMA(6, 1, 0)
        arima_train_err_o, arima_test_err_o, arima_train_mse_o, arima_test_mse_o, arima_train_plt_o, arima_test_plt_o = ARIMA_results(6, 0, 1, df_train, df_test, train, test, validation)

        arima_train_err_acf_o = plt_ACF(arima_train_err_d, 40)
        arima_test_err_acf_o = plt_ACF(arima_test_err_d, 40)

        return arima_train_mse_d, arima_test_mse_d, arima_train_plt_d, arima_test_plt_d, arima_train_err_acf_d, arima_test_err_acf_d, arima_train_mse_o, arima_test_mse_o, arima_train_plt_o, arima_test_plt_o, arima_train_err_acf_o, arima_test_err_acf_o


    def lstm_model(self):
        """LSTM, BiLSTM, Seq2seq"""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        '''LSTM'''
        #%%
        seq_length = 30
        X_train, y_train, X_test, y_test, scaler = pre_lstm_data(self.path, self.target, seq_length)

        batch_size = 128 * 20
        input_size = 1
        hidden_size = 2
        num_layers = 1
        output_size = 1

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

        lstm = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
        criterion = nn.MSELoss()
        lstm.load_state_dict(torch.load('LSTM_model_weights.pt'))

        train_prediction, lstm_pred_mse = lstm_loop(lstm, train_loader, device, criterion)
        test_forecast, lstm_fore_mse = lstm_loop(lstm, test_loader, device, criterion)

        lstm_train_pred = scaler.inverse_transform(torch.cat(train_prediction).numpy())
        lstm_test_fore = scaler.inverse_transform(torch.cat(test_forecast).numpy())

        y_train = scaler.inverse_transform(y_train.cpu().numpy())
        y_test = scaler.inverse_transform(y_test.cpu().numpy())


    def bilstm_model(self):
        '''BiLSTM'''
        #%%
        seq_length = 2
        X_train, y_train, X_test, y_test, scaler = pre_lstm_data(self.path, self.target, seq_length)

        batch_size = 128 * 20
        input_size = 1
        hidden_size = 64 * 2
        num_layers = 2 + 1
        output_size = 1

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

        Bilstm = BiLSTM(input_size, hidden_size, num_layers, output_size).to(device)
        criterion = nn.MSELoss()
        Bilstm.load_state_dict(torch.load('BiLSTM_model_weights.pt'))

        train_prediction, Bilstm_pred_mse = lstm_loop(Bilstm, train_loader, device, criterion)
        test_forecast, Bilstm_fore_mse = lstm_loop(Bilstm, test_loader, device, criterion)

        Bilstm_train_pred = scaler.inverse_transform(torch.cat(train_prediction).numpy())
        Bilstm_test_fore = scaler.inverse_transform(torch.cat(test_forecast).numpy())

    def seq2seq_model(self):
        '''Seq2seq'''
        #%%
        seq_length = 5
        X_train, y_train, X_test, y_test, scaler = pre_lstm_data(self.path, self.target, seq_length)

        batch_size = 64 * 2
        input_size = 1
        hidden_size = 64
        output_size = 1
        num_layers = 2

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

        seg2seq = Seq2SeqLSTM(input_size, hidden_size, output_size, num_layers).to(device)
        criterion = nn.MSELoss()
        seg2seq.load_state_dict(torch.load('seq2seq_model_weights.pt'))

        train_prediction, seq2seq_pred_mse = lstm_loop(seg2seq, train_loader, device, criterion)
        test_forecast, seq2seq_fore_mse = lstm_loop(seg2seq, test_loader, device, criterion)

        seq2seq_train_pred = scaler.inverse_transform(torch.cat(train_prediction).numpy())
        seq2seq_test_fore = scaler.inverse_transform(torch.cat(test_forecast).numpy())


    def main(self):
        '''Single dataset model performance'''
        ar_train_mse_d, ar_test_mse_d, ar_train_plt_d, ar_test_plt_d, ar_train_err_acf_d, ar_test_err_acf_d, ar_train_mse_o, ar_test_mse_o, ar_train_plt_o, ar_test_plt_o, ar_train_err_acf_o, ar_test_err_acf_o = self.ar_model()
        ma_train_mse, ma_test_mse, ma_train_plt, ma_test_plt, ma_train_err_acf, ma_test_err_acf= self.ma_model()
        arma_train_mse_d, arma_test_mse_d, arma_train_plt_d, arma_test_plt_d, arma_train_err_acf_d, arma_test_err_acf_d, arma_train_mse_o, arma_test_mse_o, arma_train_plt_o, arma_test_plt_o, arma_train_err_acf_o, arma_test_err_acf_o = self.arma_model()
        arima_train_mse_d, arima_test_mse_d, arima_train_plt_d, arima_test_plt_d, arima_train_err_acf_d, arima_test_err_acf_d, arima_train_mse_o, arima_test_mse_o, arima_train_plt_o, arima_test_plt_o, arima_train_err_acf_o, arima_test_err_acf_o = self.arima()

        weather = pd.DataFrame({'Models': ['AR', 'MA', 'ARMA', 'ARIMA'],
                                'Train prediction': [ar_train_plt_o, ma_train_plt, arma_train_plt_o, arima_train_plt_o],
                                'Train error ACF': [ar_train_err_acf_o, ma_train_err_acf, arma_train_err_acf_o, arima_train_err_acf_o],
                                'Test prediction': [ar_test_plt_o, ma_test_plt, arma_test_plt_o, arima_test_plt_o],
                                'Test error ACF': [ar_test_err_acf_o, ma_test_err_acf, arma_test_err_acf_o,arima_test_err_acf_o]})

        figure_1 = figure_table(weather, 'Model performance comparison')



        # Optuna vs domain knowledge
        df_grid_search = pd.DataFrame({'Models': ['AR', 'MA', 'ARMA', 'ARIMA'],
                                       'Optuna': [4, 10, (6,6), (6,1,0)],
                                       'MSE_1': [ar_test_mse_o, ma_test_mse, arma_test_mse_o, arima_test_mse_o],
                                       'GPAC & ACF/PACF': [2, 'N/A', (2,3), (2,1,3)],
                                       'MSE_2': [ar_test_mse_d, 'N/A', arma_test_mse_d, arima_test_mse_d]})


        figure_2 = plot_metric_table(df_grid_search, 'Optuna vs. Domain knowledge')

        return figure_1, figure_2


        # # Model performance between all datasets, the metric
        # dfs_metric = pd.DataFrame({'Dataset': ['Temperature', 'Air Pollution', 'Power Consumption', 'Traffic Volume'],
        #                              'AR model': [ar_test_err_mse_1, 0, 0, 0],
        #                              'MA model': [ma_test_err_mse, 0, 0, 0 ],
        #                              'ARMA model': [arma_test_err_mse_1, 0, 0, 0 ],
        #                              'ARIMA model': [arima_test_err_mse_2, 0, 0, 0 ]})
        # plot_metric_table(dfs_metric, 'Metric comparison for 4 Datasets')
        #
        #
        #
        # test_prediction = pd.DataFrame({'Models': ['AR', 'MA', 'ARMA', 'ARIMA'],
        #                         'Temperature': [ar_test_plt_2, ma_test_plt_1, arma_test_plt_2, arima_test_plt_2],
        #                         'Air pollution': [ar_test_plt_2, ma_test_plt_1, arma_test_plt_2, arima_test_plt_2],
        #                         'Power Consumption': [ar_test_plt_2, ma_test_plt_1, arma_test_plt_2, arima_test_plt_2],
        #                         'Traffic Volume': [ar_test_plt_2, ma_test_plt_1, arma_test_plt_2, arima_test_plt_2]})
        # figure_table(test_prediction, 'Test forecast comparison for 4 Datasets')