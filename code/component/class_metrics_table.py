from utils import *


class Metrics_table:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def cal_metrics(self):
        mse = MSE(self.y_true, self.y_pred)
        rmse = RMSE(self.y_true, self.y_pred)
        mae = MAE(self.y_true, self.y_pred)
        return pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'MAE'],
            'Value': [mse, rmse, mae]
        })

    def metric_table(self, title='Metrics Table'):
        df = self.cal_metrics()
        return tabel_pretty(df,title)




