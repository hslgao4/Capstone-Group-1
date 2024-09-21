from utils import *


class TimeSeriesGenerator:
    def __init__(self, n_samples=5000, freq='D'):
        self.n_samples = n_samples
        self.freq = freq
        self.date_range = pd.date_range(start='2023-01-01', periods=n_samples, freq=freq)


    '''I. Generate data without exogenous input'''
    # 1. generate deterministic data: customized function, sine, cosine, random, linear, exponential
    def gen_determ_data(self, inputs={'type': 'random', 'param': 1}, noise_mean=0, noise_var=1, seasonality=False, trend=False, **kwargs):
        type = inputs.get('type', 'random')
        param = inputs.get('param', 1)
        kwarg = inputs.get('kwargs', {})
        data = gen_deterministic(samples=self.n_samples, custom_func=None, type=type, **kwarg)
        data = param * data

        if seasonality:
            seasonal_kwargs = {k: kwargs[k] for k in ['period_s', 'amplitude'] if k in kwargs}
            data  = seasonality_determ(data, **seasonal_kwargs)

        if trend:
            trend_kwargs = {k: kwargs[k] for k in ['trend_type', 'slope', 'rate'] if k in kwargs}
            data = trend_determ(data, **trend_kwargs)

        # add random noise
        data = data + np.random.normal(noise_mean, np.sqrt(noise_var), self.n_samples)

        return data#pd.DataFrame(data, index=self.date_range)


    # 2. generate stochastic data by models: AR, MA, ARMA, ARIMA, SARIMA, Multiplicative
    def gen_stochs_data(self, ar_para=[0.6], ma_para=[-0.4],
                         sar_para=[0.5], sma_para=[-0.3],
                         d=1, D=1, seasonal_period=12, var_WN=1):

        data = generate_sarima_data(sample_size=self.n_samples,
                         ar_para=ar_para, ma_para=ma_para,
                         sar_para=sar_para, sma_para=sma_para,
                         d=d, D=D, seasonal_period=seasonal_period, var_WN=var_WN)
        return data

    '''II. Generate data with one exogenous input'''
    # 1. deterministic approach
    def gen_determ_data_exo_1(self, exog_inputs, **kwargs):
        data = self.gen_determ_data(self, inputs=[{'type': 'random', 'param': 1}], noise_mean=0, noise_var=1, seasonality=False, trend=False, **kwargs)

        if exog_inputs is None:
            exog_inputs = {'type': 'random', 'param': 0.5}  # by default one input

        exog_type = exog_inputs.get('type', 'random')
        exog_param = exog_inputs.get('param', 0.5)
        exog_kwargs = exog_inputs.get('kwargs', {})
        exog_input = gen_deterministic(samples=self.n_samples, custom_func=None, type=exog_type, **exog_kwargs)

        data = data + exog_param * exog_input

        data_series = pd.Series(data, index=self.date_range)
        exog_df = pd.DataFrame({'exog_1': exog_input}, index=self.date_range)
        df_ts = pd.concat([pd.DataFrame({'ts_data': data_series}, index=self.date_range), exog_df], axis=1)

        return data_series, exog_df, df_ts


    # 2. stochastic approach with Box-Jenkins model

