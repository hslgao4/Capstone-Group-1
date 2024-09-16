import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TimeSeriesGenerator:
    def __init__(self, n_samples=1000, freq='D'):
        self.n_samples = n_samples
        self.freq = freq
        self.date_range = pd.date_range(start='2023-01-01', periods=n_samples, freq=freq)

    def generate_ar_process(self, ar_params=[0.6, -0.3], sigma=1):
        '''Start with AR process'''
        ar = np.zeros(self.n_samples)
        for t in range(len(ar_params), self.n_samples):
            ar[t] = np.sum([ar_params[i] * ar[t - i - 1] for i in range(len(ar_params))]) + np.random.normal(
                scale=sigma)
        return ar


    def generate_exog_input(self, custom_func=None, type='random', **kwargs):
        '''Add exogenous input'''
        if custom_func is not None:
            return custom_func(self.n_samples, **kwargs)
        if type == 'random':
            return np.random.randn(self.n_samples)
        elif type == 'sine':
            period = kwargs.get('period', 365)
            amplitude = kwargs.get('amplitude', 1)
            return amplitude * np.sin(2 * np.pi * np.arange(self.n_samples) / period)
        elif type == 'cosine':
            period = kwargs.get('period', 365)
            amplitude = kwargs.get('amplitude', 1)
            return amplitude * np.cos(2 * np.pi * np.arange(self.n_samples) / period)
        elif type == 'trend':
            slope = kwargs.get('slope', 0.1)
            return slope * np.arange(self.n_samples)
        elif type == 'exponential':
            rate = kwargs.get('rate', 0.01)
            return np.exp(rate * np.arange(self.n_samples))
        else:
            raise ValueError("Invalid exogenous input type")

    def add_seasonality(self, data, period_s=365, amplitude_s=1):
        '''Optional to add seasonality'''
        seasonal = amplitude_s * np.sin(2 * np.pi * np.arange(self.n_samples) / period_s)
        return data + seasonal

    def add_cyclicity(self, data, period_c=1000, amplitude_c=1):
        '''Optional to add cyclicity - TBD'''
        cyclic = amplitude_c * np.sin(2 * np.pi * np.arange(self.n_samples) / period_c)
        return data + cyclic

    def add_trend(self, data, trend_type='linear', **kwargs):
        '''Optional to add trend'''
        if trend_type == 'linear':
            slope = kwargs.get('slope', 0.1)
            trend = slope * np.arange(self.n_samples)
        elif trend_type == 'exponential':
            rate = kwargs.get('rate', 0.001)
            trend = np.exp(rate * np.arange(self.n_samples))
        else:
            raise ValueError("Invalid trend type")
        return data + trend

    def generate_time_series(self, ar_params=[0.6, -0.3], exog_inputs=None,
                             seasonality=False, cyclicity=False,
                             trend=True, noise_scale=0.1, **kwargs):
        """Generate the final time series."""
        ar_component = self.generate_ar_process(ar_params, sigma=noise_scale)

        # Handle exogenous inputs
        if exog_inputs is None:
            exog_inputs = [{'type': 'random', 'param': 0.5}] # by default one input

        exog_components = [] # with more than one exogenous inouts
        for exog in exog_inputs:
            exog_type = exog.get('type', 'random')
            exog_param = exog.get('param', 0.5)
            exog_kwargs = exog.get('kwargs', {})
            exog_input = self.generate_exog_input(type=exog_type, **exog_kwargs)
            exog_components.append(exog_param * exog_input)

        ts = ar_component + np.sum(exog_components, axis=0)

        if seasonality:
            seasonal_kwargs = {k: kwargs[k] for k in ['period', 'amplitude'] if k in kwargs}
            ts = self.add_seasonality(ts, **seasonal_kwargs)
        if cyclicity:
            cyclic_kwargs = {k: kwargs[k] for k in ['period', 'amplitude'] if k in kwargs}
            ts = self.add_cyclicity(ts, **cyclic_kwargs)
        if trend:
            trend_kwargs = {k: kwargs[k] for k in ['trend_type', 'slope', 'rate'] if k in kwargs}
            ts = self.add_trend(ts, **trend_kwargs)

        ts_series = pd.Series(ts, index=self.date_range)
        exog_df = pd.DataFrame({f'exog_{i}': comp for i, comp in enumerate(exog_components)}, index=self.date_range)
        df_ts = pd.concat([pd.DataFrame({'time_series': ts}, index=self.date_range), exog_df], axis=1)

        return ts_series, exog_df, df_ts

    def plot_time_series(self, ts):
        plt.figure(figsize=(10, 8))
        plt.plot(ts)
        plt.xlabel('Date')
        plt.ylabel('Magnitude')
        plt.title('Genrated Time Series data over Date')
        plt.xticks(rotation=45)
        plt.show()