import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class TimeSeriesGenerator:
    def __init__(self, n_samples=1000, freq='D'):
        self.n_samples = n_samples
        self.freq = freq
        self.date_range = pd.date_range(start='2023-01-01', periods=n_samples, freq=freq)

    def generate_ar_process(self, ar_params=[0.6, -0.3], sigma=1):
        """Generate an autoregressive process."""
        ar = np.zeros(self.n_samples)
        for t in range(len(ar_params), self.n_samples):
            ar[t] = np.sum([ar_params[i] * ar[t - i - 1] for i in range(len(ar_params))]) + np.random.normal(scale=sigma)
        return ar

    def generate_exog_input(self, type='random', **kwargs):
        if type == 'random':
            return np.random.randn(self.n_samples)
        elif type == 'sine':
            period = kwargs.get('period', 365)
            amplitude = kwargs.get('amplitude', 1)
            return amplitude * np.sin(2 * np.pi * np.arange(self.n_samples) / period)
        elif type == 'trend':
            slope = kwargs.get('slope', 0.1)
            return slope * np.arange(self.n_samples)
        else:
            raise ValueError("Invalid exogenous input type")

    def add_seasonality(self, data, period=365, amplitude=1):
        seasonal = amplitude * np.sin(2 * np.pi * np.arange(self.n_samples) / period)
        return data + seasonal

    def add_cyclicity(self, data, period=1000, amplitude=1):
        cyclic = amplitude * np.sin(2 * np.pi * np.arange(self.n_samples) / period)
        return data + cyclic

    def add_trend(self, data, trend_type='linear', **kwargs):
        if trend_type == 'linear':
            slope = kwargs.get('slope', 0.1)
            trend = slope * np.arange(self.n_samples)
        elif trend_type == 'exponential':
            rate = kwargs.get('rate', 0.001)
            trend = np.exp(rate * np.arange(self.n_samples))
        else:
            raise ValueError("Invalid trend type")
        return data + trend

    def generate_time_series(self, ar_params=[0.6, -0.3], exog_params=[0.5],
                             exog_type='random', seasonality=False, cyclicity=False,
                             trend=True, noise_scale=0.1, **kwargs):
        """Generate the final time series."""
        ar_component = self.generate_ar_process(ar_params, sigma=noise_scale)
        exog_input = self.generate_exog_input(type=exog_type, **kwargs)
        exog_component = np.sum([exog_params[i] * exog_input for i in range(len(exog_params))], axis=0)

        ts = ar_component + exog_component

        if seasonality:
            seasonal_kwargs = {k: kwargs[k] for k in ['period', 'amplitude'] if k in kwargs}
            ts = self.add_seasonality(ts, **seasonal_kwargs)
        if cyclicity:
            cyclic_kwargs = {k: kwargs[k] for k in ['period', 'amplitude'] if k in kwargs}
            ts = self.add_cyclicity(ts, **cyclic_kwargs)
        if not trend:
            nonstationary_kwargs = {k: kwargs[k] for k in ['trend_type', 'slope', 'rate'] if k in kwargs}
            ts = self.add_trend(ts, **nonstationary_kwargs)

        ts_series = pd.Series(ts, index=self.date_range)
        exog_series = pd.Series(exog_input, index=self.date_range)
        df_ts = pd.DataFrame({'time_series': ts, 'exog_input': exog_input}, index=self.date_range)
        return ts_series, exog_series, df_ts


    def plot_time_series(self, ts, exog, title="Generated Time Series"):
        """Plot the generated time series and exogenous input."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax1.plot(ts)
        ax1.set_title(title)
        ax1.set_ylabel("Value")
        ax2.plot(exog, color='red')
        ax2.set_title("Exogenous Input")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Value")
        plt.tight_layout()
        plt.show()



# Example usage
generator = TimeSeriesGenerator(n_samples=1000, freq='D')
ts, exog, df_ts = generator.generate_time_series(
    ar_params=[0.6, -0.3],
    exog_params=[0.5],
    exog_type='sine',
    seasonality=True,
    cyclicity=True,
    stationary=False,
    noise_scale=0.1,
    period=365,
    amplitude=2,
    slope=0.01
)

generator.plot_time_series(ts, exog)

print(ts.head())
print("\nAutocorrelation:")
print(ts.autocorr(lag=1))
print("\nCorrelation with exogenous input:")
print(ts.corr(exog))


# def plot_time_series(self, ts, exog, title="Generated Time Series"):
#     """Plot the generated time series and exogenous input."""
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
#     ax1.plot(ts)
#     ax1.set_title(title)
#     ax1.set_ylabel("Value")
#     ax2.plot(exog, color='red')
#     ax2.set_title("Exogenous Input")
#     ax2.set_xlabel("Date")
#     ax2.set_ylabel("Value")
#     plt.tight_layout()
#     plt.show()

plt.plot(ts)
plt.show()