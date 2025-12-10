

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels
import sklearn
import itertools
import warnings
import arch

from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

forecast_length = 200

#=== Importing asset data ===
ticker = "TSLA"
stock = yf.download(ticker, period="5y", interval="1D", auto_adjust=True, progress=False)['Close']
stock.index = pd.to_datetime(stock.index)
stock = stock.asfreq('B')            # set business day frequency
stock = stock.fillna(method='ffill') # fill holidays
stock = stock.dropna()

#=== Adding return column to stock df ===
stock['Return'] = stock.pct_change()
print(stock['Return'].describe())
stock = stock.dropna()
returns_array = stock['Return'].dropna() * 100
#returns_array = returns_array.index()

#=== Running Augmented Dickey-Fuller test ===
from statsmodels.tsa.stattools import adfuller
result = adfuller(stock[f"{ticker}"].dropna())
#print("p-value:", result[1])

#=== Differencing and repeating ADF test ===
stock['Close_diff'] = stock[f'{ticker}'].diff().dropna()

result = adfuller(stock["Close_diff"].dropna())
#print("p-value:", result[1])

#=== Plotting Autocorellation Function and Partial Autocorellation Function ===
plot_acf(stock['Close_diff'].dropna(), lags=30)
plot_pacf(stock['Close_diff'].dropna(), lags=30)


#=== GARCH model and forecasting ===

model = arch_model(stock['Return'], vol='GARCH', p=1, q=1, mean='Zero', dist='t')  #Try different distributions 't' distribution or normal distribution give different results
fit = model.fit(disp='off')
forecast = fit.forecast(horizon=forecast_length)
vol_forecast = np.sqrt(forecast.variance[-1:].values.flatten())

print(fit.summary())
#print(vol_forecast)
#print(returns_array)

alpha = fit.params['alpha[1]']
beta = fit.params['beta[1]']
print('Volatility Persistence:  ', alpha + beta)

VaR_95 = -1.65 * fit.conditional_volatility * -1 #Multiplying my -1 so it gives a positive loss value. Looks much better on the chart. Got rid of the /100 term as well so it scales the same as volatility which we x100.

plt.figure(figsize=(15,8))
plt.plot(fit.conditional_volatility, color = 'red', label = 'Estimated Volatility')
future_index = pd.date_range(returns_array.index[-1], periods=forecast_length+1, freq='B')[1:]
plt.plot(future_index, vol_forecast, 'g--', label='Future Forecasted Volatility')
plt.plot(VaR_95, label='VaR 95%')
plt.title('GARCH(1,1) Conditional Volatility')
plt.xlabel('Date')
plt.ylabel('Daily percentage standard deviation / percentage risk')
plt.grid(True)
plt.legend()

plt.show()