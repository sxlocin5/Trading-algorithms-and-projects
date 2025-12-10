#Time series analysis and forecasting using ARIMA scheme.

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels
import sklearn
import itertools
import warnings

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

#=== Importing asset data ===
ticker = "AAPL"
stock = yf.download(ticker, period="5y", interval="1D", auto_adjust=True, progress=False)['Close']
stock.index = pd.to_datetime(stock.index)
stock = stock.asfreq('B')            # set business day frequency
stock = stock.fillna(method='ffill') # fill holidays
stock = stock.dropna()

#=== Plotting asset price data ===
#plt.figure(figsize=(15, 8))
#plt.plot(stock)
#plt.title(f"{ticker} stock price")
#plt.xlabel("Date")
#plt.ylabel("Price ($)")

#=== Adding return column to stock df ===
stock['Return'] = stock.pct_change()
print(stock['Return'].describe())

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

#=== Extracting data and performing ARIMA forecast ===
stock = stock.dropna()
train = stock["Return"][: - 60]
test = stock["Return"][: -60]
train = train.dropna()
test = test.dropna()

p = range(0,3)  #Defining ranges for p,dpq values for the ARIMA forecast to test which combinations give lowest AIC.
d = range(0,3)
q = range(0,3)
pdq = list(itertools.product(p,d,q))
model_results = []

best_aic = np.inf
best_model = None
best_order = None
best_forecast = None

for order in pdq:
    try:
        model = ARIMA(train, order = order) #(AutoRegression, Integration, Moving Average) ARIMA (p,d,q) 
        model_result = model.fit()
        forecast = model_result.forecast(steps=len(test))
        rmse_arima = np.sqrt(mean_squared_error(test, forecast))
        model_results.append({
            "Order": order,
            "AIC": model_result.aic,
            "RMSE": rmse_arima
        })
        if order == (2,0,2): #model_result.aic < best_aic:
            best_aic = model_result.aic #For above, use the first one if you want to use a forecast from specific p,d,q values, or use the if tstatement if you want to use the best forecast's p,d,q values. 
            best_model = model_result
            best_order = order
            best_forecast = forecast

    except Exception as e:
        continue

#print(best_order)
#print(best_forecast.head())


#=== Creating Dataframe of results and sorting by lowest AIC ===
results_df = pd.DataFrame(model_results)
results_df = results_df.sort_values("AIC").reset_index(drop = True)
#print(model_result.summary())
forecast = pd.Series(best_forecast.values, index=test.index)

residuals = model_result.resid

plt.figure(figsize = (15,8))
plt.plot(residuals)
plt.title("Residuals")
plot_acf(residuals)

#=== Calculating RMSE for comparison of ARIMA and naive forecast ===

naive_forecast = np.repeat(train.mean(), len(test))    #This is modeling using average returns. If we were using price we would use tomorrow's price = today's price.
rmse_naive = np.sqrt(mean_squared_error(test, naive_forecast))

print("\n\nThe top 4 ARIMA input varaible combinations in terms of lowest AIC are as follows:\n\n")
print(results_df.head(4))
 
print(f"\nNaive forecast RMSE: {rmse_naive:.6f}\n")

comparison = pd.DataFrame({
    "Actual": test,
    "ARIMA": forecast,
    "Naive": naive_forecast
}, index=test.index)

#print(forecast.head())

plt.figure(figsize=(15, 8))
plt.plot(comparison["Actual"], color = "grey", label = "Actual")
plt.plot(comparison["ARIMA"], color = "red", label = "ARIMA")
#plt.plot(test, color = "red", label = "Test")
plt.plot(comparison["Naive"], color = "blue", label = "Naive")
plt.title("Actual vs ARIMA forecast vs Naive forecast")
plt.xlabel("Date")
plt.ylabel("Forecast Returns")
plt.legend()
plt.show()