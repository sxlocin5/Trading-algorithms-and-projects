#Reads Excel portfolio file and carries out a Monte-Carlo simulation and VaR analysis.

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

portfolio = pd.read_excel('portfolio_info.xlsx')

tickers = portfolio["Ticker"].iloc[0:len(portfolio)-2].tolist()

# Get data
data = yf.download(tickers, period="5y", interval="1mo", auto_adjust=True, progress=False)['Close']
returns = np.log(data / data.shift(1)).dropna()
returns[len(portfolio)], returns[len(portfolio) - 1] = 0, 0

# Compute stats
portfolio["Mean Return"] = returns.mean().values
portfolio["Std Dev"] = returns.std().values
portfolio["Weighting %"] = portfolio["Weighting %"]*100

print(portfolio)

num_sims = 10000
num_months = 24
VaR_confidence_level = 95
alpha = 1-(VaR_confidence_level/100)

#Could add starting price of each asset, can model as yesterday's closing price for simplicity.

start_price = 100
portfolio_starting_capital = 1000

weights_0 = portfolio["Weighting %"].iloc[0 : len(portfolio) - 2].values / 100
mean_array = portfolio["Mean Return"].values
std_array = portfolio["Std Dev"].values

dt = 1 #Since returns have been calculated on a monthly basis

returns_matrix = np.zeros((num_months, num_sims, len(tickers)))




#=== Main Monte Carlo Loop ===
for i in range(0, len(tickers)):
    mu = mean_array[i]
    sigma = std_array[i]
    

    for j in range(0,num_months):

        #=== Using Geometric Brownian Motion to simulate asset returns ===
        Z = np.random.normal(0, 1, num_sims)
        returns_matrix[j, :, i] = np.exp((mu - 0.5*(sigma**2))*dt + sigma*np.sqrt(dt)*Z) - 1


    if i == len(tickers)-1:
        cumulative = np.cumprod(1 + returns_matrix, axis=0)

        weighted_values = weights_0 * cumulative   
        portfolio_value = weighted_values.sum(axis=2, keepdims=True)
        weights_matrix = weighted_values / portfolio_value


portfolio_gains = (portfolio_value - 1)*100
portfolio_gains = portfolio_gains.squeeze()

zeros_row = np.zeros((1, portfolio_gains.shape[1]))  # shape (1, S)
portfolio_gains = np.vstack([zeros_row, portfolio_gains])

months = np.arange(portfolio_gains.shape[0])





#=== Reference asset Monte Carlo simulation ===
ref_asset = "S&P 500"
mu_ref = 11.0 / 100
sigma_ref = 16.0 / 100
dt_ref = 1/12   #Since using yearly returns

ref_returns = np.zeros((num_months, num_sims))

for k in range(0, num_months):
    Z = np.random.normal(0, 1, num_sims)
    ref_returns[k, :] = np.exp((mu_ref - 0.5*(sigma_ref**2))*dt_ref + sigma_ref*np.sqrt(dt_ref)*Z) - 1

cumulative_ref = np.cumprod(1 + ref_returns, axis=0)
returns_ref = (cumulative_ref - 1) * 100
final_gains_ref = returns_ref[-1]
mean_gain_ref = np.mean(final_gains_ref)

mean_path_ref = returns_ref.mean(axis = 1)
mean_path_ref = np.insert(mean_path_ref, 0, 0)






#=== Plotting simulation results ===

plt.figure(figsize=(15, 8))
plt.plot(months,portfolio_gains)
plt.ylabel("Portfolio Percentage Gain")
plt.xlabel("Time in Months")
plt.title(f'\n Monte Carlo Simulation of portfolio gains over time \n')
plt.plot(months, portfolio_gains.mean(axis=1), color='black', linewidth=1.5, label='Portfolio Mean Path')
plt.plot(months, mean_path_ref, color='blue', linewidth=1.5, label='S&P 500 Mean Path')
plt.legend()
plt.show()





final_gains = portfolio_gains[-1]
mean_gain = np.mean(final_gains)

asset_gains = np.zeros((len(tickers), num_sims))
asset_gains_mean = np.zeros((len(tickers)))
asset_gains_std = np.zeros((len(tickers)))

for i in range(0, len(tickers)):
    #Want to have final gains of each asset, then find the mean.
    asset_gains[i] = (cumulative[-1, : , i] - 1) * 100
    asset_gains_mean[i] = asset_gains[i].mean()
    asset_gains_std[i] = asset_gains[i].std()

asset_gains_mean = np.round(asset_gains_mean,2)
asset_gains_std = np.round(asset_gains_std,2)

#final_weightings = np.mean(weights_matrix[num_months - 1, :, :])
final_weightings = weights_matrix[-1, :, :]  
final_weightings = final_weightings.mean(axis = 0)
#final_weightings = final_weightings*100/len(tickers)






#==== Asset performance table ====
asset_returns_df = pd.DataFrame({
    "Ticker": tickers,
    "Mean Percentage Gain": asset_gains_mean,
    "Standard Deviation": asset_gains_std,
    "Final Weighting (%)": np.round(final_weightings * 100, 2) 
})

asset_returns_df = asset_returns_df.sort_values(by="Mean Percentage Gain", ascending=False)

print(f"\n\nAsset performance breakdown over {num_months} months, in order of greatest return:\n\n {asset_returns_df}")





#==== VaR calculations ====

VaR = np.percentile(final_gains, 1-(VaR_confidence_level)/100)
CVaR = final_gains[final_gains <= VaR].mean()
VaR = abs(np.round(VaR, 2))
CVaR = abs(np.round(CVaR, 2))

mean_gain = np.round(mean_gain, 2)
outperformance = mean_gain - mean_gain_ref
outperformance = np.round(outperformance, 2)

print(f"\n\n The mean portfolio gain over {num_months} months is {mean_gain}%, outperforming the {ref_asset} by {outperformance}%, which returned an average of {np.round(mean_gain_ref,2)}% over the same period.")
print(f"\n With {VaR_confidence_level}% confidence, the portfolio will not lose more than {VaR}% over {num_months} months.\n\n In scenarios where losses do exceed this threshold, the mean loss is {CVaR}%. \n\n")

# Have the option to look into the simulations of each asset, put a loop in the end that asks for user input for the ticker and show the MC sim of that asset along with metrics.