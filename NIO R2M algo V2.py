import numpy as np
import pandas as pd
import pandas_datareader as pdr
from ta.volatility import BollingerBands

import matplotlib
matplotlib.use('Qt5Agg')
import yfinance as yf
import matplotlib.pyplot as plt

# Fetch stock data
stock = yf.download('NIO')

#Adding moving averages
#stock['9-day'] = stock['Close'].rolling(9).mean().shift()
#stock['21-day'] = stock['Close'].rolling(21).mean().shift()

stock['low1'] = stock['Low'].rolling(1).mean()
stock['high1'] = stock['High'].rolling(1).mean()

#Adding bollinger bands

stock['20_MA'] = stock['Close'].rolling(window=20).mean().shift()
stock['20_STD'] = stock['Close'].rolling(window=20).std().shift()

stock['bb_high'] = stock['20_MA'] + (stock['20_STD'] * 2) #Change these 2s to change the number of standard deviations of the BBs.
stock['bb_mid'] = stock['20_MA']
stock['bb_low'] = stock['20_MA'] - (stock['20_STD'] * 2)


#Drops first days where moving averages have not formed.
stock.dropna(inplace=True)




# Initialize signal and position columns
stock['signal'] = 0
if 'signal' not in stock.columns:
    stock['signal'] = 0

stock['position'] = 0
stock['action'] = 0


stock.dropna(inplace=True)


#print(stock.dtypes)
#print(stock.index)


# Generate signals based on Bollinger Bands
for i in range(1, len(stock)):
    low = stock['low1'].iloc[i]  # Access scalar value
    bb_low = stock['bb_low'].iloc[i]
    high = stock['high1'].iloc[i]
    bb_mid = stock['bb_mid'].iloc[i]
    #print(f"Low: {low}, bb_low: {bb_low}, High: {high}, bb_mid: {bb_mid}")


    if low < bb_low:  # Buy signal
        stock.loc[stock.index[i], 'signal'] = 1
    elif high > bb_mid:  # Sell signal
        stock.loc[stock.index[i], 'signal'] = -1
    else:
        stock.loc[stock.index[i], 'signal'] = stock['signal'].iloc[i - 1]

 # Update positions based on signals
 #for i in range(1, len(stock)):
    current_signal = stock['signal'].iloc[i]

    if current_signal == 1:  
        stock.loc[stock.index[i], 'position'] = 1   #Rolling = 1 while in a position
    elif current_signal == -1: 
        stock.loc[stock.index[i], 'position'] = 0   #Rolling = 0 while not in position
    else:
        stock.loc[stock.index[i], 'position'] = stock['position'].iloc[i - 1] #If price doesn't cross critical points then it stays the same

    stock['action'] = stock.position.diff()



 #for i in range(1, len(stock)):
    current_action = stock['action'].iloc[i]

    if current_action == 1:
        stock.loc[stock.index[i], 'BUY'] = stock['action'].iloc[i] * stock['bb_low'].iloc[i] * -1
        stock.loc[stock.index[i], 'SELL'] = 0
    elif current_action == -1:
        stock.loc[stock.index[i], 'SELL'] = stock['action'].iloc[i] * stock['bb_mid'].iloc[i] * -1
        stock.loc[stock.index[i], 'BUY'] = 0
    else:
        stock.loc[stock.index[i], 'BUY'] = 0
        stock.loc[stock.index[i], 'SELL'] = 0


print(stock[40:70])


#print(stock['Close'].iloc[len(stock)])
#print(stock['Close'].iloc[len(stock)-252])


plt.rcParams['figure.figsize'] = 12, 6
plt.grid(True, alpha = .3)
plt.plot(stock.iloc[-252:]['Close'], label = 'stock')

#Plotting Moving Averages
#plt.plot(stock.iloc[-252:]['9-day'], label = '9-day')
#plt.plot(stock.iloc[-252:]['21-day'], label = '21-day')

plt.plot(stock.iloc[-252:]['bb_high'], label = 'bb_upper')
plt.plot(stock.iloc[-252:]['bb_mid'], label = 'bb_mean')
plt.plot(stock.iloc[-252:]['bb_low'], label = 'bb_lower')

plt.plot(stock[-252:].loc[stock.action == 1].index, stock[-252:]['bb_low'][stock.action == 1], '^', color = 'g', markersize = 12)
plt.plot(stock[-252:].loc[stock.action == -1].index, stock[-252:]['bb_mid'][stock.action == -1], '^', color = 'r', markersize = 12)
plt.legend(loc=2);

plt.show()


days_back = 252 #Must match the number above
system_return = 0
hold_return = 0
start_price = 0
end_price = 0

stock['close1'] = stock['Close'].rolling(1).mean()

for i in range(len(stock) - days_back, len(stock)):
    system_return = system_return + stock['BUY'].iloc[i]*-1 + stock['SELL'].iloc[i]*-1
    if i == (len(stock) - days_back):
        start_price = stock['close1'].iloc[i]
    elif i == (len(stock)-1):
        end_price = stock['close1'].iloc[i]
    else:
        start_price = start_price
        end_price = end_price


hold_return = end_price - start_price

#plt.plot('Buy and Hold Return', hold_return, label = 'Buy and Hold Return', color = 'blue')
#plt.plot('System Return', system_return, label = 'System Return', color = 'green')
#plt.title("Buy and Hold vs System Returns")
#plt.xlabel("Date")
#plt.ylabel("Overall Return")
#plt.legend()
#plt.grid(True)


plt.bar('Buy and Hold',  hold_return, color=['blue'])
plt.bar('System', system_return, color=['green'])
plt.title('Buy and Hold vs System Returns')
plt.ylabel('Absolute Return (USD)')

plt.show()



#plt.plot(np.exp(stock['return']).cumprod(), label = 'Buy/Hold')
#plt.plot(np.exp(stock['system_return']).cumprod(), label = 'System')
#plt.legend(loc = 2)
#plt.grid(True, alpha=.3)
#plt.show()

print(f'\n\nThe system return is {system_return}\n\n')
print(f'\nThe start price is {start_price}.\nThe end price is {end_price}.\n')
print(f'\n\nThe buy and hold return is {hold_return}\n\n')