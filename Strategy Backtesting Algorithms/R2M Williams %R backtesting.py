import numpy as np
import pandas as pd
import pandas_datareader as pdr
from ta.volatility import BollingerBands

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import yfinance as yf
#3SIL.L

ticker = '3SIL.L' #input('Input ticker symbol \n\n')
date_start = '2023-03-20'   #input('Input the start date in the form YYYY-MM-DD \n\n')
timeframe = '1d'    #input('Input the desired time interval \n\n')

stock = yf.download(ticker.upper(), start = date_start, interval = timeframe.lower() )

stock['High1'] = stock['High'].rolling(1).mean()
stock['Low1'] = stock['Low'].rolling(1).mean()
stock['Open1'] = stock['Open'].rolling(1).mean()
stock['Close1'] = stock['Close'].rolling(1).mean()

stock['20_MA'] = stock['Close'].rolling(window=20).mean().shift()
stock['20_STD'] = stock['Close'].rolling(window=20).std().shift()

bb_std = 2
williams_window = 14
williams_threshold = -80
stoploss_percentage = 5.5
stoploss_trailing_percentage = 6

stock['BB_high'] = stock['20_MA'] + (stock['20_STD'] * bb_std) #Change these 2s to change the number of standard deviations of the BBs.
stock['BB_mid'] = stock['20_MA']
stock['BB_low'] = stock['20_MA'] - (stock['20_STD'] * bb_std)

stock['position'] = 0
stock['overshoot'] = 0
stock['signal'] = 0
stock['W%R'] = 0


columns = ["Entry Date","Long/Short","Exit", "Entry Price","Entry Index", "MR Price","MR Date", "MR Index", 
           "Exit Price","Exit Date","Exit Index", "Gain%"]

stock_data = pd.DataFrame(columns=columns)

def add_result_row(df):
    df.loc[len(df)] = [None] * len(df.columns)
    return df

def update_last_row(df, column, value):
    df.at[len(df) - 1, column] = value
    return df


#def calculate_williams_r(stock, williams_window = williams_window):
#    high_w = stock['High'].rolling(window = williams_window).max()
#    low_w = stock['Low'].rolling(window = williams_window).min()
#    williams_r = -100 * ((high_w - stock['Close']) / (high_w - low_w))
#    stock[f'Williams_%R_{williams_window}'] = williams_r
#    return stock



stock.dropna(inplace=True)

for i in range(1, len(stock)):

    low = stock['Low1'].iloc[i]  # Accessing scalar values
    high = stock['High1'].iloc[i]
    close = stock['Close1'].iloc[i]
    open = stock['Open1'].iloc[i]

    bb_mid = stock['BB_mid'].iloc[i]
    bb_low = stock['BB_low'].iloc[i]
    bb_high = stock['BB_high'].iloc[i]

    position = stock['position'].iloc[i-1]
    overshoot = stock['overshoot'].iloc[i-1]
    signal = stock['signal'].iloc[i]
    prev_signal = stock['signal'].iloc[i-1]

    high_n = stock['High1'].iloc[i - williams_window + 1 : i+1].max() #Check range to make sure it's calculating only using up to today's data
    low_n = stock['Low1'].iloc[i - williams_window +1 : i+1].min() #Includes lower value, excludes upper value 
    williams_r = -100 * ((high_n - close) / (high_n - low_n))

    #Long logic

    if prev_signal == 1:
        signal = 0
        entry_price_long = open
        entry_date_long = stock.index[i]
        entry_index_long = i

        stoploss = entry_price_long * (1-(stoploss_percentage/100))

    if position == 1:
        if low <= stoploss:
            position = 0
            overshoot = 0
            sell_stoploss_date = stock.index[i]
            sell_stoploss_index = i

            if open < stoploss:
                sell_stoploss = open
            else:
                sell_stoploss = stoploss

            gain = ((sell_stoploss - entry_price_long)/entry_price_long)*100

            stock_data = add_result_row(stock_data)
            stock_data = update_last_row(stock_data, "Entry Date", entry_date_long)
            stock_data = update_last_row(stock_data,"Long/Short", 'Long')
            stock_data = update_last_row(stock_data,'Exit', 'Stoploss')
            stock_data = update_last_row(stock_data,'Entry Price', entry_price_long)
            stock_data = update_last_row(stock_data, "Entry Index", entry_index_long)
            stock_data = update_last_row(stock_data, "MR Price", '-')
            stock_data = update_last_row(stock_data, "MR Date", '-')
            stock_data = update_last_row(stock_data, "MR Index", '-')
            stock_data = update_last_row(stock_data, "Exit Price", sell_stoploss)
            stock_data = update_last_row(stock_data, "Exit Date", sell_stoploss_date)
            stock_data = update_last_row(stock_data, "Exit Index", sell_stoploss_index)        
            stock_data = update_last_row(stock_data, "Gain%", gain) 

        elif high >= bb_mid:
            mean_price = bb_mid
            mean_date = stock.index[i]
            mean_index = i
            
            highest_low = low
            position = 0

            stoploss = low * (1-(stoploss_trailing_percentage/100))
            

    if position == 0 and overshoot == 1:
        if low <= stoploss:
            sell_stoploss_trailing = stoploss
            overshoot = 0
            sell_trailing_date = stock.index[i]
            sell_trailing_index = i
            gain = 0.5*(((mean_price-entry_price_long)/entry_price_long)+((sell_stoploss_trailing-entry_price_long)/entry_price_long))*100

            stock_data = add_result_row(stock_data)
            stock_data = update_last_row(stock_data, "Entry Date", entry_date_long)
            stock_data = update_last_row(stock_data,"Long/Short", 'Long')
            stock_data = update_last_row(stock_data,'Exit', 'Trailing')
            stock_data = update_last_row(stock_data,'Entry Price', entry_price_long)
            stock_data = update_last_row(stock_data, "Entry Index", entry_index_long)
            stock_data = update_last_row(stock_data, "MR Price", mean_price)
            stock_data = update_last_row(stock_data, "MR Date", mean_date)
            stock_data = update_last_row(stock_data, "MR Index", mean_index)
            stock_data = update_last_row(stock_data, "Exit Price", sell_stoploss_trailing)
            stock_data = update_last_row(stock_data, "Exit Date", sell_trailing_date)
            stock_data = update_last_row(stock_data, "Exit Index", sell_trailing_index)
            stock_data = update_last_row(stock_data, "Gain%", gain) 

        elif low > highest_low:
            highest_low = low
            stoploss = highest_low * (1-(stoploss_trailing_percentage/100))
        
    if williams_r < williams_threshold and position == 0 and overshoot == 0:
        position = 1
        overshoot = 1
        signal = 1
    
    #Make sure to write signal variables back to dataframe:
    stock.loc[stock.index[i], 'position'] = position
    stock.loc[stock.index[i], 'overshoot'] = overshoot
    stock.loc[stock.index[i], 'signal'] = signal
    stock.loc[stock.index[i], 'W%R'] = williams_r

total_gain = 0
total_loss = 0
no_wins =0
no_losses = 0

for i in range(0, len(stock_data)):
    if stock_data.loc[stock_data.index[i], 'Exit'] == 'Stoploss':
        total_loss = total_loss + stock_data.loc[stock_data.index[i], 'Gain%']#((sell_stoploss-entry_price_long)/entry_price_long)*100
        no_losses = no_losses + 1
    if stock_data.loc[stock_data.index[i], 'Exit'] == 'Trailing':
        total_gain = total_gain + stock_data.loc[stock_data.index[i], 'Gain%']#0.5*(((mean_price-entry_price_long)/entry_price_long)+((sell_stoploss_trailing-entry_price_long)/entry_price_long))*100
        no_wins = no_wins + 1

total_return = total_gain + total_loss
win_ratio = round((no_wins/(no_wins+no_losses))*100, 3)
avg_gain_per_win = (total_gain/no_wins)
avg_loss_per_loss = (total_loss/no_losses)

stock['D'] = stock.overshoot.diff()

#plt.figure(figsize=(15, 5))
#plt.plot(stock['High1'], label='High', color = 'blue')
#plt.plot(stock['Low1'], label = 'Low', color = "#679AE6")
#plt.plot(stock['BB_high'], color = 'orange')
#plt.plot(stock['BB_low'], color = 'orange')
#plt.plot(stock['BB_mid'], color = "#A4A4A4")
#plt.scatter(stock[stock['D'] == -1].index, stock[stock['D'] == -1]['Low1'], color = 'purple', label = 'End of trade')
#plt.scatter(stock[stock['signal'] == 1].index, stock[stock['signal'] == 1]['Low1'], color='red', label='Entry')
#plt.legend()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

#Price chart
ax1.set_title(f'{ticker}')
ax1.plot(stock['High1'], label='High', color = 'blue')
ax1.plot(stock['Low1'], label = 'Low', color = "#679AE6")
ax1.plot(stock['BB_high'], color = 'orange')
ax1.plot(stock['BB_low'], color = 'orange')
ax1.plot(stock['BB_mid'], color = "#A4A4A4")
ax1.scatter(stock[stock['D'] == -1].index, stock[stock['D'] == -1]['Low1'], color = 'red', label = 'End of trade')
ax1.scatter(stock[stock['signal'] == 1].index, stock[stock['signal'] == 1]['Low1'], color='green', label='Entry')
ax1.set_ylabel('Price')

#Williams %R chart
ax2.plot(stock['W%R'], label = 'Williams %R', color = 'grey')
ax2.axhline(-20, color='red', linestyle='--', linewidth=1)
ax2.axhline(williams_threshold, color='green', linestyle='--', linewidth=1)
ax2.set_ylabel('Williams %R')
ax2.set_xlabel('Date')

plt.show()

print('\nThe total return is:', round(total_return, 3), '%')
print('\nThe win ratio is:', win_ratio, '%      ', no_wins, 'W / ', no_losses, 'L')
print('\nThe average gain per win is: ', round(avg_gain_per_win, 2), '%')
print('\nThe average return per loss is: ', round(avg_loss_per_loss, 2), '%')
print('\nThe total number of trades is: ', len(stock_data),'\n')

stock.dropna(inplace=True)
#print('\n\nThe stock data is:\n', stock_data)

#print(stock.head(20))

stock_data.to_csv("results_W%R_3SIL.csv", index=True)
