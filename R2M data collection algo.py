
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from ta.volatility import BollingerBands

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import yfinance as yf


ticker = 'AG' #input('Input ticker symbol \n\n')
date_start = '2025-03-20'   #input('Input the start date in the form YYYY-MM-DD \n\n')
timeframe = '1d'    #input('Input the desired time interval \n\n')

stock = yf.download(ticker.upper(), start = date_start, interval = timeframe.lower() )



#Creating low and high columns used for comparisons for entry/exit signals

stock['High1'] = stock['High'].rolling(1).mean()
stock['Low1'] = stock['Low'].rolling(1).mean()

#Adding bollinger bands

stock['20_MA'] = stock['Close'].rolling(window=20).mean().shift()
stock['20_STD'] = stock['Close'].rolling(window=20).std().shift()

bb_std = 1

stock['BB_high'] = stock['20_MA'] + (stock['20_STD'] * bb_std) #Change these 2s to change the number of standard deviations of the BBs.
stock['BB_mid'] = stock['20_MA']
stock['BB_low'] = stock['20_MA'] - (stock['20_STD'] * bb_std)

#Adding signal column
#stock['signal'] = 0
#if 'signal' not in stock.columns:
  #  stock['signal'] = 0

#stock['position'] = 0
#stock['action'] = 0
stock['extremum'] = 0
stock['position'] = 0
stock['overshoot_pos'] = 0

extremum_sensitivity = 5 #Will compare to the previous and following 5 periods.
stock.dropna(inplace=True)

#Adding table for raw data collection

columns = ["Entry Date","Long/Short","Entry Price","Entry Index", "MR Price","MR Date", "MR Index", 
           "Extremum Price","Extremum Date","Extremum Index"]

stock_data = pd.DataFrame(columns=columns)

def add_result_row(df):
    df.loc[len(df)] = [None] * len(df.columns)
    return df

def update_last_row(df, column, value):
    df.at[len(df) - 1, column] = value
    return df


#Local max/min logic

for i in range(1 + extremum_sensitivity, len(stock) - extremum_sensitivity):
    low = stock['Low1'].iloc[i]  # Access scalar value
    high = stock['High1'].iloc[i]

    bb_mid = stock['BB_mid'].iloc[i]
    bb_low = stock['BB_low'].iloc[i]
    bb_high = stock['BB_high'].iloc[i]

    

    

    #if low = all(
     #   low < stock['low'].iloc[i-j] for j in range(1, extremum_sensitivity)
    #) and all(
    #    low < stock['low'].iloc[i+j] for j in range(1, extremum_sensitivity)
    #)

    #for j in range(extremum_sensitivity, len(stock) - extremum_sensitivity):
    # 



    #for i in range(extremum_sensitivity, len(stock) - extremum_sensitivity):             
    is_local_min = all(low < stock['Low1'].iloc[i - k] for k in range(1, extremum_sensitivity )) and \
                    all(low < stock['Low1'].iloc[i + k] for k in range(1, extremum_sensitivity ))

    is_local_max = all(high > stock['High1'].iloc[i - k] for k in range(1, extremum_sensitivity )) and \
                    all(high > stock['High1'].iloc[i + k] for k in range(1, extremum_sensitivity ))


    if is_local_min:
        stock.loc[stock.index[i], 'extremum'] = -1
    elif is_local_max:
        stock.loc[stock.index[i], 'extremum'] = 1
    else:
        stock.loc[stock.index[i], 'extremum'] = 0




#Long signal logic



for i in range(2,len(stock)):
    #stock.loc[stock.index[i], 'position'] = stock['position'].iloc[i - 1]
    
    position = stock['position'].iloc[i-1]
    overshoot = stock['overshoot_pos'].iloc[i-1]
    extremum = stock['extremum'].iloc[i]
    
    low = stock['Low1'].iloc[i]  # Access scalar value
    high = stock['High1'].iloc[i]

    bb_mid = stock['BB_mid'].iloc[i]
    bb_low = stock['BB_low'].iloc[i]
    bb_high = stock['BB_high'].iloc[i]

    #Long signal logic
    if low <= bb_low and extremum == -1:
        position = 1
        overshoot = 1
        entry_price_long = low
        entry_date_long = stock.index[i]
        entry_index_long = i


    if position == 0 and overshoot == 1:
        
        if high > highest_price:
            highest_price = high
            highest_date = stock.index[i]
            highest_index = i 

        if low < mean_price_long:
            overshoot = 0
            max_price = highest_price         

            stock_data = add_result_row(stock_data)
            stock_data = update_last_row(stock_data, "Entry Date", entry_date_long)
            stock_data = update_last_row(stock_data,"Long/Short", 'Long')
            stock_data = update_last_row(stock_data,'Entry Price', entry_price_long)
            stock_data = update_last_row(stock_data, "Entry Index", entry_index_long)
            stock_data = update_last_row(stock_data, "MR Price", mean_price_long)
            stock_data = update_last_row(stock_data, "MR Date", mean_date_long)
            stock_data = update_last_row(stock_data, "MR Index", mean_index_long)
            stock_data = update_last_row(stock_data, "Extremum Price", max_price)
            stock_data = update_last_row(stock_data, "Extremum Date", highest_date)
            stock_data = update_last_row(stock_data, "Extremum Index", highest_index)
  

        elif extremum == 1 and overshoot == 1:
            max_price = high
            max_date = stock.index[i]
            max_index = i    

            stock_data = add_result_row(stock_data)
            stock_data = update_last_row(stock_data, "Entry Date", entry_date_long)
            stock_data = update_last_row(stock_data,"Long/Short", 'Long')
            stock_data = update_last_row(stock_data,'Entry Price', entry_price_long)
            stock_data = update_last_row(stock_data, "Entry Index", entry_index_long)
            stock_data = update_last_row(stock_data, "MR Price", mean_price_long)
            stock_data = update_last_row(stock_data, "MR Date", mean_date_long)
            stock_data = update_last_row(stock_data, "MR Index", mean_index_long)
            stock_data = update_last_row(stock_data, "Extremum Price", max_price)
            stock_data = update_last_row(stock_data, "Extremum Date", max_date)
            stock_data = update_last_row(stock_data, "Extremum Index", max_index)     

    if position == 1:

        if low < entry_price_long:
            position = 0
            overshoot = 0

        elif high >= bb_mid:
            mean_price_long = bb_mid #potential issue with these values updating once it has gone below the bb_mid
            mean_date_long = stock.index[i] #need to find a way to store once then not update until next position
            mean_index_long = i
            position = 0
            highest_price = high
            highest_date = stock.index[i]
            highest_index = i        


    
    #Short signal logic
    #Entry same, add overshoot. For low goes below bbmid, make position zero
    if high >= bb_high and extremum == 1:
        position = -1
        overshoot = -1
        entry_price_short = high
        entry_date_short = stock.index[i]
        entry_index_short = i 

    if position == 0 and overshoot == -1:
        

        if low < lowest_price:
            lowest_price = low
            lowest_date = stock.index[i]
            lowest_index = i

        if high > mean_price_short:
            overshoot = 0
            min_price = lowest_price

            stock_data = add_result_row(stock_data)
            stock_data = update_last_row(stock_data, "Entry Date", entry_date_short)
            stock_data = update_last_row(stock_data,"Long/Short", 'Short')
            stock_data = update_last_row(stock_data,'Entry Price', entry_price_short)
            stock_data = update_last_row(stock_data, "Entry Index", entry_index_short)
            stock_data = update_last_row(stock_data, "MR Price", mean_price_short)
            stock_data = update_last_row(stock_data, "MR Date", mean_date_short)
            stock_data = update_last_row(stock_data, "MR Index", mean_index_short)
            stock_data = update_last_row(stock_data, "Extremum Price", min_price)
            stock_data = update_last_row(stock_data, "Extremum Date", lowest_date)
            stock_data = update_last_row(stock_data, "Extremum Index", lowest_index)

        elif extremum == -1 and overshoot == -1:
            min_price = low
            min_date = stock.index[i]
            min_index = i        

            stock_data = add_result_row(stock_data)
            stock_data = update_last_row(stock_data, "Entry Date", entry_date_short)
            stock_data = update_last_row(stock_data,"Long/Short", 'Short')
            stock_data = update_last_row(stock_data,'Entry Price', entry_price_short)
            stock_data = update_last_row(stock_data, "Entry Index", entry_index_short)
            stock_data = update_last_row(stock_data, "MR Price", mean_price_short)
            stock_data = update_last_row(stock_data, "MR Date", mean_date_short)
            stock_data = update_last_row(stock_data, "MR Index", mean_index_short)
            stock_data = update_last_row(stock_data, "Extremum Price", min_price)
            stock_data = update_last_row(stock_data, "Extremum Date", min_date)
            stock_data = update_last_row(stock_data, "Extremum Index", min_index)                             
        
        
    if position == -1:

        if high > entry_price_short:
            position = 0
            overshoot = 0
        #if high == entry_price_short:
        #    entry_price_short = high
        #    entry_date_short = stock.index[i]
        #    entry_index_short = i    

        elif low <= bb_mid:
            mean_price_short = bb_mid #potential issue with these values updating once it has gone below the bb_mid
            mean_date_short = stock.index[i] #need to find a way to store once then not update until next position
            mean_index_short = i
            position = 0
            lowest_price = low
            lowest_date = stock.index[i]
            lowest_index = i
            
    stock.loc[stock.index[i], 'position'] = position
    stock.loc[stock.index[i], 'overshoot_pos'] = overshoot          
                
        
#Short position - goes to mean, but then no local min and goes above entry price so results not logged.                
#Bollinger band data keeps updating even after it crosses bb_mid (need 2 entry/exit signals?)
#One which switches off when crosses bb_mid, one that stays on then turns off at next maxima or if price>entry.

stock['D'] = abs(stock.overshoot_pos.diff())

plt.figure(figsize=(15, 5))
plt.plot(stock['High1'], label='High', color = 'blue')
plt.plot(stock['Low1'], label = 'Low', color = "#679AE6")
plt.plot(stock['BB_high'], color = 'orange')
plt.plot(stock['BB_low'], color = 'orange')
plt.plot(stock['BB_mid'], color = "#A4A4A4")
plt.scatter(stock[stock['D'] == 1].index, stock[stock['D'] == 1]['Low1'], color = 'purple', label = 'End of trade')
plt.scatter(stock[stock['extremum'] == 1].index, stock[stock['extremum'] == 1]['High1'], color='red', label='Local Max')
plt.scatter(stock[stock['extremum'] == -1].index, stock[stock['extremum'] == -1]['Low1'], color='green', label='Local Min')
plt.legend()
plt.show()



print(stock.head(60))

print('The stock data is\n', stock_data)

stock_data.to_csv("results_with_index.csv", index=True)