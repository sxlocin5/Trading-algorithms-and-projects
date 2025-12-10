import numpy as np
import pandas as pd
import pandas_datareader as pdr

from datetime import datetime, time

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import yfinance as yf
#3SIL.L

ticker = 'ETH-USD'          #input('Input ticker symbol \n\n')
date_start = '2025-09-07'   #input('Input the start date in the form YYYY-MM-DD \n\n')
timeframe = '5M'            #input('Input the desired time interval \n\n')

stock = yf.download(ticker.upper(), start = date_start, interval = timeframe.lower(), auto_adjust=True)

stock['High1'] = stock['High'].rolling(1).mean()
stock['Low1'] = stock['Low'].rolling(1).mean()
stock['Open1'] = stock['Open'].rolling(1).mean()
stock['Close1'] = stock['Close'].rolling(1).mean()

stock['20_MA'] = stock['Close'].rolling(window=20).mean().shift()
stock['20_STD'] = stock['Close'].rolling(window=20).std().shift()


close_array = stock['Close1'].values
open_array = stock['Open1'].values
low_array = stock['Low1'].values
high_array = stock['High1'].values

trade_dates = []   
trade_times_entry = [] 
trade_positions = [] 
trade_outcomes = []
trade_times_exit = []
returns = []

trade_complete = False
strategy_active = False
risk = 0

# Tunable parameters

rsi_period = 5
rsi_lookback_period = 5
rsi_threshold_high = 60
rsi_threshold_low = 40

SL1_lookback_period = 3
SL1_ratio = 1
TP1_ratio = 4.5
TP2_ratio = 4.5
first_exit_ratio = 0.5
second_exit_ratio = 1 - first_exit_ratio

trailing_stop_counter = np.arange(1, TP2_ratio + 1, 0.5)
print(trailing_stop_counter)

# Trading window
start_time = time(8, 0)   # 08:00
end_time   = time(20, 0)  # 18:00
close_position_time = time(19, 50)

signal_long, position_long, overshoot_long = 0,0,0
signal_short, position_short, overshoot_short = 0,0,0


rsi_gains = 0
rsi_losses = 0

prev_closes = close_array[0 : rsi_period]

total_return = 0
num_trades = 0
num_wins = 0
num_losses = 0

num_short = 0
num_short_wins = 0
num_short_losses = 0

num_long = 0
num_long_wins = 0
num_long_losses = 0

num_tp2 = 0
num_sl1 = 0
num_sl2 = 0
num_reverses = 0
num_be = 0
num_eod = 0

drawdown_current = 0
drawdown_max = 0
loss_streak_current = 0
loss_streak_max = 0

profit_current = 0
profit_max = 0
win_streak_current = 0
win_streak_max = 0

#=== RSI calculations ===

for i in range(1, rsi_period):
    delta = prev_closes[i] - prev_closes[i-1]
    if delta > 0:
            rsi_gains += delta
    if delta < 0:
            rsi_losses += abs(delta)

if rsi_gains == 0 and rsi_losses == 0:
    rsi_start = 50
    rsi_avg_loss, rsi_avg_gain = 0,0

elif rsi_gains == 0:
    rsi_start = 0 
    rsi_avg_gain = 0
    rsi_avg_loss = rsi_losses/rsi_period

elif rsi_losses == 0:
    rsi_start = 100
    rsi_avg_loss = 0
    rsi_avg_gain = rsi_gains/rsi_period

else:     
    rsi_avg_gain = rsi_gains/rsi_period
    rsi_avg_loss = rsi_losses/rsi_period

    rs_start = rsi_avg_gain/rsi_avg_loss
    rsi_start = 100 - (100/(1+rs_start))

rsi_array = np.zeros(len(stock) - rsi_period)
rsi_array[0] = rsi_start


#=== Main Loop ===

for i in range(rsi_period + 1, len(stock)):
    
    #--Defining RSI--
    low = low_array[i]
    high = high_array[i]
    open = open_array[i]
    close = close_array[i]
    prev_close = close_array[i-1]

    delta = close - prev_close

    if delta > 0:
        rsi_avg_gain = ((rsi_avg_gain * (rsi_period - 1)) + delta) / rsi_period
        rsi_avg_loss = ((rsi_avg_loss * (rsi_period - 1)) + 0) / rsi_period
    elif delta < 0:
        rsi_avg_gain = ((rsi_avg_gain * (rsi_period - 1)) + 0) / rsi_period
        rsi_avg_loss = ((rsi_avg_loss * (rsi_period - 1)) + abs(delta)) / rsi_period
    elif delta == 0:
        rsi_avg_gain = ((rsi_avg_gain * (rsi_period - 1)) + delta) / rsi_period
        rsi_avg_loss = ((rsi_avg_loss * (rsi_period - 1)) + delta) / rsi_period

    if rsi_avg_gain == 0 and rsi_avg_loss == 0:
        rsi_local = 50
    elif rsi_avg_gain == 0:
        rsi_local = 0
    elif rsi_avg_loss == 0:
        rsi_local = 100
    else: 
        RS = rsi_avg_gain/rsi_avg_loss
        rsi_local = 100 - (100/(1+RS))


    rsi_array[i - rsi_period] = rsi_local

    if (i - rsi_period) < rsi_lookback_period:
        rsi_lookback_min = rsi_array[0 : i].min()
        rsi_lookback_max = rsi_array[0: i].max()
    else:
        rsi_lookback_min = rsi_array[i - rsi_period - rsi_lookback_period : i - rsi_period].min() #First value in inclusive, second boundary is exclusive
        rsi_lookback_max = rsi_array[i - rsi_period - rsi_lookback_period : i - rsi_period].max()

    SL1_lookback = open_array[i - SL1_lookback_period : i]

    current_time = stock.index[i].time()

    #=== Strategy runs if within defined trading hours ===
    if start_time <= current_time <= end_time:

        if risk == 0:
            position_long, overshoot_long, position_short, overshoot_short = 0,0,0,0

        #--Long exit conditions--
        if position_long == 1:
            if low <= stoploss:
                position_long, overshoot_long = 0, 0
                exit_price1_long = stoploss
                
                trade_outcome = 'SL1'
                gain = -1
                exit_time = stock.index[i].time()
                trade_complete = True
                
            elif rsi_local >= rsi_threshold_high:
                position_long, overshoot_long = 0, 0
                exit_price1_long = close

                trade_outcome = 'Reversed'
                gain = (close-entry_price_long)/(risk)
                exit_time = stock.index[i].time()
                trade_complete = True
                
            elif high >= tp1:
                position_long = 0
                exit_price1_long = tp1
                stoploss = entry_price_long #can make this variable entry price +5% R
                counter = 1
                trailing_stop_trigger = upside * trailing_stop_counter[counter] + entry_price_long

        
        if position_long == 0 and overshoot_long == 1:
            if low <= stoploss:
                overshoot_long = 0
                exit_price2_long = stoploss

                trade_outcome = 'SL2'
                exit_time = stock.index[i].time()
                trade_complete = True
                if risk == 0 or np.isnan(risk):
                    gain = np.nan  # or 0, or skip trade
                else:
                    gain = ((tp1-entry_price_long)/risk)*first_exit_ratio + ((stoploss - entry_price_long)/risk)*second_exit_ratio
              
            elif high >= tp2:
                exit_price2_long = tp2
                overshoot_long = 0

                trade_outcome = 'TP2'
                gain = ((tp1-entry_price_long)/risk)*first_exit_ratio + ((tp2-entry_price_long)/risk)*second_exit_ratio
                exit_time = stock.index[i].time()
                trade_complete = True
            
                # --Trailing stoploss --            
            elif high >= trailing_stop_trigger and trade_complete == False:
                stoploss = trailing_stop_trigger - upside
                counter += 1
                trailing_stop_trigger = upside * trailing_stop_counter[counter] + entry_price_long
            

        #--Short exit conditions-- 
        if position_short == 1:
            if high >= stoploss:
                position_short, overshoot_short = 0, 0
                exit_price1_short = stoploss

                trade_outcome = 'SL1'
                gain = -1
                exit_time = stock.index[i].time()
                trade_complete = True
                
            elif rsi_local <= rsi_threshold_low:
                position_short, overshoot_short = 0, 0
                exit_price1_short = close

                trade_outcome = 'Reversed'
                gain = (entry_price_short-close)/(risk)
                exit_time = stock.index[i].time()
                trade_complete = True
                
            elif low <= tp1:
                position_short = 0
                exit_price1_short = tp1
                stoploss = entry_price_short #can make this variable like entry price +5% R
                counter = 1
                trailing_stop_trigger = upside * trailing_stop_counter[counter] + entry_price_short
        
        if position_short == 0 and overshoot_short == 1:
            if low <= stoploss:
                overshoot_short = 0
                exit_price2_short = stoploss

                trade_outcome = 'SL2'
                exit_time = stock.index[i].time()
                trade_complete = True
                if risk == 0 or np.isnan(risk):
                    gain = np.nan  # or 0, or skip trade
                else:
                    gain = ((entry_price_short-tp1)/risk)*first_exit_ratio + ((entry_price_short-stoploss)/risk)*second_exit_ratio
                
            elif low <= tp2:
                exit_price2_short = tp2
                overshoot_short = 0

                trade_outcome = 'TP2'
                gain = ((entry_price_short-tp1)/risk)*first_exit_ratio + ((entry_price_short-tp2)/risk)*second_exit_ratio
                exit_time = stock.index[i].time()
                trade_complete = True

                # --Trailing stoploss --
            elif low <= trailing_stop_trigger and trade_complete == False:
                stoploss = trailing_stop_trigger - upside
                counter += 1
                trailing_stop_trigger = upside * trailing_stop_counter[counter] + entry_price_short
        
        #--Long entry--
        if signal_long == 1 and position_long == 0 and overshoot_long == 0: 
            position_long = 1
            overshoot_long = 1
            signal_long = 0
            trade_position = 'Long'

            entry_date = stock.index[i].date()
            entry_time = stock.index[i].time()
            entry_price_long = open
            swing_high = SL1_lookback.max()
            stoploss = entry_price_long - (swing_high - entry_price_long)*SL1_ratio
            risk = (swing_high - entry_price_long)*SL1_ratio
            tp1 = (swing_high - entry_price_long)*TP1_ratio + entry_price_long
            tp2 = (swing_high - entry_price_long)*TP2_ratio + entry_price_long
            
            upside =  swing_high - entry_price_long 
            

        #--Short entry--
        if signal_short == 1 and position_short == 0 and overshoot_short == 0: 
            position_short = 1
            overshoot_short = 1
            signal_short = 0
            trade_position = 'Short'

            entry_date = stock.index[i].date()
            entry_time = stock.index[i].time()
            entry_price_short = open
            swing_low = SL1_lookback.min()
            stoploss = (entry_price_short - swing_low)*SL1_ratio + entry_price_short
            risk = (entry_price_short - swing_low)*SL1_ratio
            tp1 = entry_price_short - (entry_price_short  - swing_low)*TP1_ratio
            tp2 = entry_price_short - (entry_price_short  - swing_low)*TP2_ratio

            upside = entry_price_short  - swing_low

        # --Long signal--
        if rsi_local <= rsi_threshold_low and rsi_lookback_max > 50 and position_long == 0 and overshoot_long == 0 and position_short == 0 and overshoot_short == 0: #
            signal_long = 1
            
        # --Short signal--
        if rsi_local >= rsi_threshold_high and rsi_lookback_min < 50 and position_short == 0 and overshoot_short == 0 and position_long == 0 and overshoot_long == 0: # 
            signal_short = 1   

        # --End of trading hours logic--
        if (overshoot_short or overshoot_long == 1) and current_time == close_position_time:
            exit_time = stock.index[i].time()
            trade_outcome = 'EOD'
            trade_complete = True

            if overshoot_long == 1:
                if position_long == 1:
                    gain = (close - entry_price_long)/risk
                else:
                    gain = ((tp1 - entry_price_long)/risk)*first_exit_ratio + ((close - entry_price_long)/risk)*second_exit_ratio

            if overshoot_short == 1:
                if position_short == 1:
                    gain = (entry_price_short - close)/risk
                else:
                    gain = ((entry_price_short - tp1)/risk)*first_exit_ratio + ((entry_price_short - close)/risk)*second_exit_ratio
            overshoot_short, overshoot_long, position_short, position_long = 0,0,0,0

        # --Logging trade data when trade closes--
        if trade_complete == True:
            trade_dates.append(entry_date)   # date
            trade_times_entry.append(entry_time)   # time (if you’re using intraday data)
            trade_positions.append(trade_position)
            trade_outcomes.append(trade_outcome)    
            trade_times_exit.append(exit_time)
            returns.append(gain)              # % return for that trade in terms of risk

            num_trades += 1
            total_return += gain


            # --General trade statistics logging--

            if trade_outcome == 'SL1':
                num_losses += 1
                num_sl1 += 1
            elif trade_outcome == 'SL2':
                num_wins += 1
                num_sl2 += 1
            elif trade_outcome == 'TP2':
                num_wins += 1
                num_tp2 += 1
            elif trade_outcome == 'Reversed':
                num_reverses += 1
                if gain < 0:
                    num_losses += 1
                elif gain > 0:
                    num_wins += 1
                elif gain == 0:
                    num_be += 1
            elif trade_outcome == 'EOD':
                num_eod += 1
                if gain < 0:
                    num_losses += 1
                elif gain > 0:
                    num_wins += 1
                elif gain == 0:
                    num_be += 1

            # --Long/Short trade statistics logging--

            if trade_position == 'Long':
                num_long += 1
                if gain <= 0:
                    num_long_losses += 1
                elif gain > 0:
                    num_long_wins += 1

            elif trade_position == 'Short':
                num_short += 1
                if gain <= 0:
                    num_short_losses += 1
                elif gain > 0:
                    num_short_wins += 1

            # --Drawdown/winstreak logging--

            if gain >= 0:
                win_streak_current += 1
                profit_current += gain

                loss_streak_current = 0
                drawdown_current = 0

                if win_streak_current > win_streak_max:
                    win_streak_max = win_streak_current
                if profit_current > profit_max:
                    profit_max = profit_current

            elif gain < 0:
                loss_streak_current += 1
                drawdown_current += gain

                win_streak_current = 0
                profit_current = 0

                if loss_streak_current > loss_streak_max:
                    loss_streak_max = loss_streak_current
                if drawdown_current < drawdown_max:
                    drawdown_max = drawdown_current

            trade_complete = False
         

trades_df = pd.DataFrame({
    "Date": trade_dates,
    "Entry Time": trade_times_entry,
    "Position": trade_positions,
    "Exit Time": trade_times_exit,
    "Outcome": trade_outcomes,
    "Return (Risk)": returns
})


print(trades_df)

#=== Calculating strategy statistics ===

winrate = np.round((num_wins/num_trades) * 100, 2)
long_winrate = np.round((num_long_wins/num_long)*100, 2)
short_winrate = np.round((num_short_wins/num_short)*100, 2)

expected_value = np.round(total_return/num_trades, 3)

start_date2 = datetime.strptime(date_start, "%Y-%m-%d").date()
delta = datetime.today().date() - start_date2
num_days = delta.days

trades_per_day = num_trades / num_days
return_per_day = total_return / num_days

print(f'\nThe backtest took place over {num_days} days')
print(f'\nTotal return is: {np.round(total_return, 3)}*R')
print(f'\nTotal number of trades is {num_trades}')
print(f'\nThe strategy winrate is {winrate}%     W {num_wins} / L {num_losses}')
print(f'\nNumber of Long and Short trades is:     {num_long} L / {num_short} S')
print(f'\nThe long trade winrate is {long_winrate}%, the short trade winrate is {short_winrate}%')
print(f'\nThe expected return per trade is {expected_value}*R')
print(f'\nMax drawdown:        {np.round(drawdown_max,2)}*R        Biggest loss streak: {loss_streak_max}')
print(f'\nMax profit streak:   {np.round(profit_max,2)}*R        Biggest win streak:  {win_streak_max}')
print(f'\nAverage number of trades per day: {np.round(trades_per_day,1)}      Average return per day: {np.round(return_per_day,2)}*R')


# --- Parameters ---
N_days = 3   # how many days back to show
start_date = stock.index[-1] - pd.Timedelta(days=N_days)

# Slice stock data
stock_slice = stock.loc[stock.index >= start_date]

# Align RSI slice with stock_slice (skip warmup period)
rsi_slice = rsi_array[-len(stock_slice[rsi_period:]):]

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

# Price chart
ax1.set_title(f'{ticker} (Last {N_days} days)')
ax1.plot(stock_slice.index, stock_slice['High1'], label='High', color='blue')
ax1.plot(stock_slice.index, stock_slice['Low1'], label='Low', color="#679AE6")
ax1.set_ylabel('Price')
ax1.legend(loc="upper left")

# RSI chart
ax2.plot(stock_slice.index[rsi_period:], rsi_slice, label='RSI', color='grey')
ax2.axhline(rsi_threshold_high, color='green', linestyle='--', linewidth=1, label='RSI High')
ax2.axhline(75, color='green', linestyle='--', linewidth=1)
ax2.axhline(rsi_threshold_low, color='red', linestyle='--', linewidth=1, label='RSI Low')
ax2.axhline(25, color='red', linestyle='--', linewidth=1)
ax2.set_ylabel('RSI')
ax2.set_xlabel('Date')
ax2.set_ylim(0, 100)
ax2.legend(loc="upper left")

plt.tight_layout()

plt.show()