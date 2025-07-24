import numpy as np
import matplotlib.pyplot as plt

asset = 'S&P 500'

num_days = 252*5
num_sims = 1000

start_price = 1000
mean_return = 0.0327/100
std_dev = 0.975/100

price_paths = np.zeros((num_days + 1, num_sims))
price_paths[0] = start_price

for i in range(1, num_days + 1):

    shock = np.random.normal(mean_return, std_dev, num_sims)
    price_paths[i] = price_paths[i-1] * (1+shock)

final_prices = price_paths[-1]
mean_final_price = np.mean(final_prices)

final_returns = ((price_paths - start_price)/start_price)*100
avg_return = np.mean(final_returns)

print('\n\nThe average final price is £', np.round(mean_final_price, 2), '\n\n')
print('The average final return is', np.round(avg_return, 2),'%')

plt.plot(price_paths)
plt.ylabel('Asset Price')
plt.title('\nMonte Carlo Simulation of {asset}\n')
plt.show()