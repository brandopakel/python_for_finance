import math
import numpy as np
import scipy.stats as scs
import statsmodels.api as sm
import matplotlib.pylab as plt
from scipy_describe_function import print_statistics
from gen_paths_brownian import gen_paths
from normality_test import normality_tests
import pandas as pd

symbols = ['BTC', 'ETH', 'XRP', 'SOL']

raw_btc1m = pd.read_csv('/Users/bp/Documents/python_for_finance_work/BTC_1M_graph_coinmarketcap.csv', delimiter=';')
raw_eth1m = pd.read_csv('/Users/bp/Documents/python_for_finance_work/ETH_1M_graph_coinmarketcap.csv', delimiter=';')
raw_xrp1m = pd.read_csv('/Users/bp/Documents/python_for_finance_work/XRP_1M_graph_coinmarketcap.csv', delimiter=';')
raw_sol1m = pd.read_csv('/Users/bp/Documents/python_for_finance_work/SOL_1M_graph_coinmarketcap.csv', delimiter=';')

raw_btc = pd.read_csv('/Users/bp/Documents/python_for_finance_work/BTC_1Y_graph_coinmarketcap.csv', delimiter=';')
raw_eth = pd.read_csv('/Users/bp/Documents/python_for_finance_work/ETH_1Y_graph_coinmarketcap.csv', delimiter=';')
raw_xrp = pd.read_csv('/Users/bp/Documents/python_for_finance_work/XRP_1Y_graph_coinmarketcap.csv', delimiter=';')
raw_sol = pd.read_csv('/Users/bp/Documents/python_for_finance_work/SOL_1Y_graph_coinmarketcap.csv', delimiter=';')

btc = raw_btc[['timestamp', 'close']].rename(columns={'close': 'BTC'})
eth = raw_eth[['timestamp', 'close']].rename(columns={'close': 'ETH'})
xrp = raw_xrp[['timestamp','close']].rename(columns={'close':'XRP'})
sol = raw_sol[['timestamp','close']].rename(columns={'close':'SOL'})

df_merged = btc.merge(eth, on='timestamp').merge(xrp, on='timestamp').merge(sol, on='timestamp')
df_merged['timestamp'] = pd.to_datetime(df_merged['timestamp'],format='ISO8601', utc=True)
df_merged.set_index('timestamp',inplace=True)
#print(raw_btc.columns)
#print(btc)

for col in symbols:
    df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

df_merged.dropna(inplace=True)
#print(df_merged.tail())
#print(df_merged.dtypes)
#(df_merged / df_merged.iloc[0] * 100).plot(figsize=(10,6))

log_returns = np.log(df_merged / df_merged.shift(1))
#print(log_returns.head()) # type: ignore

#log_returns.hist(bins=50, figsize=(10,8)); # type: ignore

'''for sym in symbols:
    print('\nResults for symbol {}'.format(sym))
    print(30 * '-')
    log_data = np.array(log_returns[sym].dropna()) #type: ignore
    print_statistics(log_data)'''

'''for sym in symbols:
    print('\nResults for symbol {}'.format(sym))
    print(32 * '-')
    log_data = np.array(log_returns[sym].dropna()) #type: ignore
    normality_tests(log_data)'''


'''sm.qqplot(log_returns['BTC'].dropna(), line='s') #type: ignore
plt.title('BTC')
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles')
sm.qqplot(log_returns['ETH'].dropna(), line='s') #type: ignore
plt.title('ETH')
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles')'''

#log_returns.hist(bins=40, figsize=(10,8)) #type: ignore

print(log_returns.mean() * 252) #type: ignore
print(log_returns.cov() * 252) #type: ignore 

#plt.show()