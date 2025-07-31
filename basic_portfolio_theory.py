import math
import numpy as np
import scipy.stats as scs
import statsmodels.api as sm
import matplotlib.pylab as plt
from scipy_describe_function import print_statistics
from gen_paths_brownian import gen_paths
from normality_test import normality_tests
import pandas as pd
import scipy.optimize as sco
import scipy.interpolate as sci

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



# log returns
log_returns = np.log(df_merged / df_merged.shift(1))
#print(log_returns.tail()) #type: ignore -> check for log_returns df
# clean Nans
log_returns = log_returns[~np.isnan(log_returns).any(axis=1)]



# portfolio weighting
noa = len(symbols)

rng = np.random.default_rng()
weights = rng.random(noa) # random portfolio weights
weights /= np.sum(weights) # normalized to 1 or 100%
#print(weights.sum()) # check for sum of 1




# Annualized expected portfolio return based on generalized expected portfolio return formula
mp = np.sum(log_returns.mean() * weights) * 252
#print(mp)



# Annualized portfolio variance and covariance to measure volatility

# covariance matrix
cov_matrix = np.cov(log_returns.T)
annual_cov_matrix = cov_matrix * 252

pv = np.dot(weights.T, np.dot(annual_cov_matrix, weights)) #type: ignore
pvol = math.sqrt(np.dot(weights.T, np.dot(annual_cov_matrix, weights))) #type: ignore
#print(pv)
#print(pvol)

def port_ret(w):
    return np.sum(log_returns.mean() * w) * 252

def port_vol(w):
    return np.sqrt(np.dot(w.T, np.dot(annual_cov_matrix, w)))

prets = []
pvols = []
for p in range(2500):
    weights = rng.random(noa)
    weights /= np.sum(weights)
    prets.append(port_ret(weights))
    pvols.append(port_vol(weights))
prets = np.array(prets)
pvols = np.array(pvols)

def min_func_sharpe(weights): # function to be minimized
    return -port_ret(weights) / port_vol(weights) 

cons = ({'type' : 'eq', 'fun': lambda x: port_ret(x) - tret},
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # equality constraints - 1 for efficient frontier

bnds = tuple((0,1) for x in range(noa)) # bounds for the parameters

eweights = np.array(noa * [1. / noa, ])
#print(eweights)
#print(min_func_sharpe(eweights))

opts = sco.minimize(min_func_sharpe, eweights, method='SLSQP', bounds=bnds, constraints=cons)
#print(opts)
#print(opts['x'].round(3)) # optimal portfolio weights
#print(port_ret(opts['x']).round(3)) # resulting portfolio return
#print(port_vol(opts['x']).round(3)) # resulting portfolio volatility
#print(port_ret(opts['x']) / port_vol(opts['x'])) # maximum sharpe ratio

# minimizing the variance of the portfolio
optv = sco.minimize(port_vol, eweights, method='SLSQP', bounds=bnds, constraints=cons)


# efficient frontier
bndsef = tuple((0,1) for x in weights)

trets = np.linspace(0.05, 0.2, 50)
tvols = []
for tret in trets:
    res = sco.minimize(port_vol, eweights, method='SLSQP', bounds=bndsef, constraints=cons) # minimization of portfolio volatility for different target returns
    tvols.append(res['fun'])
tvols = np.array(tvols)

ind = np.argmin(tvols) # index position of minimum volatility portfolio
evols = tvols[ind: ] # relevant portfolio volatility and return values
erets = trets[ind: ]

tck = sci.splrep(evols, erets) # cubic splines interpolation on these values

def f(x):
    """Efficient frontier function (splines approximation)."""
    return sci.splev(x, tck=tck, der=0)

def df(x):
    """First derivative of efficient frontier function."""
    return sci.splev(x, tck=tck, der=1)

'''plt.figure(figsize=(10,6))
plt.scatter(pvols * 100, prets * 100, c=prets / pvols, marker='.', cmap='coolwarm', alpha=0.8)
plt.plot(tvols, trets, 'b', lw=4.0)
plt.plot(port_vol(opts['x']), port_ret(opts['x']), 'y*', markersize=15.0)
plt.plot(port_vol(optv['x']), port_ret(optv['x']), 'r*', markersize=15.0)
plt.xlabel('expected volatility (%)')
plt.ylabel('expected return (%)')
plt.colorbar(label='Sharpe Ratio')
plt.show()'''