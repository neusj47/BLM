import numpy as np
import pandas as pd
from numpy.linalg import inv
import pandas as pd
import numpy as np
from numpy.linalg import inv
from pykrx import stock
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings( 'ignore' )
import math

asset_returns_orig = pd.read_csv('C:/Users/ysj/Desktop/asset_returns.csv', index_col='Year', parse_dates=True)
asset_weights = pd.read_csv('C:/Users/ysj/Desktop/asset_weights.csv', index_col='asset_class')

cols = ['Global Bonds (Unhedged)','Total US Bond Market','US Large Cap Growth',
            'US Large Cap Value','US Small Cap Growth','US Small Cap Value','Emerging Markets',
            'Intl Developed ex-US Market','Short Term Treasury']

asset_returns = asset_returns_orig[cols].dropna()
treasury_rate = asset_returns['Short Term Treasury']
asset_returns = asset_returns[cols[:-1]].astype(np.float).dropna()
asset_weights = asset_weights.loc[cols[:-1]]

asset_returns.mean()



excess_asset_returns = asset_returns.subtract(treasury_rate, axis=0)
cov = excess_asset_returns.cov()

global_return = excess_asset_returns.mean().multiply(asset_weights['weight'].values).sum()


market_var = np.matmul(asset_weights.values.reshape(len(asset_weights)).T,             np.matmul(cov.values, asset_weights.values.reshape(len(asset_weights))))


risk_aversion = global_return / market_var


def implied_rets(risk_aversion, sigma, w):
    implied_rets = risk_aversion * sigma.dot(w).squeeze()

    return implied_rets


implied_equilibrium_returns = implied_rets(risk_aversion, cov, asset_weights)
implied_equilibrium_returns


P = [[0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, .5, -.5, .5, -.5, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]]

P = np.asarray([[0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, .85, -.85, .15, -.15, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1]])


view1_var = np.matmul(P[0].reshape(len(P[0])),np.matmul(cov.values, P[0].reshape(len(P[0])).T))
view2_var = np.matmul(P[1].reshape(len(P[1])),np.matmul(cov.values, P[1].reshape(len(P[1])).T))
view3_var = np.matmul(P[2].reshape(len(P[2])),np.matmul(cov.values, P[2].reshape(len(P[2])).T))
print(f'The Variance of View 1 Portfolio is {view1_var}, and the standard deviation is {np.sqrt(view1_var):.3f}\n',\
      f'The Variance of View 2 Portfolio is {view2_var}, and the standard deviation is {np.sqrt(view2_var):.3f}\n',\
      f'The Variance of View 3 Portfolio is {view3_var}, and the standard deviation is {np.sqrt(view3_var):.3f}')

def error_cov_matrix(sigma, tau, P):
    matrix = np.diag(np.diag(P.dot(tau * cov).dot(P.T)))
    return matrix
tau = 0.025
omega = error_cov_matrix(cov, tau, P)


sigma_scaled = cov * tau
BL_return_vector = implied_equilibrium_returns + sigma_scaled.dot(P.T).dot(inv(P.dot(sigma_scaled).dot(P.T) + omega).dot(Q - P.dot(implied_equilibrium_returns)))


returns_table = pd.concat([implied_equilibrium_returns, BL_return_vector], axis=1) * 100
returns_table.columns = ['Implied Returns', 'BL Return Vector']
returns_table['Difference'] = returns_table['BL Return Vector'] - returns_table['Implied Returns']
returns_table.style.format('{:,.2f}%')