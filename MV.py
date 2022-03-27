import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from statsmodels.stats.correlation_tools import cov_nearest
from pykrx import stock


start_date = '20220201'
end_date = '20220228'
ticker = ['1150','1151','1152','1153','1154','1155','1156','1157','1158','1159','1160']
name = ['커뮤니케이션서비스','건설','중공업','철강소재','에너지화학','정보기술','금융','생활소비재','경기소비재','산업재','헬스케어']

def get_prc_mom(ticker, name, start_date, end_date) :
    df = pd.DataFrame()
    for i in range(0, len(ticker)) :
        df_temp = stock.get_index_ohlcv_by_date(start_date, end_date, ticker[i])['종가']
        df = pd.concat([df,df_temp], axis = 1)
    df.columns = name
    df_mom = df.iloc[len(df) - 1] / df.iloc[0] - 1
    return df, df_mom


df_prc = get_prc_mom(ticker, name, start_date, end_date)[0]
cov = df_prc.pct_change().cov()
df_rtn = df_prc.pct_change()

#Empty lists to store weights, returns, and volatilities of portfolios made:
weights = []
returns = []
vols = []

tesla_w = 0   #TSLA initial weight

#Loop to calculate each portfolio with 100 different weights for TSLA:
for i in range(101):
  #Weights calculation:
  coke_w = 1 - tesla_w              #Find coke's weight given TSLA's
  w = np.array([tesla_w, coke_w])   #groups the two weights as one
  weights.append(w)                 #Adds the two into list
  tesla_w += 0.01                   #Increase tesla weight each iteration by 0.01

  #Returns:
  ret = np.dot(w, df_rtn)       #Weighted sum between returns and weights
  returns.append(ret)               #Add it to the list

  #Volatility:
  var = np.dot(w.T, np.dot(cov, w))        #w.T*Cov*w as shown previously
  yearly_vol = np.sqrt(var)*np.sqrt(252)   #Standard deviation is volatility
  vols.append(yearly_vol)

#Putting made lists into a dictionary:
port = {'Returns': returns, 'Volatility':vols}

#Placing weights into dictionary made with a list comprehension:
for counter, symbol in enumerate(df_prc.columns.tolist()):
  port[symbol + '_Weight'] = [weight[counter] for weight in weights]

#Make a dataframe from the dictionary buil:
port_df = pd.DataFrame(port)
port_df.head()