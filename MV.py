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
