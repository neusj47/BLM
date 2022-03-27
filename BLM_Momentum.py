import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from statsmodels.stats.correlation_tools import cov_nearest
from numpy.linalg import inv
from pykrx import stock

# Black_Litterman 최적비중 산출

# 0. 대상 종목 정보 가져오기
# 1. 시가총액, 평균 수익률로 Prior_wgt 산출하기
# 2. Cov_mtrx, Lambda 로 Prior_rtn 산출하기
# 3. 단위행렬(P), 전략계수(str_Q) 셋팅하기
# 4. P와 tau로 Omega mtrx산출, str_Q로 Q array 산출하기
# 5. P와 Q와 Omega로 Post_rtn 산출, Lambda와 Post_rtn으로 Post_wgt 산출하기


# 0. 대상종목 정보 가져오기

# 1150 코스피 200 커뮤니케이션서비스
# 1151 코스피 200 건설
# 1152 코스피 200 중공업
# 1153 코스피 200 철강/소재
# 1154 코스피 200 에너지/화학
# 1155 코스피 200 정보기술
# 1156 코스피 200 금융
# 1157 코스피 200 생활소비재
# 1158 코스피 200 경기소비재
# 1159 코스피 200 산업재
# 1160 코스피 200 헬스케어

start_date = '20220228'
end_date = '20220326'
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

def get_mktcap(ticker, name, start_date, end_date) :
    df = pd.DataFrame()
    for i in range(0, len(ticker)) :
        df_temp = stock.get_index_ohlcv_by_date(start_date, end_date, ticker[i])['시가총액']
        df = pd.concat([df,df_temp], axis = 1)
    df.columns = name
    return df

df_prc = get_prc_mom(ticker, name, start_date, end_date)[0]
df_mom = get_prc_mom(ticker, name, start_date, end_date)[1]
df_mktcap = get_mktcap(ticker, name, start_date, end_date)

# 1. 시가총액, 평균 수익률로 Prior_wgt 산출하기

rtn = df_prc.pct_change().mean().to_numpy()
wgt = df_mktcap.iloc[len(df_mktcap)-1].reset_index(drop=False)
wgt = wgt.drop(['index'],axis= 1 )
wgt = wgt/wgt.sum()


# 2. Cov_mtrx, Lambda 로 Prior_rtn 산출하기
cov = df_prc.pct_change().cov()
rf = 0.005
# exp_excess_rtn = np.dot(rtn - rf , wgt)
# var = np.dot(wgt.T,np.dot(cov,wgt))
# lmbda = exp_excess_rtn / var
lmbda = 2.6

pri_rtn = (lmbda * np.dot(cov, wgt) + rf)


# 3. 단위행렬(P), 전략계수(str_Q) 셋팅하기
P = np.eye(len(ticker))
df_mom = df_mom.reset_index(drop=False)
df_mom = df_mom.drop(['index'],axis= 1 )

str_Q = df_mom

# 4. P와 tau로 Omega mtrx산출, str_Q로 Q array 산출하기
tau = 1/len(df_prc)
omega = np.dot(np.dot(P,cov),P.T) * tau * np.eye(len(ticker))

Q = pri_rtn + 100 * rf * str_Q


# 5. P와 Q와 Omega로 Post_rtn 산출, Lambda와 Post_rtn으로 Post_wgt 산출하기

post_ret = pri_rtn + np.dot( np.dot( tau * np.dot(cov,P.T) , inv(omega + tau * np.dot(P,np.dot(cov,P.T)))) , (Q - np.dot(P, pri_rtn)))
post_wgt = np.dot( 1/lmbda*inv(cov) , (post_ret - rf))
adj_post_wgt = post_wgt
# adj_post_wgt = np.where(post_wgt < 0, 0, post_wgt)
# adj_post_wgt = adj_post_wgt/adj_post_wgt.sum()


fin_wgt = pd.DataFrame([name, adj_post_wgt],
                     columns = ticker,
                     index = ['ticker','BL_wgt']).T

mkt_wgt = pd.DataFrame([name, wgt],
                     columns = ticker,
                     index = ['ticker','wgt'])

# fin_mom.plot(kind='bar', figsize=(12,8))
# plt.legend(fontsize=15)
