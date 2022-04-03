import pandas as pd
import numpy as np
from numpy.linalg import inv
from pykrx import stock
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings( 'ignore' )
import math

# BLM + adjusted_모멘텀
# 0. 기초 Input 입력하기
# 1. 가격, 시가총액 데이터 가져오기
# 2. Prior_wgt 산출
# 3. Prior_rtn 산출
# 4. View 매트릭스 산출
# 5. Post_rtn, Post_wgt 산출
# 6. 정리


# 0. 기초 Input 입력하기
start_date = '20210228'
end_date = '20220228'
strategy = 'momentum'     # 전략
k=3                       # 모멘텀 개월 수

ticker = ['1150','1151','1152','1153','1154','1155','1156','1157','1158','1159','1160']
name = ['커뮤니케이션서비스','건설','중공업','철강소재','에너지화학','정보기술','금융','생활소비재','경기소비재','산업재','헬스케어']



# 1. 가격, 시가총액 데이터 가져오기
def get_prc(ticker, name, end_date) :
    df = pd.DataFrame()
    twoyear_ago = stock.get_nearest_business_day_in_a_week(datetime.strftime(datetime.strptime(end_date, "%Y%m%d") - relativedelta(years=1), "%Y%m%d"))
    for i in range(0, len(ticker)):
        df_temp = stock.get_index_ohlcv_by_date(twoyear_ago, end_date, ticker[i])['종가']
        df = pd.concat([df, df_temp], axis=1)
    df.columns = name
    return df

def get_str_wgt(strategy, df, k) :
    month_list = df.index.map(lambda x: datetime.strftime(x, '%Y-%m')).unique()
    df_monthly = pd.DataFrame()
    for m in month_list:
        try:
            df_monthly = df_monthly.append(df[df.index.map(lambda x: datetime.strftime(x, '%Y-%m')) == m].iloc[-1])
        except Exception as e:
            print("Error : ", str(e))
        pass
    df_monthly = df_monthly / df_monthly.shift(k)
    std_daily = df.pct_change().rolling(window=k * 21, min_periods=0).std().fillna(0)
    mom = df_monthly[df_monthly.index == datetime.strftime(datetime.strptime(end_date, '%Y%m%d'), '%Y-%m-%d')]
    std = std_daily.iloc[len(std_daily) - 1] * math.sqrt(k * 21)
    adj_mom = mom / std
    adj_mom = adj_mom.reset_index(drop=True).T
    if strategy == 'momentum':
        df_str = adj_mom
    elif strategy == 'lowvol':
        df_str = 1 / std
    return df_str

def get_mktcap(ticker, name, start_date, end_date) :
    df = pd.DataFrame()
    for i in range(0, len(ticker)) :
        df_temp = stock.get_index_ohlcv_by_date(start_date, end_date, ticker[i])['시가총액']
        df = pd.concat([df,df_temp], axis = 1)
    df.columns = name
    return df

df_prc = get_prc(ticker, name, end_date)
df_str = get_str_wgt(strategy, df_prc, k)
df_mktcap = get_mktcap(ticker, name, start_date, end_date)


# 2. 시가총액, 평균 수익률로 Prior_wgt 산출하기
rtn = ((1 + df_prc.pct_change().fillna(0)) ** math.sqrt(252) - 1 ).mean().to_numpy()
mkt = df_mktcap.iloc[len(df_mktcap)-1].reset_index(drop=False)
wgt = df_mktcap.iloc[len(df_mktcap)-1]/df_mktcap.iloc[len(df_mktcap)-1].sum()
pri_wgt = mkt.drop(['index'],axis= 1 )/mkt.drop(['index'],axis= 1 ).sum()


# 2. Cov_mtrx, Lambda 로 Prior_rtn 산출하기
cov = (df_prc.pct_change()).cov()
rf = 0.015
exp_excess_rtn = np.dot(rtn - rf , pri_wgt)
var = np.dot(pri_wgt.T,np.dot(cov,pri_wgt))
lmbda = (exp_excess_rtn / var)
pri_rtn = (lmbda * np.dot(cov, pri_wgt) + rf)


# 3. 단위행렬(P), 전략계수(str_Q) 셋팅하기
P = np.eye(len(ticker))
str = np.array(df_str)
str_Q = df_str


# 4. P와 tau로 Omega mtrx산출, str_Q로 Q array 산출하기
tau = 1/30
omega = np.dot(np.dot(P,cov),P.T) * tau * np.eye(len(ticker))

Q = pri_rtn + 100 * rf * str_Q


# 5. P와 Q와 Omega로 Post_rtn 산출, Lambda와 Post_rtn으로 Post_wgt 산출하기
post_rtn = pri_rtn + np.dot( np.dot( tau * np.dot(cov,P.T) , inv(omega + tau * np.dot(P,np.dot(cov,P.T)))) , (Q - np.dot(P, pri_rtn)))
post_wgt = np.dot( 1/lmbda*inv(cov) , (post_rtn - rf))
adj_post_wgt = np.where(post_wgt < 0, 0, post_wgt)

adj_post_wgt = adj_post_wgt/adj_post_wgt.sum()


# 6. wgt 시각화
df_all = pd.DataFrame([name, str, pri_rtn, post_rtn, wgt, adj_post_wgt],
                     columns = ticker,
                     index = ['ticker','str_wgt','pri_rtn','post_rtn','pri_wgt','post_wgt']).T
for i in range(0,len(df_all)) :
    df_all.iloc[i].str_wgt = float(df_all.iloc[i].str_wgt)
    df_all.iloc[i].post_wgt = float(df_all.iloc[i].post_wgt)
    df_all.iloc[i].pri_wgt = float(df_all.iloc[i].pri_wgt)
    df_all.iloc[i].pri_rtn = float(df_all.iloc[i].pri_rtn)
    df_all.iloc[i].post_rtn = float(df_all.iloc[i].post_rtn)
df_all['wgt_diff'] = df_all['post_wgt'] - df_all['pri_wgt']
df_all = df_all.set_index('ticker')

df_all[['pri_wgt','post_wgt','wgt_diff']].plot(kind='bar', figsize=(12,8))
