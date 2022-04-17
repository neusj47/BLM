import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
from scipy.optimize import minimize
from datetime import datetime
from pykrx import stock
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings( 'ignore' )
from optimize_SR import *


# SR 최적화 백테스팅
# 0. 기초 데이터 입력
# 1. 월간 수익률 계산
# 2. 월간 비중 산출
# 3. 액티브 성과 비교


# 0. 기초데이터 입력
start_date = '20190330'
end_date = '20220330'
ticker = ['1150','1151','1152','1153','1154','1155','1156','1157','1158','1159','1160']
name = ['커뮤니케이션서비스','건설','중공업','철강소재','에너지화학','정보기술','금융','생활소비재','경기소비재','산업재','헬스케어']
rf = 0.02


# 1. 월간 수익률 계산
def get_prc(ticker, name, start_date, end_date) :
    df = pd.DataFrame()
    # oneyear_ago = stock.get_nearest_business_day_in_a_week(datetime.strftime(datetime.strptime(end_date, "%Y%m%d") - relativedelta(years=1), "%Y%m%d"))
    for i in range(0, len(ticker)):
        df_temp = stock.get_index_ohlcv_by_date(start_date, end_date, ticker[i])['종가']
        df = pd.concat([df, df_temp], axis=1)
    df.columns = name
    return df

def get_monthly_rtn(ticker, name, start_date, end_date) :
    df = get_prc(ticker, name, start_date, end_date)
    month_list = df.index.map(lambda x: datetime.strftime(x, '%Y-%m')).unique()
    df_monthly = pd.DataFrame()
    for m in month_list:
        try:
            df_monthly = df_monthly.append(df[df.index.map(lambda x: datetime.strftime(x, '%Y-%m')) == m].iloc[-1])
        except Exception as e:
            print("Error : ", str(e))
        pass
    df_monthly_rtn = (df_monthly / df_monthly.shift(1) - 1).fillna(0)
    return df_monthly_rtn

df_monthly_rtn= get_monthly_rtn(ticker, name, start_date, end_date)



# 2. 월간 비중 산출

def get_bdate_info(start_date, end_date) :
    end_date = stock.get_nearest_business_day_in_a_week(datetime.strftime(datetime.strptime(end_date, "%Y%m%d") - relativedelta(days=1),"%Y%m%d"))
    date = pd.DataFrame(stock.get_previous_business_days(fromdate=start_date, todate=end_date)).rename(columns={0: '일자'})
    prevbdate = date.shift(1).rename(columns={'일자': '전영업일자'})
    date = pd.concat([date, prevbdate], axis=1).fillna(
        datetime.strftime(datetime.strptime(stock.get_nearest_business_day_in_a_week(datetime.strftime(datetime.strptime(start_date, "%Y%m%d") - relativedelta(days=1), "%Y%m%d")), "%Y%m%d"),"%Y-%m-%d %H:%M:%S"))
    date['주말'] = ''
    for i in range(0, len(date) - 1):
        if abs(datetime.strptime(datetime.strftime(date.iloc[i + 1].일자, "%Y%m%d"), "%Y%m%d") - datetime.strptime(datetime.strftime(date.iloc[i].일자, "%Y%m%d"), "%Y%m%d")).days > 1:
            date['주말'].iloc[i] = 1
        else:
            date['주말'].iloc[i] = 0
    month_list = date.일자.map(lambda x: datetime.strftime(x, '%Y-%m')).unique()
    monthly = pd.DataFrame()
    for m in month_list:
        try:
            monthly = monthly.append(date[date.일자.map(lambda x: datetime.strftime(x, '%Y-%m')) == m].iloc[-1])
        except Exception as e:
            print("Error : ", str(e))
        pass
    date['월말'] = np.where(date['일자'].isin(monthly.일자.tolist()), 1, 0)
    return date

def get_monthly_wgt(ticker, name, start_date, end_date) :
    date = get_bdate_info(start_date, end_date)
    monthly = date[date.월말 == 1].reset_index(drop=True)
    df_wgt_all = pd.DataFrame()
    for i in range(0, len(monthly)) :
        std_date = datetime.strftime(monthly.일자[i], "%Y%m%d")
        oneyear_ago = stock.get_nearest_business_day_in_a_week(datetime.strftime(datetime.strptime(std_date, "%Y%m%d") - relativedelta(years=1), "%Y%m%d"))
        df_prc = get_prc(ticker, name, oneyear_ago, std_date)
        df_rtn = df_prc.pct_change().fillna(0)
        df_rtn_1y = ((1 + df_rtn).prod())**(255/len(df_rtn)) -1
        df_cov_1y = df_rtn.cov() * (255)
        df_wgt = get_eff_wgt(rf, df_rtn_1y, df_cov_1y)[0]
        df_wgt = df_wgt.T
        df_wgt = df_wgt.drop(df_wgt.index[0])
        df_wgt.columns = name
        df_wgt['일자'] = datetime.strftime(datetime.strptime(std_date, "%Y%m%d"),"%Y-%m-%d")
        df_wgt = df_wgt.set_index('일자')
        df_wgt_all = pd.concat([df_wgt, df_wgt_all])
    df_wgt_all = df_wgt_all.sort_index()
    return df_wgt_all



df_wgt = get_monthly_wgt(ticker, name, start_date, end_date)




df_adjusted_wgt



df_monthly_rtn.to_excel('C:/Users/ysj/Desktop/rtn2.xlsx')
df_wgt.to_excel('C:/Users/ysj/Desktop/wgt2.xlsx')