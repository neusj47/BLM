from BL_strategy import *
from datetime import datetime
from dateutil.relativedelta import relativedelta


start_date = '20181102'
end_date = '20220325'

def get_monthly_rtn(df_prc) :
    month_list = df_prc.index.map(lambda x : datetime.strftime(x, '%Y-%m')).unique()
    monthly_prc = pd.DataFrame()
    for m in month_list:
        try:
            monthly_prc = monthly_prc.append(df_prc[df_prc.index.map(lambda x: datetime.strftime(x, '%Y-%m')) == m].iloc[-1])
        except Exception as e:
            print("Error : ", str(e))
        pass
    monthly_rtn = (monthly_prc/monthly_prc.shift(1) -1).fillna(0)
    return monthly_rtn

df_prc = get_prc(ticker, name, end_date)
monthly_rtn = get_monthly_rtn(df_prc)

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

bdate = get_bdate_info(start_date, end_date)
monthly = bdate[bdate.월말 == 1][['일자','전영업일자']].reset_index(drop=True)

pf_wgt = pd.DataFrame()
for i in range(0,len(monthly)-1) :
    start_date = datetime.strftime(monthly.iloc[i].일자, '%Y%m%d')
    end_date = datetime.strftime(monthly.iloc[i + 1].일자, '%Y%m%d')

    df_prc = get_prc(ticker, name, end_date)
    df_str = get_str_wgt('momentum', df_prc, k)
    df_mktcap = get_mktcap(ticker, name, start_date, end_date)

    # 1. 시가총액, 평균 수익률로 Prior_wgt 산출하기
    rtn = ((1 + df_prc.pct_change().fillna(0)) ** math.sqrt(252) - 1).mean().to_numpy()
    mkt = df_mktcap.iloc[len(df_mktcap) - 1].reset_index(drop=False)
    wgt = df_mktcap.iloc[len(df_mktcap) - 1] / df_mktcap.iloc[len(df_mktcap) - 1].sum()
    pri_wgt = mkt.drop(['index'], axis=1) / mkt.drop(['index'], axis=1).sum()

    # 2. Cov_mtrx, Lambda 로 Prior_rtn 산출하기
    cov = df_prc.pct_change().cov()
    rf = 0.015
    exp_excess_rtn = np.dot(rtn - rf, pri_wgt)
    var = np.dot(pri_wgt.T, np.dot(cov, pri_wgt))
    lmbda = (exp_excess_rtn / var)
    pri_rtn = (lmbda * np.dot(cov, pri_wgt) + rf)

    # 3. 단위행렬(P), 전략계수(str_Q) 셋팅하기
    P = np.eye(len(ticker))
    str = np.array(df_str)
    str_Q = df_str

    # 4. P와 tau로 Omega mtrx산출, str_Q로 Q array 산출하기
    tau = 1 / 30
    omega = np.dot(np.dot(P, cov), P.T) * tau * np.eye(len(ticker))
    Q = pri_rtn + 100 * rf * str_Q

    # 5. P와 Q와 Omega로 Post_rtn 산출, Lambda와 Post_rtn으로 Post_wgt 산출하기
    post_rtn = pri_rtn + np.dot(np.dot(tau * np.dot(cov, P.T), inv(omega + tau * np.dot(P, np.dot(cov, P.T)))),
                                (Q - np.dot(P, pri_rtn)))
    post_wgt = np.dot(1 / lmbda * inv(cov), (post_rtn - rf))
    adj_post_wgt = np.where(post_wgt < 0, 0, post_wgt)
    adj_post_wgt = adj_post_wgt / adj_post_wgt.sum()

    # 6. wgt 시각화
    df_all = pd.DataFrame([name, str, pri_rtn, post_rtn, wgt, adj_post_wgt],
                          columns=ticker,
                          index=['ticker', 'str_wgt', 'pri_rtn', 'post_rtn', 'pri_wgt', 'post_wgt']).T
    for i in range(0, len(df_all)):
        df_all.iloc[i].str_wgt = float(df_all.iloc[i].str_wgt)
        df_all.iloc[i].post_wgt = float(df_all.iloc[i].post_wgt)
        df_all.iloc[i].pri_wgt = float(df_all.iloc[i].pri_wgt)
        df_all.iloc[i].pri_rtn = float(df_all.iloc[i].pri_rtn)
        df_all.iloc[i].post_rtn = float(df_all.iloc[i].post_rtn)
    df_all['wgt_diff'] = df_all['post_wgt'] - df_all['pri_wgt']
    df_all = df_all.set_index('ticker')

    pf_wgt_temp = df_all[['post_wgt']].T
    pf_wgt_temp['날짜'] = datetime.strftime(monthly.iloc[i + 1].일자, '%Y-%m-%d')
    pf_wgt = pd.concat([pf_wgt, pf_wgt_temp], axis=0)

# monthly_rtn.to_excel('C:/Users/Check/Desktop/rtn.xlsx')
pf_wgt.to_excel('C:/Users/Check/Desktop/pf_wgts.xlsx')
