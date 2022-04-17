import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
from scipy.optimize import minimize

# SRP 최적화
# 0. 초기 값 입력 (무위험수익률, 기간)
# 1. 동일가중 PF 예시
# 2. 효율적 프론티어(리턴, 표준편차) 구성
# 3. 샤프비율 최적 비중 계싼


# 0. 초기값 입력
rf = 0.02
start_date = '2021-04-01'
end_date = '2022-04-15'
ticker = ['AAPL','TSLA','TSM','KO','COST','ADBE','NKE','SBUX','BAC']
num = 30

# 1. 대상종목 가격 가져오기
df = pd.DataFrame()
for i in range(0,len(ticker)) :
    df_temp = pd.DataFrame(pdr.get_data_yahoo(ticker[i],start_date, end_date)['Adj Close'])
    df_temp = df_temp.rename(columns = {'Adj Close':ticker[i]})
    df = pd.concat([df, df_temp], axis = 1)

df_rtn = df.pct_change().dropna()
df_dev = df_rtn.std()

df_rtn_1y = ((1 + df_rtn).prod())**(255/len(df_rtn)) -1
df_dev_1y = df_rtn.std()*((255/len(df_rtn))**0.5)
df_srp_1y = ((df_rtn_1y - rf) / df_dev_1y) * 1/100
df_cov_1y = df_rtn.cov()*(255/len(df_rtn))


def get_mdd(rtn):
    df_cumrtn = (1 + rtn).cumprod()
    df_cumrtn_peak = df_cumrtn.cummax()
    df_mdd = (df_cumrtn - df_cumrtn_peak) / df_cumrtn_peak
    return df_mdd

# 2. 동일가중 포트폴리오 예시
def get_pf_rtn(wgt, rtn) :
    return wgt.T @ rtn
def get_pf_dev(wgt, cov) :
    return np.sqrt(wgt.T @ cov @ wgt)
def get_pf_sr(wgt, rf, rtn, cov) :
    return ((get_pf_rtn(wgt, rtn) - rf) / get_pf_dev(wgt, cov)) * 1/100

wgt= np.repeat(1/len(ticker), len(ticker))
pf_rtn = get_pf_rtn(wgt, df_rtn_1y)
pf_dev = get_pf_dev(wgt, df_cov_1y)
pf_srp = get_pf_sr(wgt, rf, df_rtn_1y, df_cov_1y)


# 3. 효율적 프론티어 구성
def get_min_sr_wgt(rf, rtn, cov):
    bounds = ((0, 1),) * len(rtn)
    wgt_constrnts = {'type': 'eq','fun': lambda wgts: np.sum(wgts) - 1}
    def neg_sharpe(wgts, rf, rtn, cov):
        r = get_pf_rtn(wgts, rtn)
        dev = get_pf_dev(wgts, cov)
        return -(r - rf)/dev
    wgts = minimize(neg_sharpe, np.repeat(1/len(rtn), len(rtn))
                   ,args=(rf, rtn, cov), method='SLSQP'
                   ,options={'disp': False}
                   ,constraints=(wgt_constrnts,)
                   ,bounds=bounds)
    return wgts.x

def optimal_weights(num, rtn, cov):
    def min_vol(tgt_rtn, rtn, cov):
        bounds = ((0, 1),) * len(rtn)
        wgt_constrnts = {'type': 'eq','fun': lambda wgts: np.sum(wgts) - 1}
        rtn_maximize = {'type': 'eq','args': (rtn,),'fun': lambda wgts, rtn: tgt_rtn - get_pf_rtn(wgts,rtn)}
        wgts = minimize(get_pf_dev, np.repeat(1/len(rtn), len(rtn))
                        ,args=(cov,), method='SLSQP'
                        ,options={'disp': False}
                        ,constraints=(wgt_constrnts,rtn_maximize)
                        ,bounds=bounds)
        return wgts.x
    tgt_rtns = np.linspace(rtn.min(), rtn.max(), num)
    wgts = [min_vol(tgt_rtn, rtn, cov) for tgt_rtn in tgt_rtns]
    return wgts

def get_eff_frontier(num, rtn, cov) :
    wgts = optimal_weights(num, rtn, cov)
    rtns = [get_pf_rtn(wgt, rtn) for wgt in wgts]
    devs = [get_pf_dev(wgt, cov) for wgt in wgts]
    df = pd.DataFrame({"Rtns": rtns, "Devs": devs})
    ax = df.plot.line(x="Devs", y="Rtns", style='.-')
    ax.set_xlim(left = 0)
    eff_wgt = get_min_sr_wgt(rf, rtn, cov)
    eff_rtn = get_pf_rtn(eff_wgt, rtn)
    eff_dev = get_pf_dev(eff_wgt, cov)
    line_x = [0, eff_dev]
    line_y = [rf, eff_rtn]
    return ax.plot(line_x, line_y, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=12)

get_eff_frontier(num, df_rtn_1y, df_cov_1y)


# 4. 최적 포트폴리오 비중, 수익률 계산
def get_eff_wgt(rf, rtn, cov):
    eff_wgt = get_min_sr_wgt(rf, rtn, cov)
    eff_rtn = get_pf_rtn(eff_wgt, rtn)
    df = pd.DataFrame([ticker, eff_wgt]).T
    return df, eff_rtn


opt_wgt = get_eff_wgt(rf, df_rtn_1y, df_cov_1y)[0]
opt_rtn = get_eff_wgt(rf, df_rtn_1y, df_cov_1y)[1]
opt_srp = get_pf_sr((opt_wgt[1]).array, rf, df_rtn_1y, df_cov_1y)

print("[equal]","rtn:",pf_rtn,", srp:",pf_srp)
print("[optimized]","rtn:",opt_rtn,", srp:",opt_srp)



