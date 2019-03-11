import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import scipy.cluster.hierarchy as sch
import scipy.optimize
import warnings
import hrp_helper
import MC
import afp_plot
import os
from scipy.cluster.hierarchy import dendrogram

warnings.filterwarnings("ignore")


def F(w, cov_matrix, N):
    """equal risk parity formula"""
    diff = 0
    for i in range(N):
        diff += np.power(w[i] - (np.dot(w, np.dot(cov_matrix,w))/
                          (N * np.dot(cov_matrix, w)[i])),2)
    return diff


def calc_eq_risk_parity_weights(x, cov_forecast_df):
    """calculate weight for each asset,
    assigning equal risk to each asset,
    only constraint is weight sum to 1,
    each asset's weight constrained from -1 to 1"""
    date = x.index.values[0][0]
    cov_matrix = cov_forecast_df.loc[date]
    N = cov_matrix.shape[0]
    para_init = np.ones(N)/N
    bounds = ((0, 1.0),) * N  # long only
    w = scipy.optimize.minimize(F, x0=para_init,
                                args=(cov_matrix.values, N),
                                method='SLSQP',
                                constraints=({'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs)}),
                                bounds=bounds)
    return w.x


def calc_hrp_corr_weights(x, corr_forecast_df, cov_forecast_df, return_mean, obj_df=None, measure='corr'):
    date = x.index.values[0][0]
    corr_matrix = corr_forecast_df.loc[date]
    cov_matrix = cov_forecast_df.loc[date]
    mean_matrix = return_mean.loc[date]
    if measure == 'abs_dist':
        if date not in obj_df.index:
            return pd.Series()
        obj = obj_df.loc[date]
        obj = obj.apply(lambda x: x if x>0 else 0)  # force negative side to cluster
        dist = np.zeros((obj.shape[0], obj.shape[0]))
        for h in range(obj.shape[0]):
            for k in range(obj.shape[0]):
                dist[h, k] = np.abs(obj.iloc[h] - obj.iloc[k])
        dist = pd.DataFrame(dist, columns=obj.index, index=obj.index)

    elif measure == 'corr':
        dist = (abs(1 - corr_matrix) / 2.) ** .5
    else:
        raise ValueError("Measure is not valid")
    link = sch.linkage(dist, 'single')
    # fig = plt.figure(figsize=(8, 5))
    # dn = dendrogram(link, labels=return_mean.columns)
    # plt.show()
    # sortIx = hrp_helper.getQuasiDiag(link)
    # sortIx = corr_matrix.index[sortIx].tolist()
    #
    # hrp = hrp_helper.getRecBipart(cov_matrix, sortIx)
    sort_ls = hrp_helper.getClusterLink(link)
    hrp = hrp_helper.getRecPart(cov_matrix, sort_ls)
    hrp[np.abs(hrp) > 1] = 1
    hrp = hrp / hrp.sum()
    return hrp

#def getHRP(cov,corr):
#    # Construct a hierarchical portfolio
#    corr, cov = pd.DataFrame(corr), pd.DataFrame(cov)
#    dist = correlDist(corr)
#    link = sch.linkage(dist,'single')
#    sortIx = getQuasiDiag(link)
#    sortIx = corr.index[sortIx].tolist() # recover labels
#    hrp = getRecBipart(cov,sortIx)
#    return hrp.sort_index()


def calc_weights(method='risk_parity',
                 vol_forecast_df=None,
                 corr_forecast_df=None,
                 cov_forecast_df=None,
                 returns_df=None,
                 lookback_period=60,
                 pre_calc=False,
                 obj_df=None):
    return_mean = returns_df.rolling(lookback_period).mean()
    if method == 'risk_parity':
        inv_vol = 1/vol_forecast_df
        inv_vol_sum = inv_vol.sum(axis=1)
        weights = inv_vol.apply(lambda x: x/inv_vol_sum, axis=0)
        return weights

    elif method == 'equal_risk_parity':
        w = cov_forecast_df.groupby(level=0).apply(
            lambda x: calc_eq_risk_parity_weights(x, cov_forecast_df))
        w = w.apply(pd.Series)
        w.columns = cov_forecast_df.columns
        return w
    elif method == 'hrp':
        w = corr_forecast_df.groupby(level=0).apply(
            lambda x: calc_hrp_corr_weights(x, corr_forecast_df, cov_forecast_df, return_mean))#.unstack(level=-1)
        return w
    elif method == 'hrp_dd':
        # can be changed to expanding if want to take into account all data
        drawdown_df = returns_df.expanding(lookback_period).apply(lambda x: -hrp_helper.mdd(x), raw=False)
        #standardize dd
        drawdown_df = drawdown_df.sub(drawdown_df.mean(axis = 1), axis = 0)
        drawdown_df[drawdown_df<0] = 0
        drawdown_df[drawdown_df>0] = np.exp(drawdown_df[drawdown_df>0])
        w = corr_forecast_df.groupby(level=0).apply(
            lambda x: calc_hrp_corr_weights(x, corr_forecast_df, cov_forecast_df, return_mean, obj_df=drawdown_df,
                                            measure='abs_dist'))#.unstack(-1)
        return w
    elif pre_calc:
        w = corr_forecast_df.groupby(level=0).apply(
            lambda x: calc_hrp_corr_weights(x, corr_forecast_df, cov_forecast_df, return_mean, obj_df=obj_df,
                                            measure='abs_dist'))#.unstack(level=-1)
        return w

def get_data(file_location):
    # get the data, clean it, merge it
    df = pd.read_csv(file_location, index_col=0)
    df.index = pd.DatetimeIndex(df.index)
    # df = df.sort_index(ascending=True)
    return df


def save_fig(df, file_location):
    df.plot(figsize=[12, 8])
    plt.savefig(file_location)
    return None


def calc_rebal(x, portfolio_df, returns_df, weights_df, txn_cost):
    prev_period = x.index[0]
    curr_start = np.datetime64(portfolio_df.index[portfolio_df.index.get_loc(x.index[0]) + 1])
    curr_end = (x.index[1])

    # calculate percentage increase over the period
    period_return = np.exp(returns_df.loc[curr_start:curr_end, :].cumsum())

    # apply to portfolio
    prev_port_values = portfolio_df.loc[prev_period, :]
    period_values = prev_port_values*period_return
    portfolio_df.loc[curr_start:curr_end, :] = period_values

    # find the size of portfolio
    total_size = period_values.sum(axis=1).iloc[-1]

    # rebalance portfolio
    new_weights = weights_df.loc[curr_end, :]
    portfolio_df.loc[curr_end, :] = new_weights*total_size

    # calc txn costs
    costs = np.abs(portfolio_df.loc[curr_end, :] - period_values.iloc[-1, :])*txn_cost
    portfolio_df.loc[curr_end, :] = portfolio_df.loc[curr_end, :] - costs

    return 0


def adj_sr(ret):
    sr = ret.mean() / ret.std() * np.sqrt(12)
    skew = ret.skew()
    kurt = ret.kurtosis()
    return sr * (1 + skew / 6 * sr - (kurt - 3) / 24 * sr ** 2)


def cert_eqv_ret(ret, gamma=3, rf=0.02):
    mu = ret.mean() * 12
    sigma = ret.std() * np.sqrt(12)
    return (mu - rf) - gamma * 0.5 * sigma ** 2


def max_drawdown(ret):
    ret_sum = ret.cumsum().apply(lambda x: np.exp(x))
    dd = ret_sum / ret_sum.cummax() - 1
    mdd = dd.min()
    end = dd.idxmin()
    start = ret.loc[:end].idxmax()
    return mdd, start, end


def turnover(weights):
    s = 0
    for i in range(weights.shape[1] - 1):
        s += np.abs(weights.iloc[:, i] - weights.iloc[:, i + 1]).sum()
    return s / weights.shape[1]


def ss_ports_wt(weights):
    return np.sum(np.sum(weights ** 2)) / weights.shape[1]


def calc_metrics(title, car, weights):
    returns = car.pct_change().apply(lambda x: np.log(1+x))
    results = {
        'mean': returns.mean() * 12,
        'std': returns.std() * np.sqrt(12),
        'skew': returns.skew(),
        'kurtosis': returns.kurtosis(),
        'ir': (returns.mean()/returns.std()),
        'adj sr': adj_sr(returns),
        'cer': cert_eqv_ret(returns),
        'mdd': max_drawdown(returns)[0],
        'turnover': turnover(weights),
        'sspw': ss_ports_wt(weights)
    }
    return pd.DataFrame(results, index=[title])


def calc_results_matrix(returns_df,
                        weights_df,
                        txn_cost=0.001,
                        rebal_period='Y'):
    # setup the portfolio as the weights
    portfolio_df = weights_df.resample('M').fillna('pad')
    weights_df = weights_df.resample(rebal_period).last()
    weights_df.rolling(2).apply(lambda x: calc_rebal(x, portfolio_df, returns_df, weights_df, txn_cost), raw=False)
    return portfolio_df.dropna()


def calc_vol_forecast(returns_df, method='r_vol', lookback_period=60):
    if method == 'r_vol':
        r_vol = returns_df.rolling(lookback_period).std() * (12 ** 0.5)
        r_var = r_vol*r_vol
        return r_vol, r_var
    elif method == 'garch':
        r_vol = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
        r_var = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
        for index in returns_df.columns:
            returns_index = returns_df[index]
            am = arch_model(returns_index, vol='Garch', p=1, o=0, q=1, dist='Normal')
            res = am.fit(last_obs=returns_index.index[12])
            tmp = res.forecast(horizon=20)
            var_proj = tmp.variance.mean(axis=1) * 12
            std_proj = np.power(var_proj, 0.5)
            r_vol[index] = std_proj
            r_var[index] = var_proj
        return r_vol, r_var
    return None, None


def calc_cor_forecast(returns_df, method='r_cor', lookback_period=60):
    if method == 'r_cor':

        r_corr = returns_df.rolling(lookback_period).corr()
        r_cov = returns_df.rolling(lookback_period).cov() * (12 ** 0.5)
        return r_corr, r_cov

    return None, None


def calc_final_results(total_return,
                       results_metrics,
                       returns_df=None,
                       weights_df=pd.DataFrame(),
                       method='',
                       rebal_period='Y'):
    # parse method types
    method_name = method
    method = method.split(' ')[0]
    # forecast volatility and variance
    vol_forecast_df, var_forecast_df = calc_vol_forecast(returns_df, method='r_vol', lookback_period=60)
    cor_forecast_df, cov_forecast_df = calc_cor_forecast(returns_df, method='r_cor', lookback_period=60)
    if len(weights_df) == 0:
        if method in ['hrp_beta', 'hrp_val', 'hrp_strc']:
            obj_df = get_data("int_results/%s.csv" % method).loc[:, returns_df.columns].dropna()
            weights_df = calc_weights(method=method,
                                      vol_forecast_df=vol_forecast_df.dropna(),
                                      corr_forecast_df=cor_forecast_df.dropna(),
                                      cov_forecast_df=cov_forecast_df.dropna(),
                                      returns_df=returns_df,
                                      lookback_period=60,
                                      pre_calc=True,
                                      obj_df=obj_df)
        else:
            weights_df = calc_weights(method=method,
                                      vol_forecast_df=vol_forecast_df.dropna(),
                                      corr_forecast_df=cor_forecast_df.dropna(),
                                      cov_forecast_df=cov_forecast_df.dropna(),
                                      returns_df=returns_df,
                                      lookback_period=60)
    # save figures
    weights_df.columns = returns_df.columns
    save_fig(weights_df.resample(rebal_period).last(), os.getcwd() + r'/pic/weights_new/%s.pdf' % method_name)
    rebal_df = calc_results_matrix(returns_df=returns_df, weights_df=weights_df, rebal_period=rebal_period)
    total_return = pd.concat([total_return, rebal_df.sum(axis=1).rename(method_name)], axis=1)
    results_metrics = pd.concat(
        [results_metrics, calc_metrics(method_name, total_return[method_name].dropna(), weights_df)], axis=0)
    return total_return, results_metrics


#def calc_final_results_MC(total_return, results_metrics, ret, method):
#    """calculate weight for Monte-Carlo simulation"""
#    ret_df = pd.DataFrame(ret)
#    vol_forecast, var_forecast = calc_vol_forecast(ret_df, method='r_vol')
#    cor_forecast, cov_forecast = calc_cor_forecast(ret_df, method='r_cor')
#    total_return, results_metrics = calc_final_results(total_return,
#                                                       results_metrics,
#                                                       returns_df=ret_df,
#                                                       method=method)
#    return total_return, results_metrics


def main():
    # get Tier data
    ret_df = get_data("data/combined_dataset_new.csv").applymap(lambda x: np.nan if x<1 else x).pct_change()
    ret_df = ret_df.replace(0.0, np.nan).apply(lambda x: np.log(1+x))    # to prevent volatility to explode
    # ret_df = ret_df.drop(['Oil', 'TRCommodity'], axis=1)
    tier1 = ret_df.loc[:, ['USEq', 'USBond10Y']].dropna()
    tier2 = ret_df.loc[:, 'GermanBond10Y':'USEq'].dropna()
    tier3 = ret_df.dropna()#.drop(['Oil', 'TRCommodity'], axis=1)
    total_return = pd.DataFrame()
    results_metrics = pd.DataFrame()
    print("=========== Portfolio Construction Started ===========")

#    ################# Benchmark ##################
    # benchmark 1 - 60/40
    weights_dict = {'USBond10Y': 0.4, 'USEq': 0.6}
    weights_b1 = tier1.copy().apply(lambda x: pd.Series(tier1.columns.map(weights_dict).values), axis=1)
    total_return, results_metrics = calc_final_results(total_return, results_metrics, returns_df=tier1,
                                                       weights_df=weights_b1, method='6040')
    print("=========== 60/40 Portfolio Completed ===========")

    # benchmark 2 - All Weather
    aw_df = tier2.loc[:, ['Gold', 'TRCommodity', 'USBond10Y', 'USEq']].dropna()
    weights_dict = {'Gold':0.075, 'TRCommodity':0.075, 'USBond10Y':0.55, 'USEq':0.3}
    weights_aw = aw_df.copy().apply(lambda x: pd.Series(aw_df.columns.map(weights_dict).values), axis=1)
    total_return, results_metrics = calc_final_results(total_return, results_metrics, returns_df=aw_df,
                                                       weights_df=weights_aw, method='all-weather (original)')
    print("=========== All Weather Portfolio Completed ===========")

    # benchmark 2_prime - All Weather with factors
    aw_df = tier3.loc[:, ['Gold', 'TRCommodity', 'USBond10Y', 'USEq'
                          , 'BAB', 'CS', 'UMD_Large', 'UMD_Small']].dropna()
    weights_dict = {'Gold':0.075,
                    'TRCommodity':0.075,
                    'USBond10Y':0.45,
                    'USEq':0.2,
                    'BAB':0.05,
                    'CS':0.05,
                    'UMD_Large':0.05,
                    'UMD_Small':0.05}
    weights_aw = aw_df.copy().apply(lambda x: pd.Series(aw_df.columns.map(weights_dict).values), axis=1)
    total_return, results_metrics = calc_final_results(total_return, results_metrics, returns_df=aw_df,
                                                       weights_df=weights_aw, method='all-weather (star)')
    print("=========== All Weather star Portfolio Completed ===========")

    # benchmark 3 - Risk Parity Tier 2
    total_return, results_metrics = calc_final_results(total_return, results_metrics, returns_df=tier2,
                                                       method='risk_parity (tier 2)')
    print("=========== Risk Parity Tier 2 Portfolio Completed ===========")

    # benchmark 4 - Risk Parity Tier 3
    total_return, results_metrics = calc_final_results(total_return, results_metrics, returns_df=tier3,
                                                       method='risk_parity (tier 3)')
    print("=========== Risk Parity Tier 3 Portfolio Completed ===========")

    ################## HRP ##################
    # HRP - Covariance
    total_return, results_metrics = calc_final_results(total_return, results_metrics, returns_df=tier2,
                                                       method='hrp (tier 2)')
    total_return, results_metrics = calc_final_results(total_return, results_metrics, returns_df=tier3,
                                                       method='hrp (tier 3)')
    print("=========== HRP Covariance Portfolio Completed ===========")

    # HRP - Maximum Drawdown
    total_return, results_metrics = calc_final_results(total_return, results_metrics, returns_df=tier2,
                                                       method='hrp_dd (tier 2)')
    total_return, results_metrics = calc_final_results(total_return, results_metrics, returns_df=tier3,
                                                       method='hrp_dd (tier 3)')
    print("=========== HRP Maximum Drawdown Portfolio Completed ===========")

    # HRP - Downside Beta
    total_return, results_metrics = calc_final_results(total_return, results_metrics, returns_df=tier2,
                                                       method='hrp_beta (tier 2)')
    total_return, results_metrics = calc_final_results(total_return, results_metrics, returns_df=tier3,
                                                       method='hrp_beta (tier 3)')
    print("=========== HRP Downside Beta Portfolio Completed ===========")

    # HRP - Values
    total_return, results_metrics = calc_final_results(total_return, results_metrics, returns_df=tier2,
                                                       method='hrp_val (tier 2)')
    total_return, results_metrics = calc_final_results(total_return, results_metrics, returns_df=tier3,
                                                       method='hrp_val (tier 3)')
    print("=========== HRP Value Portfolio Completed ===========")

    # # HRP - Structural Change
    # total_return, results_metrics = calc_final_results(total_return, results_metrics, returns_df=tier2,
    #                                                    method='hrp_strc (tier 2)')
    # total_return, results_metrics = calc_final_results(total_return, results_metrics, returns_df=tier3,
    #                                                    method='hrp_strc (tier 3)')
    # print("=========== HRP Structural Break Portfolio Completed ===========")

    # display/plot results
    total_return.loc['2003-01-01':, :].apply(lambda x: x/x[0]).plot(grid=True, title='Cumulative Return', figsize=[12, 8])
    plt.savefig("pic/total_return.pdf")
    # add crisis plot
    afp_plot.crisis_period_plot(tier1, total_return)

    print(results_metrics.to_string())
    total_return.to_csv("results/total_return.csv")
    results_metrics.to_csv("results/results.csv")

    print("=========== Monte Carlo Started ===========")
    # Monte Carlo, define all the methods we want to test
    methods = ['all-weather (star)' , 'hrp', 'risk_parity']
    col_name = tier3.columns
    total_return_MC, results_metrics_MC = MC.hrpMC(methods,
                                                   col_name,
                                                   numIters=10,
                                                   size0=7,
                                                   size1=7,
                                                   )
    for method in methods:
        total_return_MC[method].to_csv("results/%s.csv" % (method + " total_return_MC"))
        results_metrics_MC[method].to_csv("results/%s.csv" % (method + " results_MC"))

    print("=========== Monte Carlo Completed ===========")


if __name__ == "__main__":
    main()

