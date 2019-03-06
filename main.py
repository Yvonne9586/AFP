import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import hrp
import scipy.cluster.hierarchy as sch
import scipy.optimize


def F(w, cov_matrix):
    """equal risk parity forula"""
    N = cov_matrix.shape[0]
    return np.dot(w, np.diag(cov_matrix)) * N - np.dot(w, cov_matrix.dot(w))


def calc_eq_risk_parity_weights(x, cov_forecast_df):
    """calculate weight for each asset,
    assigning equal risk to each asset,
    only constraint is weight sum to 1,
    each asset's weight constrained from -1 to 1"""
    date = x.index.values[0][0]
    cov_matrix = cov_forecast_df.loc[date]
    N = cov_matrix.shape[0]
    para_init = np.ones(N)/N
    bounds = ((-1.0, 1.0),) * N
    w = scipy.optimize.minimize(F, x0=para_init,
                                args=(cov_matrix),
                                method='SLSQP',
                                constraints=({'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs)}),
                                bounds=bounds)
    return w.x


def calc_hrp_corr_weights(x, corr_forecast_df, cov_forecast_df, return_mean, obj_df=None, measure='corr'):
    date = x.index.values[0][0]
    corr_matrix = corr_forecast_df.loc[date]
    cov_matrix = cov_forecast_df.loc[date]
    mean_matrix = return_mean.loc[date]
    if measure in ['drawdown', 'beta', 'value', 'str_change']:
        obj = obj_df.loc[date]
        dist = np.zeros((obj.shape[0], obj.shape[0]))
        for h in range(obj.shape[0]):
            for k in range(obj.shape[0]):
                dist[h, k] = np.abs(obj.iloc[h] - obj.iloc[k])
        dist = pd.DataFrame(dist, columns=obj.index, index=obj.index)

    elif measure == 'corr':
        dist = ((1 - corr_matrix) / 2.) ** .5
    elif measure == 'beta':
        beta = beta_df.loc[date]
        dist = np.zeros((beta.shape[0],beta.shape[0]))
        for h in range(beta.shape[0]):
            for k in range(beta.shape[0]):
                dist[h,k] = np.abs(beta.iloc[h] - beta.iloc[k])
        dist = pd.DataFrame(dist, columns = beta.index, index = beta.index)
    else:
        dist = ((1 - corr_matrix) / 2.) ** .5

    link = sch.linkage(dist, 'single')
    sortIx = hrp.getQuasiDiag(link)
    sortIx = corr_matrix.index[sortIx].tolist()
    df0 = corr_matrix.loc[sortIx, sortIx]

    hrp = hrp.getRecBipart(cov_matrix, sortIx, mean_matrix)
    hrp[np.abs(hrp) > 1] = 1
    hrp = hrp / hrp.sum()
    return hrp


def calc_weights(method='risk_parity',
                 vol_forecast_df=None,
                 corr_forecast_df=None,
                 cov_forecast_df=None,
                 returns_df=None,
                 lookback_period=60):
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
        return_mean = returns_df.rolling(lookback_period).mean()
        w = corr_forecast_df.groupby(level=0).apply(
            lambda x: calc_hrp_corr_weights(x, corr_forecast_df, cov_forecast_df, return_mean)).unstack(level=-1)

        return w
    elif method == 'hrp_dd':
        return_mean = returns_df.rolling(lookback_period).mean()
        # can be changed to expanding if want to take into account all data
        drawdown_df = returns_df.expanding(lookback_period).apply(lambda x: -hrp.mdd(x), raw=True)

        w = corr_forecast_df.groupby(level=0).apply(
            lambda x: calc_hrp_corr_weights(x, corr_forecast_df, cov_forecast_df, return_mean, drawdown_df=drawdown_df,
                                            measure='drawdown')).unstack(level=-1)
        return w

    # elif method == 'hrp_beta':
    #     beta = returns_df.rolling(lookback_period).apply(lambda x: HRP.beta4_standardized(x), raw = False)
    #     beta = beta.dropna()

    #     w = corr_forecast_df.groupby(level=0).apply(
    #         lambda x: calc_hrp_corr_weights(x, corr_forecast_df, cov_forecast_df, return_mean, beta_df=beta,
    #                                         measure='beta')).unstack(level=-1)

    #     return w


def get_data(file_location):
    # get the data, clean it, merge it
    df = pd.read_csv(file_location, index_col=0)#.dropna()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index(ascending=True)
    return df


def calc_rebal(x, portfolio_df, returns_df, weights_df, txn_cost):
    prev_period = x.index[0]
    curr_period = x.index[1]

    # calculate percentage increase over the period
    # prev_index_values = index_df.loc[:prev_period, :].iloc[-1]
    # curr_index_values = index_df.loc[:curr_period, :].iloc[-1]
    period_return = returns_df.loc[curr_period, :].values + 1

    # apply to portfolio
    prev_port_values = portfolio_df.loc[:prev_period, :].iloc[-1]
    rebal_return = prev_port_values*period_return

    # find the size of portfolio
    total_size = rebal_return.sum()

    # rebalance portfolio
    new_weights = weights_df.loc[:curr_period, :].iloc[-1]
    portfolio_df.loc[curr_period, :] = new_weights*total_size

    # calc txn costs
    costs = np.abs(portfolio_df.loc[curr_period, :] - rebal_return)*txn_cost
    portfolio_df.loc[curr_period, :] = portfolio_df.loc[curr_period, :] - costs
    return 0


def adj_sr(ret):
    sr = ret.mean() / ret.std() * np.sqrt(12)
    skew = ret.skew()
    kurt = ret.kurtosis()
    return sr * (1 + skew / 6 * sr - (kurt - 3) / 24 * sr ** 2)


def cert_eqv_ret(ret, gamma=3, rf=2):
    mu = ret.mean() * 12
    sigma = ret.std() * np.sqrt(12)
    return (mu - rf) - gamma * 0.5 * sigma ** 2


def max_drawdown(ret):
    ret_sum = ret.cumsum()
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
    returns = (car - car.shift(1))/car.shift(1)
    results = {
        'total return': car.values[-1]/car.values[0],
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
                        rebal_period='M',
                        txn_cost=0.001,
                        saving_spread=0.005,
                        borrowing_spread=0.025,
                        leverage_cap=8,
                        execution_delay=0):

    # setup the portfolio as the weights df and resample at the desired rebal_period
    portfolio_df = weights_df.resample(rebal_period).last()
    # bit of a hack in using rolling
    portfolio_df.rolling(2).apply(lambda x: calc_rebal(x, portfolio_df, returns_df, weights_df, txn_cost), raw=False)

    return portfolio_df.dropna()


def calc_vol_forecast(returns_df, method='r_vol', lookback_period=24):
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


def get_static_benchmark(tier_df, total_return, results_metrics, method, tier_name):
    """calculate benchmark including: all_weather, equity_bond"""
#      static rebal
    if method == "all_weather":
        aw_df = tier_df.loc[:, ['Gold', 'TRCommodity', 'USBond10Y', 'USEq']].dropna()
        weights_dict = {'Gold': 0.075, 'TRCommodity': 0.075, 'USBond10Y': 0.55, 'USEq': 0.3}  # removed USBond5Y
    elif method == "equity_bond":
        aw_df = tier_df.loc[:, ['USBond10Y', 'USEq']].dropna()
        weights_dict = {'USBond10Y': 0.4, 'USEq': 0.6}
    weights_aw = aw_df.copy().apply(lambda x: pd.Series(aw_df.columns.map(weights_dict).values), axis=1)
    weights_aw.columns = aw_df.columns
    aw_rebal = calc_results_matrix(index_df=aw_df, weights_df=weights_aw.iloc[1:, :], rebal_period='M')

#    combine results
    title = method + tier_name
    total_return = pd.concat([total_return, aw_rebal.sum(axis=1).rename(title)], axis=1).dropna()
    results_metrics = pd.concat([results_metrics, calc_metrics(title, total_return[title], weights_aw)], axis=0)

    return total_return, results_metrics


def get_dynamic_result(tier_df, total_return, results_metrics, method, tier_name):
    """calculate dynamic allocation results"""
    # get percentage change: price -> return
    tier_change_df = tier_df.pct_change().dropna()


def calc_final_results(total_return,
                       results_metrics,
                       vol_forecast_df=None,
                       cor_forecast_df=None,
                       cov_forecast_df=None,
                       returns_df=None,
                       weights_df=pd.DataFrame(), 
                       method=''):
    method_name = method
    method = method.split(' ')[0]
    if len(weights_df) == 0:
        weights_df = calc_weights(method=method,
                              vol_forecast_df=vol_forecast_df,
                              corr_forecast_df=cor_forecast_df,
                              cov_forecast_df=cov_forecast_df,
                              returns_df=returns_df)
    rebal_df = calc_results_matrix(returns_df=returns_df, weights_df=weights_df, rebal_period='M')
    total_return = pd.concat([total_return, rebal_df.sum(axis=1).rename(method_name)], axis=1)
    results_metrics = pd.concat([results_metrics, calc_metrics(method_name, total_return[method_name].dropna(), weights_df)], axis=0)
    return total_return, results_metrics


def main():
    # get Tier data
    ret_df = get_data("data/combined_dataset_new.csv").pct_change()
    ret_df = ret_df.replace(0.0, np.nan)    # to prevent volatility to explode
    tier1 = ret_df.loc[:, ['USEq', 'USBond10Y']].dropna()
    tier2 = ret_df.loc[:, 'GermanBond10Y':'USEq'].dropna()
    tier3 = ret_df.dropna()
    total_return = pd.DataFrame()
    results_metrics = pd.DataFrame()

    # forecast volatility and variance
    vol_tier2, var_tier2 = calc_vol_forecast(tier2, method='r_vol')
    cor_tier2, cov_tier2 = calc_cor_forecast(tier2, method='r_cor')
    vol_tier3, var_tier3 = calc_vol_forecast(tier3, method='r_vol')
    cor_tier3, cov_tier3 = calc_cor_forecast(tier3, method='r_cor')

    ################# Benchmark ##################
    # benchmark 1 - 60/40
    weights_dict = {'USBond10Y': 0.4, 'USEq': 0.6}
    weights_b1 = tier1.copy().apply(lambda x: pd.Series(tier1.columns.map(weights_dict).values), axis=1)
    total_return, results_metrics = calc_final_results(total_return, results_metrics, returns_df=tier1, weights_df=weights_b1, method='60/40')

    # benchmark 2 - All Weather
    aw_df = tier2.loc[:, ['Gold', 'TRCommodity', 'USBond10Y', 'USEq']].dropna()
    weights_dict = {'Gold':0.075, 'TRCommodity':0.075, 'USBond10Y':0.55, 'USEq':0.3}
    weights_aw = aw_df.copy().apply(lambda x: pd.Series(aw_df.columns.map(weights_dict).values), axis=1)
    total_return, results_metrics = calc_final_results(total_return, results_metrics, returns_df=aw_df, weights_df=weights_aw, method='all-weather')

    # benchmark 3 - Risk Parity Tier 2
    total_return, results_metrics = calc_final_results(total_return, results_metrics, vol_forecast_df=vol_tier2, cor_forecast_df=cor_tier2,
                       cov_forecast_df=cov_tier2, returns_df=tier2, method='risk_parity (tier2)')

    # benchmark 4 - Risk Parity Tier 3
    total_return, results_metrics = calc_final_results(total_return, results_metrics, vol_forecast_df=vol_tier3, cor_forecast_df=cor_tier3,
                       cov_forecast_df=cov_tier3, returns_df=tier3, method='risk_parity (tier3)')

    # ################## HRP ##################
    # # HRP - Covariance
    # total_return, results_metrics = calc_final_results(total_return, results_metrics, vol_forecast_df=vol_tier2, cor_forecast_df=cor_tier2,
    #                    cov_forecast_df=cov_tier2, returns_df=tier2, method='hrp')
    #
    # # HRP - Maximum Drawdown
    # total_return, results_metrics = calc_final_results(total_return, results_metrics, vol_forecast_df=vol_tier2, cor_forecast_df=cor_tier2,
    #                    cov_forecast_df=cov_tier2, returns_df=tier2, method='hrp_dd')

    # HRP - Downside Beta

    # HRP - Values

    # HRP Structural Change

    # # get index data
    # index_df = get_data("data/indexes.csv")
    # index_df = index_df.loc[:, ['US10Y', 'RTY INDEX', 'SPX INDEX', 'GOLD']].dropna()
    # # get percentage change
    # index_change_df = index_df.pct_change().dropna()

    

    # # calculate weights
    # weights_df = calc_weights(method='hrp_dd',
    #                           vol_forecast_df=vol_forecast_df.dropna(),
    #                           corr_forecast_df=cor_forecast_df.dropna(),
    #                           cov_forecast_df=cov_forecast_df.dropna(),
    #                           returns_df=index_change_df.dropna())

    # # calc rebal
    # df_rebal = calc_results_matrix(index_df=index_df, weights_df=weights_df, rebal_period='M')
    # total_return = df_rebal.sum(axis=1).rename('Risk Parity')

    # # calc metrics
    # results_metrics = calc_metrics('Risk parity', total_return, weights_df)

    # display/plot results
    plt.rcParams["figure.figsize"] = (8, 5)
    total_return.plot(grid=True, title='Cumulative Return')
    print(results_metrics.to_string())
    plt.show()


if __name__ == "__main__":
    main()
