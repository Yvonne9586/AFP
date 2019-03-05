import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import HRP
import scipy.cluster.hierarchy as sch

LOOKBACK_PERIOD = 2*252


def calc_hrp_corr_weights(x, corr_forecast_df, cov_forecast_df, return_mean, drawdown_df=None, measure='corr'):
    date = x.index.values[0][0]
    corr_matrix = corr_forecast_df.loc[date]
    cov_matrix = cov_forecast_df.loc[date]
    mean_matrix = return_mean.loc[date]
    if measure == 'drawdown':
        drawdown = drawdown_df.loc[date]
        dist = np.zeros((drawdown.shape[0], drawdown.shape[0]))
        for h in range(drawdown.shape[0]):
            for k in range(drawdown.shape[0]):
                dist[h, k] = np.abs(drawdown.iloc[h] - drawdown.iloc[k])
        dist = pd.DataFrame(dist, columns=drawdown.index, index=drawdown.index)
    elif measure == 'corr':
        dist = ((1 - corr_matrix) / 2.) ** .5
    else:
        dist = ((1 - corr_matrix) / 2.) ** .5

    link = sch.linkage(dist, 'single')
    sortIx = HRP.getQuasiDiag(link)
    sortIx = corr_matrix.index[sortIx].tolist()
    df0 = corr_matrix.loc[sortIx, sortIx]

    hrp = HRP.getRecBipart(cov_matrix, sortIx, mean_matrix)
    hrp[np.abs(hrp) > 1] = 1
    hrp = hrp / hrp.sum()
    return hrp


def calc_weights(method='risk_parity',
                 vol_forecast_df=None,
                 corr_forecast_df=None,
                 cov_forecast_df=None,
                 returns_df=None):
    if method == 'risk_parity':
        inv_vol = 1/vol_forecast_df
        inv_vol_sum = inv_vol.sum(axis=1)
        weights = inv_vol.apply(lambda x: x/inv_vol_sum, axis=0)
        return weights
    elif method == 'hrp':
        return_mean = returns_df.rolling(LOOKBACK_PERIOD).mean()
        w = corr_forecast_df.groupby(level=0).apply(
            lambda x: calc_hrp_corr_weights(x, corr_forecast_df, cov_forecast_df, return_mean)).unstack(level=-1)

        return w
    elif method == 'hrp_dd':
        return_mean = returns_df.rolling(LOOKBACK_PERIOD).mean()
        # can be changed to expanding if want to take into account all data
        drawdown_df = returns_df.expanding(LOOKBACK_PERIOD).apply(lambda x: -HRP.mdd(x), raw=True)

        w = corr_forecast_df.groupby(level=0).apply(
            lambda x: calc_hrp_corr_weights(x, corr_forecast_df, cov_forecast_df, return_mean, drawdown_df=drawdown_df,
                                            measure='drawdown')).unstack(level=-1)

        return w


def get_data(file_location):
    # get the data, clean it, merge it
    df = pd.read_csv(file_location, index_col=0)#.dropna()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index(ascending=True)
    return df


def calc_rebal(x, portfolio_df, index_df, weights_df, txn_cost):
    prev_period = x.index[0]
    curr_period = x.index[1]

    # calculate percentage increase over the period
    prev_index_values = index_df.loc[:prev_period, :].iloc[-1]
    curr_index_values = index_df.loc[:curr_period, :].iloc[-1]
    period_return = curr_index_values/prev_index_values

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


def calc_results_matrix(index_df,
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
    portfolio_df.rolling(2).apply(lambda x: calc_rebal(x, portfolio_df, index_df, weights_df, txn_cost), raw=False)

    return portfolio_df


def calc_vol_forecast(returns_df, method='r_vol'):
    if method == 'r_vol':
        r_vol = returns_df.rolling(LOOKBACK_PERIOD).std() * (252 ** 0.5)
        r_var = r_vol*r_vol
        return r_vol, r_var
    elif method == 'garch':
        r_vol = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
        r_var = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
        for index in returns_df.columns:
            returns_index = returns_df[index]
            am = arch_model(returns_index, vol='Garch', p=1, o=0, q=1, dist='Normal')
            res = am.fit(last_obs=returns_index.index[252])
            tmp = res.forecast(horizon=20)
            var_proj = tmp.variance.mean(axis=1) * 252
            std_proj = np.power(var_proj, 0.5)
            r_vol[index] = std_proj
            r_var[index] = var_proj
        return r_vol, r_var
    return None, None


def calc_cor_forecast(returns_df, method='r_cor'):
    if method == 'r_cor':

        r_corr = returns_df.rolling(LOOKBACK_PERIOD).corr()
        r_cov = returns_df.rolling(LOOKBACK_PERIOD).cov() * (252 ** 0.5)
        return r_corr, r_cov

    return None, None


def main():
    # get index data
    index_df = get_data("data/indexes.csv")
    index_df = index_df.loc[:, ['US10Y', 'RTY INDEX', 'SPX INDEX', 'GOLD']].dropna()
    # get percentage change
    index_change_df = index_df.pct_change().dropna()

    # forecast volatility and variance
    vol_forecast_df, var_forecast_df = calc_vol_forecast(index_change_df, method='r_vol')
    cor_forecast_df, cov_forecast_df = calc_cor_forecast(index_change_df, method='r_cor')

    # calculate weights
    weights_df = calc_weights(method='hrp_dd',
                              vol_forecast_df=vol_forecast_df.dropna(),
                              corr_forecast_df=cor_forecast_df.dropna(),
                              cov_forecast_df=cov_forecast_df.dropna(),
                              returns_df=index_change_df.dropna())

    # calc rebal
    df_rebal = calc_results_matrix(index_df=index_df, weights_df=weights_df, rebal_period='M')
    total_return = df_rebal.sum(axis=1).rename('Risk Parity')

    # calc metrics
    results_metrics = calc_metrics('Risk parity', total_return, weights_df)

    # output dfs
    # df_rebal.to_csv('data/rebal.csv')
    # weights_df.to_csv('data/weights.csv')
    # vol_forecast_df.to_csv('data/vol_forecast.csv')

    # all-weather static rebal
    aw_df = get_data("data/gfd_monthly.csv").loc[:, ['Gold', 'TRCommodity', 'USBond10Y', 'USBond5Y', 'USEq']].dropna()
    weights_dict = {'Gold':0.075, 'TRCommodity':0.075, 'USBond10Y':0.4, 'USBond5Y':0.15, 'USEq':0.3}
    weights_aw = aw_df.copy().apply(lambda x: pd.Series(aw_df.columns.map(weights_dict).values), axis=1)
    weights_aw.columns = aw_df.columns
    aw_rebal = calc_results_matrix(index_df=aw_df, weights_df=weights_aw.iloc[1:, :], rebal_period='M')

    # all-weather results
    total_return = pd.concat([total_return, aw_rebal.sum(axis=1).rename('All Weather')], axis=1).dropna()
    results_metrics = pd.concat([results_metrics, calc_metrics('All Weather', total_return['All Weather'], weights_aw)], axis=0)

    # display/plot results
    plt.rcParams["figure.figsize"] = (8, 5)
    total_return.plot(grid=True, title='Cumulative Return')
    
    print(results_metrics.to_string())
    plt.show()


if __name__ == "__main__":
    main()
