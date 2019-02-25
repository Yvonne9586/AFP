import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model


def calc_weights(method='risk_parity', vol_forecast_df=None):
    if method == 'risk_parity':
        inv_vol = 1/vol_forecast_df
        inv_vol_sum = inv_vol.sum(axis=1)
        weights = inv_vol.apply(lambda x: x/inv_vol_sum, axis=0)
        return weights
    return None


def get_data(file_location):
    # get the data, clean it, merge it
    df = pd.read_csv(file_location, index_col=0).dropna()
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
    sr = ret.mean() / ret.std() * np.sqrt(252)
    skew = ret.skew()
    kurt = ret.kurtosis()
    return sr * (1 + skew / 6 * sr - (kurt - 3) / 24 * sr ** 2)


def cert_eqv_ret(ret, gamma=3, rf=2):
    mu = ret.mean() * 252
    sigma = ret.std() * np.sqrt(252)
    return (mu - rf) - gamma * 0.5 * sigma ** 2


def max_drawdown(ret):
    dd = ret.cumsum() / ret.cummax() - 1
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
        'mean': returns.mean(),
        'std': returns.std(),
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
        r_vol = returns_df.rolling(252).std() * (252 ** 0.5)
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


def main():
    # get index data
    index_df = get_data("data/indexes.csv")
    index_df = index_df.loc[:, ['US10Y', 'RTY INDEX', 'SPX INDEX', 'GOLD']]
    # get percentage change
    index_change_df = index_df.pct_change().dropna()

    # forecast volatility and variance
    vol_forecast_df, var_forecast_df = calc_vol_forecast(index_change_df, method='r_vol')

    # calculate weights
    weights_df = calc_weights(method='risk_parity', vol_forecast_df=vol_forecast_df.dropna())

    # calc rebal
    df_rebal = calc_results_matrix(index_df=index_df, weights_df=weights_df, rebal_period='M')
    total_return = df_rebal.sum(axis=1)

    # output dfs
    # df_rebal.to_csv('data/rebal.csv')
    # weights_df.to_csv('data/weights.csv')
    # vol_forecast_df.to_csv('data/vol_forecast.csv')

    # display/plot results
    plt.rcParams["figure.figsize"] = (8, 5)
    total_return.plot(grid=True, title='Risk parity')
    results_metrics = calc_metrics('Risk parity', total_return, weights_df)
    print(results_metrics.to_string())
    plt.show()


if __name__ == "__main__":
    main()
