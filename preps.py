from random import gauss, seed
from math import sqrt, exp, pow
from scipy.stats import genpareto
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression


def create_GBM(s0, mu, sigma):
    """
    Generates a price following a geometric brownian motion process based on the input of the arguments:
    - s0: Asset inital price.
    - mu: Interest rate expressed annual terms.
    - sigma: Volatility expressed annual terms.
    """
    st = s0

    def generate_value():
        nonlocal st
        st *= exp((mu - 0.5 * sigma ** 2) * (1. / (365. * 24 * 12)) + sigma * sqrt(1. / (365. * 24 * 12)) *
                  gauss(mu=0, sigma=1))
        return st
    return generate_value


def create_dataset(ran_seed: int = 1234, starter: float = 100, mu: float = 0,
                   std: float = 0.1, simnum: int = 345600) -> tuple:
    """
    Creates relative return dataset and path of stock

    :param ran_seed: random seed
    :param starter: starter stock price
    :param mu: mean of starter GBM
    :param std: std of started GBM
    :param simnum: number of simulations
    :return: tuple containing relative return data days x 5_minutes and the original path dadaset
    """
    path = []
    days = []
    seed(ran_seed)
    gbm = create_GBM(starter, mu, std)
    day = 1
    for _ in range(simnum):
        if (_ + 1) % 288 == 0:
            day = day + 1
            gbm = create_GBM(st, gauss(mu=0, sigma=1) * 2, 0.1)
        days.append(day)
        st = gbm()
        path.append(st)
        if _ % 100000 == 0:
            print(f'Completion: {_ / simnum}% \n')

    data = pd.DataFrame({'DAYS': days, 'STOCK_PRICE': path})
    data['RELATIVE_RETURN'] = data['STOCK_PRICE'].shift(1) / data['STOCK_PRICE']
    by_days = data['RELATIVE_RETURN'].values.reshape(1200, int(len(data['RELATIVE_RETURN']) / 1200))
    return by_days, data


def historical_brt_diff(threshold: float, data: np.array, brt_len: int = 30, var_p: float = 0.05) -> float:
    """
    Calculates the error of a VaR induced by a given threshold and the historical one

    :param threshold: float of threshold to cut the data
    :param data: appropriate sized array of relative returns
    :param brt_len: length of EVT VaR calculation in days
    :param var_p: percentile of VaR
    :return: error magnitude
    """

    brt_data = (data[:brt_len, :] - threshold)
    evt_var = np.quantile(data[brt_len:, :], var_p)
    filt = brt_data + threshold > threshold

    gpd_values = np.squeeze(brt_data[filt])
    if len(gpd_values) == 0:
        return 1e7
    fitted = genpareto.fit(gpd_values)
    n_u = len(gpd_values)
    VAR = threshold + fitted[2] / fitted[0] * (pow(brt_len * 288 / n_u * (1 - var_p), -fitted[0]) - 1)
    return abs(VAR - evt_var) * 10000


def finalize_dataset(prev: np.array, brt_len: int = 30, window_len: int = 60, var_p: float = 0.05,
                     save: bool = False, pre_calculated=None, threshold: float = 1) -> pd.DataFrame:
    """
    Calculates additional attributes for analysis and creates dataframe

    :param prev: previously calculated data containing relative returns
    :param brt_len: length of EVT VaR calculation in days
    :param window_len: length of moving window in days
    :param var_p: percentile of VaR
    :param save: bool for saving historical BRT-s
    :param pre_calculated: if not None it is used as preloaded historical BRT
    :param threshold: float of threshold to cut the data for PoT EVT VaR
    :return: completed dataframe
    """

    brt_temp = []
    avg_20_temp = []
    std_20_temp = []
    mean_std_temp = []
    VAR_tomorrow_temp = []
    EVT_tomorrow_temp = []
    amb_temp = []
    lookback_data = []
    for i in range(1200 - window_len - 1):
        snippet_1 = prev[i:window_len + i, :]
        brt_data = snippet_1[brt_len - 1, :] - threshold
        tomorrow = snippet_1[brt_len - 1, :]
        snippet_2 = snippet_1[brt_len - 20 - 1:brt_len - 1, :]
        if pre_calculated is None:
            hist = minimize(historical_brt_diff, np.array(1), (snippet_1, brt_len, var_p))
            brt_temp.append(hist.x)
            print(hist.x)
        avg_20_temp.append(np.mean(snippet_2))
        std_local = np.std(snippet_2)
        std_20_temp.append(std_local)
        VAR_tomorrow_temp.append(np.quantile(tomorrow, var_p))
        filt = brt_data + threshold > threshold
        gpd_values = np.squeeze(brt_data[filt])
        fitted = genpareto.fit(gpd_values)
        n_u = len(gpd_values)
        EVT_tomorrow_temp.append(
            threshold + fitted[2] / fitted[0] * (pow(brt_len * 288 / n_u * (1 - var_p), -fitted[0]) - 1))
        mean_std_local = np.std(np.mean(snippet_2, axis=1))
        mean_std_temp.append(mean_std_local)
        amb_temp.append(ambiguity(std_local, mean_std_local))
        lookback_data.append(np.squeeze(snippet_2).astype(float))

    if save:
        pd.DataFrame(brt_temp).to_csv('brt_values.csv')
    if pre_calculated is None:
        df = pd.DataFrame({'historical_brt': brt_temp,
                           'avg_20': avg_20_temp,
                           'std_20': std_20_temp,
                           'mean_std': mean_std_temp,
                           'ambiguity': amb_temp,
                           'VAR_tomorrow': VAR_tomorrow_temp,
                           'EVT_tomorrow': EVT_tomorrow_temp,
                           'lookback': lookback_data}
                          )
    else:
        df = pd.DataFrame({'historical_brt': pre_calculated,
                           'avg_20': avg_20_temp,
                           'std_20': std_20_temp,
                           'mean_std': mean_std_temp,
                           'ambiguity': amb_temp,
                           'VAR_tomorrow': VAR_tomorrow_temp,
                           'EVT_tomorrow': EVT_tomorrow_temp,
                           'lookback': lookback_data}
                          )
    return df


def ambiguity(sigma: float, mean_std: float) -> float:
    """
    Calculates ambiguity based on given parameters

    :param sigma: historical std of returns
    :param mean_std: std of means of returns
    :return: calculated ambiguity
    """

    return -(mean_std * sigma * (-3 * sqrt(2 * math.pi) * sqrt(mean_std ** 2 + sigma ** 2) + mean_std * sigma * sqrt(
        (12 * mean_std ** 2 + 9 * sigma ** 2) / (mean_std ** 3 * sigma + mean_std * sigma ** 3)))) / \
           (6 * math.pi * sqrt((mean_std ** 2 + sigma ** 2) * (4 * mean_std ** 2 + 3 * sigma ** 2)))


def predict_brt(data: pd.DataFrame, train_perc: float) -> pd.DataFrame:
    """
    Trains the linear regression and adds the result to the dataframe

    :param data: previously obtained dataframe containing all necessary information for BRT regression
    :param train_perc: percentage of data to use for training
    :return: extended dataframe
    """

    train_len = int(len(data)*train_perc)
    train_X = data[['std_20', 'ambiguity']].iloc[:train_len]
    train_y = data['historical_brt'].iloc[:train_len]
    reg = LinearRegression(n_jobs=-1).fit(train_X, train_y)
    result = pd.DataFrame({'predicted_brt': reg.predict(data[['std_20', 'ambiguity']])})
    return pd.concat([data, result], axis=1)


def calculate_amb_var(data: pd.DataFrame, brt_len: int = 20, var_p: float = 0.05) -> pd.DataFrame:
    """
    Calculates the deviation-ambiguity based BRT induced EVT-VaR
    :param data: previously obtained dataframe containing all necessary information
    :param brt_len: window length for GPD fitting
    :param var_p: VaR percentile
    :return: VaR extended dataframe
    """

    var_temp = []
    for i in data[['predicted_brt', 'lookback']].values:
        threshold = float(i[0])
        brt_data = i[1] - threshold
        filt = brt_data + threshold > threshold
        try:
            gpd_values = np.squeeze(brt_data[filt])
            fitted = genpareto.fit(gpd_values)
            n_u = len(gpd_values)
            var_temp.append(threshold + fitted[2] / fitted[0] * (pow(brt_len * 288 / n_u * (1 - var_p), -fitted[0]) - 1))
        except:
            var_temp.append(1)
    return pd.concat([data, pd.DataFrame({'amb_var': var_temp})], axis=1)
