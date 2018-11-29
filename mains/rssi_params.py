"""
script to resolve optimization problem:
$ RSSI_{reçu} = RSSI_{send} + err + 10 n log(d/d0) $

"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.optimize.linesearch import LineSearchWarning
from scipy.linalg import norm
from multiprocessing import Pool
from functools import partial
import warnings
from geopy.distance import geodesic



#TODO consider using `jac` argument in optimize.minimize
def main():
    bs_path = 'data/base_station_coord.csv'
    data_path = 'data/Toy/db_toy.csv'
    # data_path = 'data/Train/train_dataset.csv'
    df_bs = pd.read_csv(bs_path, sep=';')
    df = pd.read_csv(data_path, sep=';')[['bsid', 'snr', 'rssi', 'latitude',
                                          'longitude', 'freq']]

    df = df.merge(df_bs, on='bsid')
    # res_dic = {}

    bsids = df['bsid'].unique()

    n_jobs=4

    pool = Pool(n_jobs)
    optimize_fn = partial(optimize_bs, df)
    res = pool.map(optimize_fn, bsids)
    pool.close()
    pool.join()

    # df_res = pd.concat(res)
    df_res = pd.DataFrame(res)
    df_res.to_csv('data/rssi_params_toy.csv', sep=';')
    print(df_res.head())

    return 0

def optimize_bs(df, bsid):
    # using scipy's optimize
    print('Optimizing bs: {} --- '.format(bsid))
    mask_bs = df['bsid'] == bsid
    df_masked = df[mask_bs]

    fn_args = (df_masked['lat'].values, df_masked['lng'].values,
               df_masked['latitude'].values,
               df_masked['longitude'].values, df_masked['rssi'].values,
               df_masked['freq'].values)
    #x0 = np.array([1, 10, -200])
    x0 = np.array([2, -50, 30])  # (n, A, C)
    options = {'disp': False, 'maxiter': 20}
    rssi_max = df_masked['rssi'].max()
    constraints = [{'type': 'ineq', 'fun': lambda x: x[1] - rssi_max},
                   {'type': 'ineq', 'fun': lambda x: x[2]}]
    warnings.filterwarnings('error')
    try:
        res_optim = minimize(objective_fn, x0=x0, args=fn_args,
                             options=options, constraints=constraints)
        print('For bs: {}'.format(bsid), res_optim)
        print('\n')
        return {'bsid': bsid, 'n': res_optim.x[0], 'A': res_optim.x[1],
                'C': res_optim.x[2],'objective': res_optim.fun}
    except (LineSearchWarning, RuntimeWarning) as e:
        print('Raised: ', e)
        return {'bsid': bsid, 'n': np.nan, 'A': np.nan, 'C': np.nan,
                'objective': np.nan}


def objective_fn(n_A_C, latitude_bs, longitude_bs, latitude, longitude, rssis,
                 freq, frac=1):
    n,  A, C = n_A_C[0], n_A_C[1], n_A_C[2]
    warnings.filterwarnings("error")

    # n_messages = latitude.shape[0]
    res = 0
    counter = 0
    for index, rssi in enumerate(rssis):
        # distance = np.sqrt((latitude_bs[index] - latitude[index])**2 +
        #                (longitude_bs[index] - longitude[index])**2)
        distance = geodesic((latitude_bs[index], longitude_bs[index]),
                            (latitude[index], longitude[index])).m
        freq_norm = freq[index] / 1e6
        new_value = \
            (rssi - A + 10*n*(np.log10(distance * freq_norm )-3) + C)**2
        res += new_value
        counter += 1
    return res/counter


def grad_fn_j(n_i, d0_i, latitude_bs, longitude_bs, latitude, longitude):
    distance = np.sqrt((latitude_bs - latitude)**2 + (longitude_bs - longitude)**2)
    grad_n = 10 * np.log(distance / d0_i)
    grad_d0 = (10 * n_i) / (d0_i)

    return grad_n, grad_d0





if __name__ == "__main__":
    main()
