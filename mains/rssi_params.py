"""
script to resolve optimization problem:
$ RSSI_{reçu} = RSSI_{send} + err + 10 n log(d/d0) $

"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import norm
import warnings



#TODO consider using `jac` argument in optimize.minimize
def main():
    bs_path = 'data/base_station_coord.csv'
    data_path = 'data/Toy/db_toy.csv'
    # data_path = 'data/Train/train_dataset.csv'
    df_bs = pd.read_csv(bs_path, sep=';')
    df = pd.read_csv(data_path, sep=';')[['bsid', 'snr', 'rssi', 'latitude',
                                          'longitude']]

    df = df.merge(df_bs, on='bsid')
    res_dic = {}

    for bsid in df['bsid'].unique():
        # using scipy's optimize
        print('Optimizing bs:{}'.format(bsid))
        mask_bs = df['bsid'] == bsid
        df_masked = df[mask_bs]

        fn_args = (df_masked['lat'].values, df_masked['lng'].values,
                   df_masked['latitude'].values,
                   df_masked['longitude'].values, df_masked['rssi'].values)
        x0 = np.array([-1, 1e2])
        options = {'disp': True, 'maxiter': 30, 'gtol':1e-3}
        res_optim = minimize(objective_fn, x0=x0, args=fn_args, method='cg',
                             options=options)
        print(res_optim)
        res_dic[bsid] = (res_optim.x)

    # pd.DataFrame(res_dic).to_csv('data/rssi_params_res.csv', sep=';')


    return 0




def objective_fn(nd0, latitude_bs, longitude_bs, latitude, longitude, rssis,
                 frac=1):
    n = nd0[0]
    d0 = nd0[1]
    warnings.filterwarnings("error")

    # n_messages = latitude.shape[0]
    res = 0
    counter = 0
    for index, rssi in enumerate(rssis):
        random = np.random.rand()
        if random < frac:
            distance = np.sqrt((latitude_bs[index] - latitude[index])**2 +
                           (longitude_bs[index] - longitude[index])**2)
            try:
                new_value = (rssi + 10 * n * np.log(distance / d0))**2
                res += new_value
                counter += 1
            except RuntimeWarning:
                print("Runtime warning", distance, d0, "on iteration {}".format(counter))
                break
    # print("Objective value: {:.3f}".format(res/counter), n, d0, counter)
    return res / counter


def grad_fn_j(n_i, d0_i, latitude_bs, longitude_bs, latitude, longitude):
    distance = np.sqrt((latitude_bs - latitude)**2 + (longitude_bs - longitude)**2)
    grad_n = 10 * np.log(distance / d0_i)
    grad_d0 = (10 * n_i) / (d0_i)

    return grad_n, grad_d0





if __name__ == "__main__":
    main()