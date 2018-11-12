"""main script to buld feature matrix"""
import pandas as pd

import argparse
import sys
sys.path.append('.')

from utils.features_builder import Builder

#TODO add build2 to construct the whole feature matrix as in feature_engineering?


def build():
    """ build the dataframe containing client and bs features for each message

    Returns
    -------

    """
    parser = argparse.ArgumentParser(description='fetching and engineering features')
    parser.add_argument('set', type=str, default='train',
                        help='train or test \n weather fetch train or test set')

    args = parser.parse_args()
    set = args.set  # either 'train' or 'test'

    assert(set in ['train', 'test'])

    if set == 'train':
        #train_path = 'data/Toy/db_toy.csv'
        path = 'data/Train/train_dataset.csv' #, sep=','
        data = pd.read_csv(path, sep=',')
        train_labels = data[['messageid', 'latitude', 'longitude']] \
            .groupby('messageid', as_index=True).first()
        bsids = data['bsid'].unique()

    else: # set is train
        path = 'data/Validation/test_dataset.csv'
        data = pd.read_csv(path, sep=',')

    clt_cols = ['dtid', 'time_ux_client', 'motion', 'speed', 'data_type',
                'radius', 'seqnumber']

    bs_cols = ['nseq', 'rssi', 'snr', 'freq', 'time_ux']

    builder = Builder('messageid', 'bsid', clt_cols, bs_cols, 50)

    # df_clt_feat = client_features(train_data, clt_cols, 'messageid')
    # df_feat = bs_features(train_data, df_clt_feat, bs_cols,
    #                       'messageid', 'bsid', verbose=50)
    builder.client_features(data)
    df_gp = builder.gb_bs_features(data)

    df_client_features = builder.df_features_
    df_bs_features = builder.fast_bs_features(-1)

    # print('client features shape: {} \n basestation shape:{}'.format(builder.df_features_,
    #                                                                  df_bs_features))

    # df_res = builder.df_features_.merge(df_bs_features)
    df_res = pd.concat([df_client_features, df_bs_features], axis=1)
    # df_res.set_index('messageid', inplace=True)

    print('shape of output df: {}'.format(df_res.shape))
    print(list(df_res.columns))

    if set == 'train':
        Xy = df_res.merge(train_labels, left_index=True, right_index=True)

    else:
        Xy = df_res

    Xy['motion'] = Xy['motion'].apply(lambda u: 1 if u == 't' else 0)
    Xy.fillna({'motion': 0, 'speed': 0}, inplace=True)

    # data-type -> one-hot encoding, 0% missing values
    dtype_ohe = pd.get_dummies(Xy['data_type'])
    Xy.drop('data_type', axis=1, inplace=True)
    Xy = pd.concat([Xy, dtype_ohe], axis=1)

    # radius, numeric, 82% missing values input median
    radius_median = Xy['radius'].median()
    Xy.fillna({'radius': radius_median}, inplace=True)

    time_columns = ['time_ux{}'.format(bsid) for bsid in bsids]

    #TODO change sign
    for time_column in time_columns:
        Xy[time_column] = (Xy[time_column] // 1000 - Xy['time_ux_client'] // 1000)

    Xy.drop('time_ux_client', axis=1, inplace=True)

    print('feature matrix shape: {}'.format(Xy.shape))
    if set == 'train':
        Xy.to_parquet('data/feature_matrix_train_v0.parquet')
    # Xy.to_hdf('data/store.h5', key='feature_matrix_toy', mode='w', dropna=True)
    # store = pd.HDFStore('data/store.h5')
    # store.put('feature_matrix_toy', Xy)
    # store.close()
    else:
        Xy.to_parquet('data/feature_matrix_test_v0.csv')

    print('done')
    return 0


if __name__ == "__main__":
    build()