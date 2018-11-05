"""main script to buld feature matrix"""
import modin.pandas as pd

from utils.features_builder import  client_features, bs_features

import os


def build():
    train_path = 'data/Toy/db_toy.csv'
    label_path = 'data/Toy/labels_toy.csv'

    train_data = pd.read_csv(train_path, sep=';')
    # train_label = pd.read_csv( label_path, sep=';')

    clt_cols = ['dtid', 'time_ux_client', 'motion', 'speed', 'data_type',
                'radius', 'seqnumber']

    bs_cols = ['nseq', 'rssi', 'snr', 'freq', 'time_ux']

    df_clt_feat = client_features(train_data, clt_cols, 'messageid')

    df_feat = bs_features(train_data, df_clt_feat, bs_cols,
                          'messageid', 'bsid')


    df_feat.head()
    print(list(df_feat.columns))

    df_feat.to_csv('data/toy_data_featurized.csv', sep=';', encoding='utf8')

    print('done')

    return 0


if __name__ == "__main__":
    build()