"""main script to buld feature matrix"""
# import modin.pandas as pd
import pandas as pd

import sys
sys.path.append('.')

from utils.features_builder import Builder


def build():
    train_path = 'data/Toy/db_toy.csv'
    # train_path = 'data/Train/Train_dataset-002.csv' , sep=','
    train_data = pd.read_csv(train_path, sep=';')

    clt_cols = ['dtid', 'time_ux_client', 'motion', 'speed', 'data_type',
                'radius', 'seqnumber']

    bs_cols = ['nseq', 'rssi', 'snr', 'freq', 'time_ux']

    builder = Builder('messageid', 'bsid', clt_cols, bs_cols, 50)

    # df_clt_feat = client_features(train_data, clt_cols, 'messageid')
    # df_feat = bs_features(train_data, df_clt_feat, bs_cols,
    #                       'messageid', 'bsid', verbose=50)
    builder.client_features(train_data)
    df_gp = builder.gb_bs_features(train_data)

    df_client_features = builder.df_features_
    df_bs_features = builder.fast_bs_features(2)

    # print('client features shape: {} \n basestation shape:{}'.format(builder.df_features_,
    #                                                                  df_bs_features))

    # df_res = builder.df_features_.merge(df_bs_features)
    df_res = pd.concat([df_client_features, df_bs_features], axis=1)

    print('shape of output df: {}'.format(df_res.shape))
    print(list(builder.df_features_.columns))

    #TODO write df to archive .tar.gz
    # builder.df_features_.to_csv('data/train_data_featurized_v0.csv',
    #                             sep=';', encoding='utf8')

    print('done')

    return 0


if __name__ == "__main__":
    build()