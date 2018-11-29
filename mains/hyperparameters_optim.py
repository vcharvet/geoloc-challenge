"""main script to search classifiers' optimal hyperparameters"""
import pandas as pd
import hyperopt as hp
import numpy as np
from hyperopt import fmin, tpe, rand, hp, Trials, STATUS_OK
from hyperopt.pyll.stochastic import sample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from xgboost import XGBRegressor
from scipy.sparse import csr_matrix

import gc
import re
import sys
sys.path.append('.')
from utils.evaluation import criterion, vincenty_vec



def optimize_rf():
    #train_path = 'data/feature_matrix_toy.parquet'
    train_path = 'data/feature_matrix_train_vMLP.parquet'
    Xy = pd.read_parquet(train_path)

    dtid_ohe = pd.get_dummies(Xy['dtid'])
    Xy.drop('dtid', axis=1, inplace=True)
    Xy = pd.concat([Xy, dtid_ohe], axis=1)
    del dtid_ohe

    # features = np.loadtxt('data/features.txt', dtype=str)
    features = list(Xy.columns)
    features.remove('latitude')
    features.remove('longitude')

    r_nseq = re.compile('nseq')
    nseqs = [x for x in features if r_nseq.match(x)]

    r_time = re.compile('time_ux')
    times = [x for x in features if r_time.match(x)]

    r_freq = re.compile('freq')
    freqs = [x for x in features if r_freq.match(x)]

    r_snr = re.compile('snr')
    snrs = [x for x in features if r_snr.match(x)]

    r_dist = re.compile('estim_distance')
    dists = [x for x in features if r_dist.match(x)]

    X, y = Xy.loc[:, features], Xy.loc[:, ['latitude', 'longitude']]
    del Xy

    X.drop(nseqs + times + freqs + snrs, axis=1, inplace=True)
    
    mean_lat, std_lat = y['latitude'].mean(), y['latitude'].std()
    mean_lng, std_lng = y['longitude'].mean(), y['longitude'].std()

    mask = (y['latitude'] < mean_lat + 2*std_lat ) & (y['latitude'] > mean_lat - 2*std_lat) & \
            (y['longitude'] < mean_lng + 2*std_lng) & (y['longitude'] > mean_lng - 2*std_lng)

    n_zeros = np.count_nonzero(X, axis=0)

    treshold = int(X.shape[0] // 50)

    feature_mask = np.where(n_zeros >= treshold, True, False)
    
    # X_scale = StandardScaler().fit_transform(X)

    # X_scale_sparse = csr_matrix(X_scale)
    # print(sys.getsizeof(X_scale), sys.getsizeof(X_scale_sparse))
    # x_train, x_test, y_train, y_test = \
    #         train_test_split(X_scale, y, test_size=0.1, random_state=0)
    # df_ytest = pd.DataFrame(y_test, columns=['latitude', 'longitude'])
    custom_loss = make_scorer(criterion, greater_is_better=False, is_df=False)

    ## for random forest
    # space = { #TODO consider changing max_features param
    #     'n_estimators': hp.qlognormal('n_estimators', 4, 1,  1),
    #     'max_depth': hp.qnormal('max_depth', 30, 5, 1)
    # }
    space = {'max_depth': hp.qnormal('max_depth',15, 5, 1),
             'n_estimators': hp.qnormal('n_estimators', 100, 10, 1),
             'learning_rate': hp.qnormal('learning_rate', 0.1, 0.03, 1)
             }
    df_result_hp = pd.DataFrame(columns=['LOSS', 'estimators'] + list(space.keys()))

    i = 0
    def objective(space, i):
        # global i
        i += 1
        # clf = RandomForestRegressor(n_estimators=int(space['n_estimators'])+1,
        #                             max_depth=int(space['max_depth'])+1,
        #                             max_features='sqrt',
        #                             n_jobs=24)
        clf = XGBRegressor(n_estimators=int(space['n_estimators'])+1,
                           max_depth=int(space['max_depth'])+1,
                           learning_rate=space['learning_rate'],
                           n_jobs=3)
        cv = cross_val_score(clf, X.loc[mask, :], y.values[mask], scoring=custom_loss,
                             cv=5, verbose=0, n_jobs=5)
        loss = - cv.mean()
        variance = cv.std()

        # clf.fit(x_train, y_train)
        # pred = pd.DataFrame(clf.predict(x_test), columns=['pred_lat', 'pred_long'])
        # loss = criterion(pred, df_ytest, True)
        print(
            "iteration {} on {} \n >>> criterion={:.3f}".format(i, space, loss))

        df_result_hp.loc[i, ['LOSS', 'estimators'] + list(space.keys())] = \
            [loss, clf] + list(space.values())
        return {'loss': loss, 'status': STATUS_OK, 'loss_variance': variance}


    trials = Trials()

    best = fmin(fn=lambda x: objective(x, i), space=space, algo=tpe.suggest,
                max_evals=50, trials=trials)
    print(best)
    df_result_hp.to_csv('data/hyperopt_xgb_v0.csv', sep=';')

    print("done")

    return 0



def optimize_xgb(): #@TODO
    train_path = 'train_path'
    df = pd.read_csv(train_path, sep=';')

    features = list(df.columns)
    features.remove('latitude')
    features.remove('longitude')

    X, y = df.loc[:, features], df.loc[:, ['latitude', 'longitude']]

    X_scale = StandardScaler().fit_transform(X)
    x_train, x_test, y_train, y_test = \
            train_test_split(X_scale, y, test_size=0.1, random_state=0)


if __name__ == "__main__":
    optimize_rf()
