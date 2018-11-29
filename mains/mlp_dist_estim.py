""" distance estimation using multilayer perceptron"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from geopy.distance import geodesic
from multiprocessing import Pool
from scipy.stats import linregress
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.callbacks import EarlyStopping
from keras.losses import mean_squared_error

def main():
    train_path = 'data/Toy/db_toy.csv'
    # train_path = 'data/Train/train_dataset.csv'
    train_df = pd.read_csv(train_path, sep=';')
    df_bs = pd.read_csv('data/base_station_coord.csv', sep=';').drop('Unnamed: 0', axis=1)

    features = ['snr', 'freq', 'rssi', 'bsid', 'latitude', 'longitude']
    df_features = train_df[features].merge(df_bs, on='bsid')
    df_features.rename({'lng': 'longitude_bs', 'lat': 'latitude_bs'}, axis=1, inplace=True)

    pool = Pool(4)
    distances = pool.map(exact_dist, df_features.iterrows())
    pool.close(); pool.join()

    df_features['exact_distance'] = distances

    X = df_features[['latitude_bs', 'longitude_bs', 'rssi', 'freq']]
    y = df_features['exact_distance']

    X_scale = StandardScaler().fit_transform(X)
    x_train, x_test ,y_train, y_test = \
        train_test_split(X_scale, y, test_size=0.25)

    # data augmentation
    noise = np.random.normal(0, 0.01, x_train.shape[0]).reshape((x_train.shape[0], 1))
    x_train = np.vstack((x_train, x_train + noise))
    y_train = np.hstack((y_train, y_train))
    mask = np.where(y_train < 1000, True, False)

    # multilayer perceptron
    model = Sequential()

    model.add(Dense(16, activation='relu', input_dim=x_train.shape[1]))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='relu'))

    model.compile(loss=mean_squared_error, optimizer='adam',
                  metrics=[mean_squared_error])

    model.fit(x_train, y_train, epochs=50,
              validation_split=0.1,
              verbose=1,
              callbacks=[EarlyStopping(min_delta=0.1, patience=5)])

    pred = model.predict(x_test).flatten()
    print(model.evaluate(x_test, y_test))
    print(linregress(y_test, np.where(pred >= 0, pred, 0)))

    np.savetxt('data/mlp_dist.txt', pred)



    return 0

def exact_dist(index_row):
    index, row = index_row
#     print(index)
    return geodesic((row['latitude'], row['longitude']), (row['latitude_bs'], row['longitude_bs'])).km

if __name__ == "__main__":
    main()