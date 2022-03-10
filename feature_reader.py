import pandas as pd 
import numpy as np 

def load_ann_features(df):
    return np.array(df[['r1_temp', 'r2_temp', 'r1_pressure', 'r2_pressure']])

def load_regression_features(df):
    r1t = np.array(df['r1_temp'])
    r2t = np.array(df['r2_temp'])
    r1p = np.array(df['r1_pressure'])
    r2p = np.array(df['r2_pressure'])

    features = [r1t, 
                r2t, 
                r1p, 
                r2p, 
                r1t * r2t, 
                r1t * r1p,
                r1t * r2p,
                r2t * r1p,
                r2t * r2p,
                r1p * r2p,
                r1t * r2t * r1p,
                r1t * r2t * r2p,
                r1t * r1p * r2p,
                r2t * r1p * r2p,
                r1t * r2t * r1p * r2p
    ]

    return np.vstack(features).T

def load_nn_features(df):
    r1t = np.array(df['r1_temp'])
    r2t = np.array(df['r2_temp'])
    r1p = np.array(df['r1_pressure'])
    r2p = np.array(df['r2_pressure'])

    r1t_features = np.vstack((r1t, r1t * r1t)).T[:,np.newaxis,:]
    r2t_features = np.vstack((r2t, r2t * r2t)).T[:,np.newaxis,:]
    r1p_features = np.vstack((r1p, r1p * r1p)).T[:,np.newaxis,:]
    r2p_features = np.vstack((r2p, r2p * r2p)).T[:,np.newaxis,:]

    return np.concatenate([r1t_features, r2t_features, r1p_features, r2p_features], axis=1)

# df = pd.read_csv('iso_train.csv')
# X = load_nn_features(df)
# print(X.shape)