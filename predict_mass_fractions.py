import numpy as np
import pandas as pd
import nn
import numpy.random as rng
import json

with open('predict_mass_fractions.json', 'r') as f:
    cfg = json.load(f)

df = pd.read_csv('iso.csv')
X = np.array(df[['r1_temp', 'r2_temp', 'r1_pressure', 'r2_pressure']])
Y = np.array(df[cfg['output_cols']])

print("Dataset size: %d" % (X.shape[0]))

list_train_ix = []
list_yhat = []
for r in range(cfg['reps']):

    ix = rng.permutation(X.shape[0])
    n_test = int(cfg['test_prop'] * X.shape[0])
    n_train = X.shape[0] - n_test

    train_ix = np.zeros(X.shape[0]).astype(bool)
    train_ix[ix[:n_train]] = 1
    list_train_ix.append(train_ix)

    Xtrain = X[train_ix,:]
    Ytrain = Y[train_ix,:]
    
    sublist_yhat = []
    
    for model_cfg in cfg['models']:

        model = nn.FeedforwardNN(model_cfg)

        model.train(Xtrain, Ytrain)

        Yhat = model.predict(X)

        print("Completed (%d, %s)" % (r, model_cfg['name']))

        sublist_yhat.append(Yhat)
    
    list_yhat.append(sublist_yhat)

np.savez("tmp/mass_fractions_preds.npz", list_train_ix=list_train_ix, list_yhat=list_yhat, cfg=cfg)
