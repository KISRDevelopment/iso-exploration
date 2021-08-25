import numpy as np
import pandas as pd
import nn
import numpy.random as rng
import json
import sys 
import tensorflow as tf
path = sys.argv[1]

with open(path, 'r') as f:
    cfg = json.load(f)

train_df = pd.read_csv('iso_train.csv')
Xtrain = np.array(train_df[['r1_temp', 'r2_temp', 'r1_pressure', 'r2_pressure']])
Ytrain = np.array(train_df[cfg['output_cols']])
print("Train size: %d" % (Xtrain.shape[0]))

ix = rng.permutation(Xtrain.shape[0])
n_valid = int(cfg['p_valid'] * Xtrain.shape[0])

valid_ix = ix[:n_valid]
train_ix = ix[n_valid:]


Xvalid = Xtrain[valid_ix,:]
Yvalid = Ytrain[valid_ix,:]
Xtrain = Xtrain[train_ix,:]
Ytrain = Ytrain[train_ix,:]


print("Train size: %d, Valid size: %d" % (Xtrain.shape[0], Xvalid.shape[0]))
test_df = pd.read_csv('iso_test.csv')
Xtest = np.array(test_df[['r1_temp', 'r2_temp', 'r1_pressure', 'r2_pressure']])
Ytest = np.array(test_df[cfg['output_cols']])
print("Test size: %d" % (Xtest.shape[0]))

cce = tf.keras.losses.CategoricalCrossentropy()

for model_cfg in cfg['models']:
    print(model_cfg['name'])
    model = nn.FeedforwardNN(model_cfg)

    all_yhats = []
    for i in range(cfg['ensemble_size']):
        model.train(Xtrain, Ytrain, Xvalid, Yvalid)
        yhat = model.predict(Xtest)
        all_yhats.append(yhat)
    all_yhats = np.array(all_yhats)
    yhat = np.mean(all_yhats,  axis=0)

    xe = float(cce(Ytest, yhat).numpy())
    ae = np.abs(yhat - Ytest)

    model_cfg['test_loss'] = xe 

    mae_by_output = [float(e) for e in np.mean(ae, axis=0)]
    model_cfg['test_mae_by_output'] = { c: v for c,v in zip(cfg['output_cols'], mae_by_output) }

with open('evaluation_results.json', 'w') as f:
    json.dump(cfg, f, indent=4)
