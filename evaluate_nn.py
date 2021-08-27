import numpy as np
import pandas as pd
import nn
import numpy.random as rng
import json
import sys 
import tensorflow as tf
import feature_reader

path = sys.argv[1]
output_path = sys.argv[2]

with open(path, 'r') as f:
    cfg = json.load(f)
loader_func = getattr(feature_reader, cfg['model']['features'])

train_df = pd.read_csv('iso_train.csv')
X = loader_func(train_df)
Y = np.array(train_df[cfg['output_cols']])

valid_ix = train_df['is_valid'] == 1
train_ix = train_df['is_valid'] == 0

Xtrain = X[train_ix,:]
Ytrain = Y[train_ix,:]
Xvalid = X[valid_ix,:]
Yvalid = Y[valid_ix,:]

print("Train size: %d, Validation: %d" % (Xtrain.shape[0], Xvalid.shape[0]))

test_df = pd.read_csv('iso_test.csv')
Xtest = loader_func(test_df)
Ytest = np.array(test_df[cfg['output_cols']])
print("Test size: %d" % (Xtest.shape[0]))

model = nn.FeedforwardNN(cfg['model'])
model.train(Xtrain, Ytrain, Xvalid, Yvalid)
yhat = model.predict(Xtest)
ae = np.abs(yhat - Ytest)
mae_by_output = [float(e) for e in np.mean(ae, axis=0)]
cfg['test_mae_by_output'] = { c: v for c,v in zip(cfg['output_cols'], mae_by_output) }

with open(output_path, 'w') as f:
    json.dump(cfg, f, indent=4)
