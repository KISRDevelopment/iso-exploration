import numpy as np
import pandas as pd
import rm
import numpy.random as rng
import json
import sys

path = sys.argv[1]
output_path = sys.argv[2]

with open(path, 'r') as f:
    cfg = json.load(f)

train_df = pd.read_csv('iso_train.csv')
X = np.array(train_df[['r1_temp', 'r2_temp', 'r1_pressure', 'r2_pressure']])
Y = np.array(train_df[cfg['output_cols']])

valid_ix = train_df['is_valid'] == 1
train_ix = train_df['is_valid'] == 0

Xtrain = X[train_ix,:]
Ytrain = Y[train_ix,:]
Xvalid = X[valid_ix,:]
Yvalid = Y[valid_ix,:]

print("Train size: %d, Validation: %d" % (Xtrain.shape[0], Xvalid.shape[0]))

test_df = pd.read_csv('iso_test.csv')
Xtest = np.array(test_df[['r1_temp', 'r2_temp', 'r1_pressure', 'r2_pressure']])
Ytest = np.array(test_df[cfg['output_cols']])
print("Test size: %d" % (Xtest.shape[0]))

model = rm.RegressionModel(cfg['model'])
model.train(Xtrain, Ytrain, Xvalid, Yvalid)
yhat = model.predict(Xtest)
ae = np.abs(yhat - Ytest)
mae_by_output = [float(e) for e in np.mean(ae, axis=0)]
cfg['test_mae_by_output'] = { c: v for c,v in zip(cfg['output_cols'], mae_by_output) }

with open(output_path, 'w') as f:
    json.dump(cfg, f, indent=4)
