import numpy as np
import pandas as pd
import nn
import numpy.random as rng
import json
import sys 
from sklearn.model_selection import KFold
import itertools 
import tensorflow as tf 
path = sys.argv[1]

P_VALID = 0.2

with open(path, 'r') as f:
    cfg = json.load(f)

# read dataset
df = pd.read_csv(cfg['dataset'])
X = np.array(df[['r1_temp', 'r2_temp', 'r1_pressure', 'r2_pressure']])
Y = np.array(df[cfg['output_cols']])
print("Dataset size: %d" % (X.shape[0]))

# get combinations
keys = list(cfg['hyperparams'].keys())
search_space = [cfg['hyperparams'][k] for k in keys]
cfgs = []
base_cfg = cfg['model']
for comb in itertools.product(*search_space):
    new_cfg = dict(base_cfg)
    new_cfg.update({ k: v for k,v in zip(keys, comb ) })
    new_cfg['results'] = []
    new_cfg['comb'] = comb
    cfgs.append(new_cfg)
    
# evaluate CV performance of each combination
kf = KFold(n_splits=cfg['folds'])
cce = tf.keras.losses.CategoricalCrossentropy()

for train_index, test_index in kf.split(X):

    rng.shuffle(train_index)

    n_valid = int(P_VALID * len(train_index))

    valid_index = train_index[:n_valid]
    train_index = train_index[n_valid:]

    Xtrain = X[train_index,:]
    Ytrain = Y[train_index,:]

    Xvalid = X[valid_index,:]
    Yvalid = Y[valid_index,:]

    Xtest = X[test_index,:]
    Ytest = Y[test_index,:]

    for cfg in cfgs:
        
        model = nn.FeedforwardNN(cfg)

        # train
        
        model.train(Xtrain, Ytrain, Xvalid, Yvalid)

        # test
        Yhat = model.predict(Xtest)
        loss = cce(Ytest, Yhat).numpy()
        cfg['results'].append(float(loss))
        print("[%s] Loss: %f" % (cfg['comb'],loss))

# summarize model performance
for cfg in cfgs:
    cfg['mean_loss'] = np.mean(cfg['results'])

best_cfg = min(cfgs, key=lambda c: c['mean_loss'])
print("Best:")
print(json.dumps(best_cfg))

with open('hyperparam_opt_results.json', 'w') as f:
   json.dump(cfgs, f, indent=4)
