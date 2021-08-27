import numpy as np
import pandas as pd
import nn
import numpy.random as rng
import json
import sys 
from sklearn.model_selection import KFold
import itertools 
import tensorflow as tf 
import feature_reader

path = sys.argv[1]
output_path = sys.argv[2]

P_VALID = 0.2

with open(path, 'r') as f:
    cfg = json.load(f)

# read dataset
df = pd.read_csv(cfg['dataset'])
loader_func = getattr(feature_reader, cfg['model']['features'])
X = loader_func(df)
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
kf = KFold(n_splits=cfg['folds'], shuffle=True, random_state=51514)
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
        loss = model.evaluate(Xtest, Ytest)

        cfg['results'].append(float(loss))
        print("[%s] Loss: %f" % (cfg['comb'],loss))

best_cfg = min(cfgs, key=lambda c: np.mean(c['results']))
print("Best:")
print(json.dumps(best_cfg))

with open(output_path, 'w') as f:
   json.dump(cfgs, f, indent=4)
