import numpy as np
import pandas as pd
import nn
import numpy.random as rng
import json
import sys 
from sklearn.model_selection import KFold
import itertools 

path = sys.argv[1]

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
    cfgs.append(new_cfg)
    
# evaluate CV performance of each combination
kf = KFold(n_splits=cfg['folds'])
for train_index, test_index in kf.split(X):

    for cfg in cfgs:
        
        model = nn.FeedforwardNN(cfg)

        # train
        Xtrain = X[train_index,:]
        Ytrain = Y[train_index,:]
        model.train(Xtrain, Ytrain)

        # test
        Xtest = X[test_index,:]
        Ytest = Y[test_index,:]
        loss = model.evaluate(Xtest, Ytest)

        cfg['results'].append(loss)
        print("Loss: %f" % loss)

# summarize model performance
for cfg in cfgs:
    cfg['mean_loss'] = np.mean(cfg['results'])

# write out
with open('tmp/hyperparam_opt_results.json', 'w') as f:
    json.dump(cfgs, f, indent=4)
