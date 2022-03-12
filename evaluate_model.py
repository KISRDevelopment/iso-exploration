import numpy as np
import pandas as pd
import nn
import numpy.random as rng
import json
import sys 
import tensorflow as tf
import rm 
import itertools 
from sklearn.model_selection import KFold
import null_model
import rbfm
from multiprocessing import Pool 

def main(path, split_path, output_path, n_processes=10):

    with open(path, 'r') as f:
        cfg = json.load(f)

    df = pd.read_csv('iso.csv')
    X = np.array(df[['r1_temp', 'r2_temp', 'r1_pressure', 'r2_pressure']])
    Y = np.array(df[cfg['output_cols']])

    splits = np.load(split_path)

    with Pool(n_processes) as p:
        results = p.map(eval_model, [(cfg, X, Y, splits, s) for s in range(splits.shape[0])])

    split_ae = np.array([r[1] for r in results])
    
    model_cfgs = [r[0] for r in results]

    np.savez(output_path, cfg=cfg, model_cfgs=model_cfgs, split_ae=split_ae)

def eval_model(packed):
    cfg, X, Y, splits, s = packed 

    split = splits[s,:]

    train_ix = split == 0
    valid_ix = split == 1
    test_ix = split == 2

    # hyperparameter optimization on the training portion
    combined_ix = train_ix | valid_ix
    best_model_cfg = hyperparam_opt(cfg, X[combined_ix,:],Y[combined_ix,:])

    Xtrain = X[train_ix,:]
    Ytrain = Y[train_ix,:]
    Xvalid = X[valid_ix,:]
    Yvalid = Y[valid_ix,:]
    Xtest = X[test_ix,:]
    Ytest = Y[test_ix,:]

    if cfg['model']['module'] == 'nn':
        model = nn.FeedforwardNN(best_model_cfg)
    elif cfg['model']['module'] == 'rm':
        model = rm.RegressionModel(best_model_cfg)
    elif cfg['model']['module'] == 'null':
        model = null_model.NullModel(best_model_cfg)
    elif cfg['model']['module'] == 'rbfm':
        model = rbfm.RbfModel(best_model_cfg)
    
    model.train(Xtrain, Ytrain, Xvalid, Yvalid)
    yhat = model.predict(Xtest)
    ae = np.abs(yhat - Ytest)

    return (best_model_cfg, ae)

def hyperparam_opt(cfg, X, Y):
    P_VALID = 0.2 

    cfgs = get_cfg_combinations(cfg)
    if len(cfgs) == 1:
        return cfgs[0]
    
    module = cfg['model']['module']
    kf = KFold(n_splits=cfg['folds'], shuffle=True)

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
        
            if module == 'nn':
                model = nn.FeedforwardNN(cfg)
            elif module == 'rm':
                model = rm.RegressionModel(cfg)

            # train
            model.train(Xtrain, Ytrain, Xvalid, Yvalid)

            # test
            loss = model.evaluate(Xtest, Ytest)

            cfg['results'].append(float(loss))
            print("[%s] Loss: %f" % (cfg['comb'],loss))

    best_cfg = min(cfgs, key=lambda c: np.mean(c['results']))
    #print(best_cfg['comb'])
    print("Best: %s" % (best_cfg['comb'],))

    return best_cfg

def get_cfg_combinations(cfg):

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

    return cfgs

if __name__ == "__main__":
    path = sys.argv[1]
    split_path = sys.argv[2]
    output_path = sys.argv[3]
    main(path, split_path, output_path)

