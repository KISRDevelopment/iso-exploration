import numpy as np
import pandas as pd
import nn
import numpy.random as rng
import json
import sys 
import tensorflow as tf
import rm 

def main(path, split_path, output_path):

    with open(path, 'r') as f:
        cfg = json.load(f)

    df = pd.read_csv('iso.csv')
    X = np.array(df[['r1_temp', 'r2_temp', 'r1_pressure', 'r2_pressure']])
    Y = np.array(df[cfg['output_cols']])

    splits = np.load(split_path)

    split_mae = []
    split_mae_by_output = []
    for s in range(splits.shape[0]):
        split = splits[s,:]

        train_ix = split == 0
        valid_ix = split == 1
        test_ix = split == 2

        Xtrain = X[train_ix,:]
        Ytrain = Y[train_ix,:]
        Xvalid = X[valid_ix,:]
        Yvalid = Y[valid_ix,:]
        Xtest = X[test_ix,:]
        Ytest = Y[test_ix,:]

        print("Train size: %d, Validation: %d, Test: %d" % (Xtrain.shape[0], Xvalid.shape[0], Xtest.shape[0]))

        if cfg['model']['module'] == 'nn':
            model = nn.FeedforwardNN(cfg['model'])
        elif cfg['model']['module'] == 'rm':
            model = rm.RegressionModel(cfg['model'])
        
        model.train(Xtrain, Ytrain, Xvalid, Yvalid)
        yhat = model.predict(Xtest)
        ae = np.abs(yhat - Ytest)

        mae = np.mean(ae)
        mae_by_output = np.mean(ae, axis=0)

        split_mae.append(mae)
        split_mae_by_output.append(mae_by_output)

    split_mae = np.array(split_mae)
    split_mae_by_output = np.array(split_mae_by_output)

    np.savez(output_path, cfg=cfg, split_mae=split_mae, split_mae_by_output=split_mae_by_output)

if __name__ == "__main__":
    path = sys.argv[1]
    split_path = sys.argv[2]
    output_path = sys.argv[3]
    main(path, split_path, output_path)

