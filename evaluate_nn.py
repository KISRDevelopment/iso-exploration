import numpy as np
import pandas as pd
import nn
import numpy.random as rng
import json
import sys 

path = sys.argv[1]

with open(path, 'r') as f:
    cfg = json.load(f)

train_df = pd.read_csv('iso_train.csv')
Xtrain = np.array(train_df[['r1_temp', 'r2_temp', 'r1_pressure', 'r2_pressure']])
Ytrain = np.array(train_df[cfg['output_cols']])
print("Train size: %d" % (Xtrain.shape[0]))

test_df = pd.read_csv('iso_test.csv')
Xtest = np.array(test_df[['r1_temp', 'r2_temp', 'r1_pressure', 'r2_pressure']])
Ytest = np.array(test_df[cfg['output_cols']])
print("Test size: %d" % (Xtest.shape[0]))

aes = []
for model_cfg in cfg['models']:

    model = nn.FeedforwardNN(model_cfg)

    model.train(Xtrain, Ytrain)

    loss = model.evaluate(Xtest, Ytest)
    
    print("%32s %8.4f" % (model_cfg['name'], loss))

    # (n_samplesx19)
    yhat = model.predict(Xtest)
    
    ae = np.abs(yhat - Ytest)

    aes.append(np.mean(ae, axis=0))
aes = np.array(aes)
print(np.array(aes))
print(aes[0,:] > aes[1,:])