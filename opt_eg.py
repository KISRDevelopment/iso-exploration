#
# epsilon-greedy optimization with neural networks
#
import numpy as np 
import numpy.random as rng
import pandas as pd 
import json
import nn 

def main(cfg, output_path):
    
    min_ron = cfg['min_ron']
    min_yield = cfg['min_yield']
    n_init_samples = cfg['n_init_samples']
    n_samples = cfg['n_samples']
    n_reps = cfg['n_reps']

    # load data 
    df = pd.read_csv('iso.csv')
    df = df[df['r1_charge_heater'] >= 0]

    X = np.array(df[['r1_temp', 'r2_temp', 'r1_pressure', 'r2_pressure']])
    Y = np.array(df[['r1_charge_heater', 'process_ron', 'process_yield']])

    # prepare NN
    traces = []
    for r in range(n_reps):
        
        model = nn.FeedforwardNN(cfg)

        # pick random points to seed the model
        chosen_ix = rng.choice(X.shape[0], n_init_samples, replace=False).tolist()
        
        # keep track of available experiments to run
        avail_indecies = np.ones(X.shape[0]).astype(bool)
        avail_indecies[chosen_ix] = False 

        while len(chosen_ix) <= n_samples:
            
            # train model
            Xtrain = X[chosen_ix, :]
            Ytrain = Y[chosen_ix, :]

            feasible_ix = (Ytrain[:,1] >= min_ron) & (Ytrain[:,2] >= min_yield)
            best_feasible_ch = np.min(Ytrain[feasible_ix,0], axis=0) if np.sum(feasible_ix) > 0 else np.inf
            print("Number of experiments: %d, best feasible CH: %f, latest: %s" % (len(chosen_ix), best_feasible_ch, Ytrain[-1,:]))

            if cfg['epsilon'] < 1:
                model.train(Xtrain, Ytrain)
            
            if rng.binomial(1, 1-cfg['epsilon']):
                preds = model.predict(X)
                feasible_ix = (preds[:,1] >= min_ron) & (preds[:,2] >= min_yield)
                preds[chosen_ix, 0] = np.inf
                preds[~feasible_ix, 0] = np.inf 
                ix = rng.permutation(preds.shape[0])
                next_ix = min(ix, key=lambda i: preds[i,0])
            else:
                next_ix = rng.choice(X.shape[0])
            
            chosen_ix.append(next_ix)
            
        traces.append(chosen_ix)
    
    traces = pd.DataFrame(data=traces)
    traces.to_csv(output_path)

    return traces
if __name__ == "__main__":
    import sys 
    
    with open(sys.argv[1], 'r') as f:
        cfg = json.load(f)

    main(cfg, sys.argv[2])