#
# epsilon-greedy optimization with neural networks
#
import numpy as np 
import numpy.random as rng
import pandas as pd 
import json
import nn 

def main(exp_cfg, model_cfg, output_path):
    
    n_init_samples = exp_cfg['n_init_samples']
    n_samples = exp_cfg['n_samples']
    n_reps = exp_cfg['n_reps']
    
    # load data 
    df = pd.read_csv('iso.csv')
    df = df[df['r1_charge_heater'] >= 0]

    X = np.array(df[['r1_temp', 'r2_temp', 'r1_pressure', 'r2_pressure']])
    Y = np.array(df[['r1_charge_heater', 'process_ron', 'process_yield']])

    # prepare NN
    traces = []
    obs_traces = []
    for r in range(n_reps):
        
        model = nn.FeedforwardNN(model_cfg)

        # pick random points to seed the model
        chosen_ix = rng.choice(X.shape[0], n_init_samples, replace=False).tolist()
        observed_y = [sample(Y, exp_cfg, i) for i in chosen_ix]


        min_ron = exp_cfg['min_ron']
        min_yield = exp_cfg['min_yield']

        while len(chosen_ix) <= n_samples:
            
            # train model
            Xtrain = X[chosen_ix, :]
            Ytrain = np.array(observed_y)

            feasible_ix = (Ytrain[:,1] >= min_ron) & (Ytrain[:,2] >= min_yield)
            best_feasible_ch = np.min(Ytrain[feasible_ix,0], axis=0) if np.sum(feasible_ix) > 0 else np.inf
            print("Number of experiments: %d, best feasible CH: %f, latest: %s" % (len(chosen_ix), best_feasible_ch, Ytrain[-1,:]))

            if model_cfg['epsilon'] < 1:
                model.train(Xtrain, Ytrain)
            
            # pick out a region of X that is eligible for exploration
            eligible_next_ix = determine_next_eligible_region(exp_cfg, Xtrain, X)

            if rng.binomial(1, 1-model_cfg['epsilon']):
                preds = model.predict(X, model_cfg['n_samples'])
                print(preds.shape)

                best_ch = np.min(Ytrain[:,0], axis=0)
                ch_less_than_best = preds[:,:,0] < best_ch 
                ron_more_than_min = preds[:,:,1] >= min_ron 
                yield_more_than_min = preds[:,:,2] >= min_yield 

                # criteria: maximize P(ch < best_ch, ron > min_ron, yield > min_yield)
                if model_cfg['criteria'] == 'pi':
                    
                    combined = ch_less_than_best * ron_more_than_min * yield_more_than_min
                    probs = np.mean(combined, axis=0)
                    next_ix = np.argmax(probs)
                    print("Max prob: ",  np.max(probs))
                    print("Next: %f,%f,%f,%f" % tuple(X[next_ix,:].tolist()))
                else:
                    # criteria: maximize E(I)
                    #I = best_ch - ch
                    #E(I) = best_ch - E(ch) under p(ch, ron > min_ron, yield > min_yield)
                    ron_yield_greater_than_min = ron_more_than_min * yield_more_than_min
                    valid_ch = preds[:,:,0] * ron_yield_greater_than_min
                    denom = np.sum(ron_yield_greater_than_min, axis=0)
                    print("Zero denom: %d" % np.sum(denom == 0))

                    e_ch = np.sum(valid_ch, axis=0) / denom 
                    e_i = best_ch - e_ch 
                    e_i = np.nan_to_num(e_i, nan=0)

                    next_ix = np.argmax(e_i)
                    print("Max EI: ",  np.max(e_i))
                    print("Next: %f,%f,%f,%f" % tuple(X[next_ix,:].tolist()))

            else:
                eligible_indecies = np.where(eligible_next_ix)[0]
                next_ix = rng.choice(eligible_indecies)
            
            chosen_ix.append(next_ix)
            
            # sample experiment
            observed_y.append(sample(Y, exp_cfg, next_ix))

        traces.append([int(i) for i in chosen_ix])
        obs_traces.append(observed_y)

    with open(output_path, 'w') as f:
        json.dump({
            "exp_cfg" : exp_cfg,
            "model_cfg" : model_cfg,
            "traces" : traces,
            "obs_traces" : obs_traces
        }, f, indent=4)
    
def determine_next_eligible_region(exp_cfg, Xtrain, X):

    max_delta = np.array(exp_cfg['max_selection_delta'])
    last_exp = Xtrain[-1,:]

    ix = np.all(np.abs(X - last_exp) <= max_delta, axis=1)
    print("Eligible next region: %d (Total %d)" % (np.sum(ix), X.shape[0]))
    return ix 

def sample(Y, cfg, ix):
    return [
        np.maximum(0, add_noise_rel_perc(Y[ix, 0], cfg['ch_rel_perc'])),
        np.clip(add_noise_within_delta(Y[ix, 1], cfg['ron_delta']), 0, 100),
        np.clip(add_noise_within_delta(Y[ix, 2], cfg['yield_delta']), 0, 100)
    ]

def add_noise_rel_perc(v, perc):
    if perc == 0:
        return v 
    
    delta = (perc * v) / 100.0
    return add_noise_within_delta(v, delta)

def add_noise_within_delta(v, delta):
    if delta == 0:
        return v 
    
    lower = v - delta
    upper = v + delta
    return rng.uniform(lower, upper, size=np.shape(v))


def add_noise(value, std):
    if std == 0:
        return value 
    return value + rng.normal(0, std)

if __name__ == "__main__":
    import sys 
    exp_cfg_path = sys.argv[1]
    model_cfg_path = sys.argv[2]
    output_path = sys.argv[3]

    with open(exp_cfg_path, 'r') as f:
        exp_cfg = json.load(f)

    with open(model_cfg_path, 'r') as f:
        model_cfg = json.load(f)

    main(exp_cfg, model_cfg, output_path)