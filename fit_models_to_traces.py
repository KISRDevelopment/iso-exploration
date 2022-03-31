import numpy as np
import nn 
import pandas as pd 
import sys 
import json

def main(path, trace_file, output_path):

    df = pd.read_csv(path)
    df = df[df['r1_charge_heater'] >= 0]

    X = np.array(df[['r1_temp', 'r2_temp', 'r1_pressure', 'r2_pressure']])
    Y = np.array(df[['r1_charge_heater', 'process_ron', 'process_yield']])

    with open(trace_file, 'r') as f:
        results = json.load(f)
    traces = np.array(results['traces'])
    
    #results['model_cfg'].update({ 'stochastic' : False })

    yhats = []
    for i in range(traces.shape[0]):
        Xtraced = X[traces[i, :],:]
        Ytraced = Y[traces[i,:],:]

        model = nn.FeedforwardNN(results['model_cfg'])
        model.train(Xtraced, Ytraced)
        yhat = model.predict(X, n_samples=100)
        yhats.append(yhat)
        print("Done %d" % i)
    
        np.savez(output_path, 
                    yhats=np.array(yhats), 
                    model_cfg=results['model_cfg'], 
                    exp_cfg=results['exp_cfg'])
        
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])

