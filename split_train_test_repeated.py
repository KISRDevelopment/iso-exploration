#
# split the simulation dataset into a development and withheld dataset repeatedly
#
import numpy as np 
import pandas as pd 
from skopt.sampler import Lhs

def main(dataset_path, output_path, seed=896784):
        
    np.random.seed(seed)
    P_TRAIN = 0.25
    P_VALID = 0.2
    N_REPEATS = 10

    # read dataset
    df = pd.read_csv(dataset_path)
    df['id'] = np.arange(df.shape[0])
    parameters = list(zip(df['r1_temp'], df['r2_temp'], df['r1_pressure'], df['r2_pressure']))
    parameters_set = set(parameters)
    df.index = pd.MultiIndex.from_tuples(parameters)
    r1temps = list(set(df['r1_temp']))
    r2temps = list(set(df['r2_temp']))
    r1pressures = list(set(df['r1_pressure']))
    r2pressures = list(set(df['r2_pressure']))

    all_splits = []
    for r in range(N_REPEATS):

        # sample randomly using Latin hypercube sampling
        n_train = int(P_TRAIN * (1-P_VALID) * df.shape[0])
        n_valid = int(P_TRAIN * P_VALID * df.shape[0])
        
        lhs = Lhs(lhs_type="classic", criterion=None)

        sampled_parameters_set_train = set()
        while len(sampled_parameters_set_train) < n_train:
            to_sample = n_train - len(sampled_parameters_set_train)
            s = set([tuple(p) for p in lhs.generate([r1temps, r2temps, r1pressures, r2pressures], to_sample)])
            eligible_s = s & parameters_set
            sampled_parameters_set_train = sampled_parameters_set_train | eligible_s
        
        sampled_parameters_set_valid = set()
        while len(sampled_parameters_set_valid) < n_valid:
            to_sample = n_valid - len(sampled_parameters_set_valid)
            s = set([tuple(p) for p in lhs.generate([r1temps, r2temps, r1pressures, r2pressures], to_sample)])
            eligible_s = (s - sampled_parameters_set_train) & parameters_set
            sampled_parameters_set_valid = sampled_parameters_set_valid | eligible_s
        
        sampled_parameters_train = list(sampled_parameters_set_train)
        sampled_parameters_valid = list(sampled_parameters_set_valid)

        train_index = df.loc[sampled_parameters_train]['id']
        valid_index = df.loc[sampled_parameters_valid]['id']
        test_index = df.loc[list(parameters_set - sampled_parameters_set_train - sampled_parameters_set_valid)]['id']
        
        split = np.zeros(df.shape[0])
        split[train_index] = 0
        split[valid_index] = 1
        split[test_index] = 2
        
        
        print("Traing size: %d (validation: %d), test size: %d, (Sum: %d, All: %d)" % (np.sum(split == 0), np.sum(split == 1), np.sum(split == 2), 
            np.sum(split == 0) + np.sum(split == 1) + np.sum(split == 2),
            split.shape[0]))

        assert np.sum(train_index & valid_index) == 0 
        assert np.sum(train_index & test_index) == 0
        assert np.sum(valid_index & test_index) == 0

        #counts = np.array([np.sum(split == s) for s in [0,1,2]])
        #print(counts/np.sum(counts))

        all_splits.append(split)

    all_splits = np.array(all_splits)
    unique_rows = np.unique(all_splits, axis=0)
    assert unique_rows.shape[0] == N_REPEATS

    np.save(output_path, all_splits)

if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
