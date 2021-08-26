#
# split the simulation dataset into a development and withheld dataset
#
import numpy as np 
import pandas as pd 
from skopt.sampler import Lhs

np.random.seed(542345)
P_TRAIN = 0.25

# read dataset
df = pd.read_csv("iso.csv")
parameters = list(zip(df['r1_temp'], df['r2_temp'], df['r1_pressure'], df['r2_pressure']))
parameters_set = set(parameters)
df.index = pd.MultiIndex.from_tuples(parameters)
r1temps = list(set(df['r1_temp']))
r2temps = list(set(df['r2_temp']))
r1pressures = list(set(df['r1_pressure']))
r2pressures = list(set(df['r2_pressure']))

# sample randomly using Latin hypercube sampling
n_train = int(P_TRAIN * df.shape[0])
lhs = Lhs(lhs_type="classic", criterion=None)
sampled_parameters_set = set()
while len(sampled_parameters_set) < n_train:
    to_sample = n_train - len(sampled_parameters_set)
    s = set([tuple(p) for p in lhs.generate([r1temps, r2temps, r1pressures, r2pressures], to_sample)])
    eligible_s = s & parameters_set
    sampled_parameters_set = sampled_parameters_set | eligible_s
sampled_parameters = list(sampled_parameters_set)

# build training and testing dfs
train_df = df.loc[sampled_parameters]
test_df = df.loc[list(parameters_set - sampled_parameters_set)]

print("Traing size: %d, test size: %d" % (train_df.shape[0], test_df.shape[0]))

train_df.to_csv("iso_train.csv", index=False)
test_df.to_csv("iso_test.csv", index=False)
