import tensorflow as tf 
import tensorflow.keras as keras 
import numpy as np 
import numpy.random as rng 
import sklearn.metrics 
import pandas as pd 
class RbfModel:

    def __init__(self, cfg):
        self._cfg = cfg 
                
    def train(self, Xtrain, Ytrain, Xvalid=None, Yvalid=None):

        D = sklearn.metrics.pairwise_distances(Xtrain, Xvalid).T
        
        Dsq = np.power(D, 2)

        gammas = np.linspace(1e-2, 3, 100)
        
        maes = []
        for g in gammas:

            # (n_valid, n_train)
            r = np.exp(-g * Dsq) 

            # (n_valid)
            rnormed = np.sum(r, axis=1)

            # (n_valid, n_output)
            yhat = np.dot(r / rnormed[:,np.newaxis], Ytrain) 

            ae = np.abs(yhat - Yvalid)

            mae = np.mean(ae)

            #print("%f: %f" % (g, mae))
            maes.append(mae)
        
        best_ix = np.argmin(maes)

        best_gamma = gammas[best_ix]
        print("Best %f: %f" % (best_gamma, maes[best_ix]))

        self._gamma = best_gamma
        self._X = np.vstack((Xtrain, Xvalid))
        self._Y = np.vstack((Ytrain, Yvalid))
    
    def train(self, Xtrain, Ytrain, Xvalid=None, Yvalid=None):

        D = sklearn.metrics.pairwise_distances(Xtrain, Xvalid).T
        
        Dsq = np.power(D, 2)

        gammas = np.linspace(1e-2, 3, 100)
        
        best_gammas = []
        for j in range(Ytrain.shape[1]):
            maes = []
            for g in gammas:

                # (n_valid, n_train)
                r = np.exp(-g * Dsq) 

                # (n_valid)
                rnormed = np.sum(r, axis=1)

                # (n_valid,)
                yhat = np.dot(r / rnormed[:,np.newaxis], Ytrain[:,j]) 
                
                ae = np.abs(yhat - Yvalid[:,j])

                mae = np.mean(ae)

                maes.append(mae)
            
            best_ix = np.argmin(maes)

            best_gamma = gammas[best_ix]
            print("Best for %d, %f: %f" % (j, best_gamma, maes[best_ix]))
            best_gammas.append(best_gamma)
        
        best_gammas = np.array(best_gammas)

        self._gamma = best_gammas
        self._X = np.vstack((Xtrain, Xvalid))
        self._Y = np.vstack((Ytrain, Yvalid))

    def predict(self, X):
        
        D = sklearn.metrics.pairwise_distances(self._X, X).T
        
        Dsq = np.power(D, 2)

        yhats = []
        for j in range(self._gamma.shape[0]):
            gamma = self._gamma[j]

            # (n_valid, n_train)
            r = np.exp(-gamma * Dsq) 

            # (n_valid)
            rnormed = np.sum(r, axis=1)

            # (n_valid, )
            yhat = np.dot(r / rnormed[:,np.newaxis], self._Y[:,j]) 

            yhats.append(yhat)

        yhats = np.array(yhats).T
        
        return yhats

    def evaluate(self, X, Y):

        yhat = self.predict(X)

        ae = np.abs(yhat - Y)

        return np.mean(ae)

# output_cols = [
#         "process_Hydrogen",
#         "process_Methane",
#         "process_Ethane",
#         "process_Propane",
#         "process_i-Butane",
#         "process_n-Butane",
#         "process_i-Pentane",
#         "process_n-Pentane",
#         "process_Cyclopentane",
#         "process_22-Mbutane",
#         "process_23-Mbutane",
#         "process_2-Mpentane",
#         "process_3-Mpentane",
#         "process_n-Hexane",
#         "process_Mcyclopentan",
#         "process_Benzene",
#         "process_Cyclohexane",
#         "process_2-Mhexane",
#         "process_n-Heptane"
# ]

# train_df = pd.read_csv('iso_train.csv')
# X = np.array(train_df[['r1_temp', 'r2_temp', 'r1_pressure', 'r2_pressure']])
# Y = np.array(train_df[output_cols])

# valid_ix = train_df['is_valid'] == 1
# train_ix = train_df['is_valid'] == 0

# Xtrain = X[train_ix,:]
# Ytrain = Y[train_ix,:]
# Xvalid = X[valid_ix,:]
# Yvalid = Y[valid_ix,:]

# train_df = pd.read_csv('iso_test.csv')
# Xtest = np.array(train_df[['r1_temp', 'r2_temp', 'r1_pressure', 'r2_pressure']])
# Ytest = np.array(train_df[output_cols])


# model = RbfModel({})
# model.train(Xtrain, Ytrain, Xvalid, Yvalid)
#model.predict(Xtest)
# mae = model.evaluate(Xtest, Ytest)
# print(mae)