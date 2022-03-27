#
# Co-regionalized GP Model
#
import GPy
import numpy as np 
import pandas as pd 

class GPModel:

    def __init__(self, cfg):

        self._cfg = cfg 
        self._currX = None 

    def train(self, Xtrain, Ytrain, Xvalid=None, Yvalid=None):

        n_inputs = Xtrain.shape[1]
        n_outputs = Ytrain.shape[1]

        K = GPy.util.multioutput.LCM(n_inputs, 
                                 n_outputs, 
                                 [GPy.kern.Matern52(n_inputs) for i in range(n_outputs)], 
                                 W_rank=1)

        m = GPy.models.GPCoregionalizedRegression([Xtrain for i in range(n_outputs)], 
                                                  [Ytrain[:,[i]] for i in range(n_outputs)],
                                                  kernel=K)
            
        m['.*B.*kappa'].constrain_fixed(1.)
        m.optimize()

        self._m = m 
    
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        
    def predict(self, X, n_samples=1):
        if self._currX is X:
            newX = self._newX
            noise_dict = self._noise_dict 
        else:
            ones = np.ones(X.shape[0])[:,np.newaxis]
            newX = np.vstack([np.hstack([X,ones*i]) for i in range(self._n_outputs)])
            noise_dict = {'output_index': newX[:,[-1]].astype(int)}
            self._currX = X 
            self._newX = newX
            self._noise_dict = noise_dict
        
        samples = self._m.posterior_samples(newX, Y_metadata=noise_dict, size=n_samples)
        samples = np.squeeze(samples, axis=1)

        samples = np.reshape(samples, (self._n_outputs, X.shape[0], n_samples))
        samples = np.transpose(samples, (2, 1, 0))

        return samples 
