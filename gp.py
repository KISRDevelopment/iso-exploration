#
# Co-regionalized GP Model
#
import GPy
import numpy as np 
import pandas as pd 

def zscore(X):

    mu = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, ddof=1, keepdims=True)
    
    return zscore_mu_std(X, mu, std)

def zscore_mu_std(X, mu, std):
    Z = (X - mu) / (1e-6+std) 

    return Z, mu, std 
class GPModel:

    def __init__(self, cfg):

        self._cfg = cfg 
        self._currX = None 

    def train(self, Xtrain, Ytrain, Xvalid=None, Yvalid=None):

        n_inputs = Xtrain.shape[1]
        n_outputs = Ytrain.shape[1]

        # normalize inputs and outputs for better convergence
        Xtrain, self._Xtrain_mu, self._Xtrain_std = zscore(Xtrain)
        Ytrain, self._Ytrain_mu, self._Ytrain_std = zscore(Ytrain)
        

        # setup the Semi-Parameterized Latent Factor Model 
        slfm_kernel = GPy.kern.RBF(n_inputs).prod(GPy.kern.Coregionalize(1, n_outputs, active_dims=[n_inputs], rank=1, name='B'), name='k0')
        slfm_kernel.B.kappa.constrain_fixed(1)
        for i in range(1,n_outputs):
            k = GPy.kern.RBF(n_inputs).prod(GPy.kern.Coregionalize(1, n_outputs, 
                    active_dims=[n_inputs], rank=1, name='B'), name='k%d' % i)
            k.B.kappa.constrain_fixed(1)
            slfm_kernel = slfm_kernel + k

        m = GPy.models.GPCoregionalizedRegression([Xtrain for i in range(n_outputs)], 
                                                  [Ytrain[:,[i]] for i in range(n_outputs)],
                                                  kernel=slfm_kernel)
            
        m.optimize()

        self._m = m 
    
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        
    def predict(self, X, n_samples=1):
        #X = X[:10,:]
        if self._currX is X:
            newX = self._newX
            noise_dict = self._noise_dict 
        else:
            Xnormed,_,_ = zscore_mu_std(X, self._Xtrain_mu, self._Xtrain_std)
            ones = np.ones(X.shape[0])[:,np.newaxis]
            newX = np.vstack([np.hstack([Xnormed,ones*i]) for i in range(self._n_outputs)])
            noise_dict = {'output_index': newX[:,[-1]].astype(int)}
            self._currX = X 
            self._newX = newX
            self._noise_dict = noise_dict
        
        print(self._newX.shape)
        print(noise_dict['output_index'].shape)

        samples = self._m.posterior_samples(newX, Y_metadata=noise_dict, size=n_samples)

        samples = np.squeeze(samples, axis=1) * self._Ytrain_std + self._Ytrain_mu

        samples = np.reshape(samples, (self._n_outputs, X.shape[0], n_samples))
        samples = np.transpose(samples, (2, 1, 0))

        return samples 
