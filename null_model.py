import tensorflow as tf 
import tensorflow.keras as keras 
import numpy as np 
import numpy.random as rng 

class NullModel:

    def __init__(self, cfg):
        self._cfg = cfg 
                
    def train(self, Xtrain, Ytrain, Xvalid=None, Yvalid=None):
        self._mean = np.mean(np.vstack((Ytrain, Yvalid)), axis=0)
        
    def predict(self, X):
        return np.tile(self._mean, (X.shape[0], 1))

    def evaluate(self, X, Y):
        pass 
