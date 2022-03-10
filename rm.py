import tensorflow as tf 
import tensorflow.keras as keras 
import numpy as np 
import numpy.random as rng 
import itertools
def zscore(X):

    mu = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, ddof=1, keepdims=True)
    
    return zscore_mu_std(X, mu, std)

def zscore_mu_std(X, mu, std):
    Z = (X - mu) / (1e-6+std) 

    return Z, mu, std

class RegressionModel:

    def __init__(self, cfg):
        self._cfg = cfg 
                
    def train(self, Xtrain, Ytrain, Xvalid=None, Yvalid=None):
        self._cfg['n_input'] = Xtrain.shape[1]
        self._cfg['n_output'] = Ytrain.shape[1]
        
        # normalize inputs and outputs for better convergence
        Xtrain, self._Xtrain_mu, self._Xtrain_std = zscore(Xtrain)
        if Xvalid is not None:
            Xvalid,_,_ = zscore_mu_std(Xvalid, self._Xtrain_mu, self._Xtrain_std)

        Ytrain, self._Ytrain_mu, self._Ytrain_std = zscore(Ytrain)
        if Xvalid is not None:
            Yvalid,_,_ = zscore_mu_std(Yvalid, self._Ytrain_mu, self._Ytrain_std)

        callbacks = []
        if Xvalid is not None:
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self._cfg['patience'], restore_best_weights=True)
            callbacks.append(callback)
        
        # transform inputs
        Xtrain_inputs = self._transform(Xtrain)
        if Xvalid is not None:
            Xvalid_inputs = self._transform(Xvalid)

        self._make_model() 

        self._model.fit(Xtrain_inputs, 
                        Ytrain, 
                        batch_size=int(Xtrain.shape[0] * self._cfg['batch_size_p']),
                        validation_data=(Xvalid_inputs, Yvalid) if Xvalid is not None else (),
                        epochs=self._cfg['n_epochs'], 
                        verbose=self._cfg['verbose'],
                        callbacks=callbacks)
        
    def _transform(self, X):

        # transform each col
        transformed_inputs = []
        for i in range(X.shape[1]):
            col = X[:,[i]]
            tcols = []
            for p in range(0, self._cfg['degree']):
                tcol = np.power(col, p+1)
                tcols.append(tcol)
            
            #[batchxd]
            transformed_input = np.hstack(tcols)
            print(transformed_input.shape)
            transformed_inputs.append(transformed_input)
        
        return transformed_inputs

    def _make_model(self):
        cfg = self._cfg 
        
        keras.backend.clear_session()

        inputs = []
        transformation_layers = []
        for i in range(self._cfg['n_input']):
            input_layer = keras.layers.Input(shape=(cfg['degree'],))
            if self._cfg['degree'] == 1:
                trans_layer = input_layer
            else:
                trans_layer = keras.layers.Dense(1, activation='linear')(input_layer)
            #trans_layer = keras.layers.Dense(cfg['n_hidden'], activation='tanh')(trans_layer)
            #trans_layer = keras.layers.Dense(1, activation='linear')(trans_layer)

            transformation_layers.append(trans_layer)
            inputs.append(input_layer)

        # [Batch Size x 4]
        transformed_inputs = keras.layers.Concatenate()(transformation_layers)

        # create the permutation matrix to generate interaction terms
        if cfg['effects'] == 'main_only':
            P = create_main_effects_permutation_matrix(cfg['n_input'])
        elif cfg['effects'] == 'all':
            P = create_full_permutation_matrix(cfg['n_input'])
        
        def transform(X):
            # [batch x n_input x 2^n_input]
            tiledX = tf.tile(X[:,:,None], [1,1,P.shape[1]])
            r = tf.reduce_prod(tiledX * P + (1-P), axis=1) # computes the interactions
            return r 
        
        interaction_layer = keras.layers.Lambda(transform)(transformed_inputs)

        # just for debugging
        #interaction_layer = keras.layers.Dense(cfg['n_hidden'], activation='tanh')(transformed_inputs)

        output_layer = keras.layers.Dense(cfg['n_output'], activation='linear')(interaction_layer)
        
        model = keras.Model(inputs, output_layer)
        
        loss = keras.losses.MeanAbsoluteError()
        model.compile(
            loss=loss,
            optimizer=keras.optimizers.Nadam()
        )
        print(model.summary())
        
        self._model = model 
    
    def predict(self, X, n_samples=1):
        
        X,_,_ = zscore_mu_std(X, self._Xtrain_mu, self._Xtrain_std)

        Xtest_inputs = self._transform(X)
        return self._model.predict(Xtest_inputs, batch_size=100000) * self._Ytrain_std + self._Ytrain_mu
        

def create_full_permutation_matrix(n_inputs):

    P = list(itertools.product([0, 1], repeat=n_inputs))
    return tf.convert_to_tensor(np.array(P).T, dtype="float32")

def create_main_effects_permutation_matrix(n_inputs):

    return tf.eye(n_inputs, dtype="float32")    

# P = create_full_permutation_matrix(4)
# print(P)


# import pandas as pd
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

# rm = RegressionModel({ 'patience' : 50, 'degree' : 2, 'batch_size_p' : 1, 'n_epochs' : 10000, 'verbose' : True })
# rm.train(Xtrain, Ytrain, Xvalid, Yvalid)

# test_df = pd.read_csv('iso_test.csv')
# Xtest = np.array(test_df[['r1_temp', 'r2_temp', 'r1_pressure', 'r2_pressure']])
# Ytest = np.array(test_df[output_cols])
# print("Test size: %d" % (Xtest.shape[0]))

# yhat = rm.predict(Xtest)
# ae = np.abs(yhat - Ytest)
# mae_by_output = [float(e) for e in np.mean(ae, axis=0)]
# print(mae_by_output)