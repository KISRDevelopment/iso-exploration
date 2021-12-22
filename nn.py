import tensorflow as tf 
import tensorflow.keras as keras 
import numpy as np 
import numpy.random as rng 

def zscore(X):

    mu = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, ddof=1, keepdims=True)
    Z = (X - mu) / std 

    return Z, mu, std 

class FeedforwardNN:

    def __init__(self, cfg):
        self._cfg = cfg 
                
    def train(self, Xtrain, Ytrain, Xvalid=None, Yvalid=None):
        self._cfg['n_input'] = Xtrain.shape[1]

        if self._cfg.get('constrained', False):
            self._make_model_constrained()
        else:
            self._make_model()
        #print(self._model.summary())
        
        # normalize inputs and outputs for better convergence
        Xtrain, _, _ = zscore(Xtrain)

        if Xvalid is not None:
            Xvalid,_,_ = zscore(Xvalid)

        Ytrain, self._Ytrain_mu, self._Ytrain_std = zscore(Ytrain)

        if Xvalid is not None:
            Yvalid = (Yvalid - self._Ytrain_mu) / self._Ytrain_std
            
        if self._cfg.get('scramble', False):
            Xtrain = Xtrain[rng.permutation(Xtrain.shape[0]), :]
        
        callbacks = []
        if Xvalid is not None:
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self._cfg['patience'], restore_best_weights=True)
            callbacks.append(callback)
        
        self._model.fit(Xtrain, 
                        Ytrain, 
                        batch_size=int(Xtrain.shape[0] * self._cfg['batch_size_p']),
                        validation_data=(Xvalid, Yvalid) if Xvalid is not None else (),
                        epochs=self._cfg['n_epochs'], 
                        verbose=self._cfg['verbose'],
                        callbacks=callbacks)
        
    def _make_model(self):
        cfg = self._cfg 
        
        keras.backend.clear_session()

        input_layer = keras.layers.Input(shape=(cfg['n_input'],))
        layer = input_layer
        layer = self._make_dropout(layer)

        if cfg['n_hidden'] > 0:
            layer = keras.layers.Dense(cfg['n_hidden'], activation=cfg['hidden_activation'])(layer)
            layer = self._make_dropout(layer)

        output_activation = 'linear'
        loss = keras.losses.MeanAbsoluteError()
        output_layer = keras.layers.Dense(cfg['n_output'], activation=output_activation)(layer)
        
        model = keras.Model(input_layer, output_layer)
        model.compile(
            loss=loss,
            optimizer=keras.optimizers.Nadam()
        )

        self._model = model 
    
    def _make_model_constrained(self):
        cfg = self._cfg 
        
        keras.backend.clear_session()

        input_layer = keras.layers.Input(shape=(cfg['n_input'],))
        layer = self._make_dropout(input_layer)
        preouts = []
        for i in range(cfg['n_output']):
            sub_nn = keras.layers.Dense(cfg['n_hidden'] // cfg['n_output'], activation=cfg['hidden_activation'])(layer)
            sub_nn = self._make_dropout(sub_nn)
            sub_nn = keras.layers.Dense(1, activation='linear')(sub_nn)
            preouts.append(sub_nn)
        output_layer = keras.layers.Concatenate()(preouts)
        
        model = keras.Model(input_layer, output_layer)
        model.compile(
            loss=keras.losses.MeanAbsoluteError(),
            optimizer=keras.optimizers.Nadam()
        )

        self._model = model 

    def _make_dropout(self, input_layer):
        if self._cfg['stochastic']:
            return keras.layers.Dropout(self._cfg['dropout_p'])(input_layer, training=True)
        else:
            return keras.layers.Dropout(self._cfg['dropout_p'])(input_layer)
        
    def predict(self, X, n_samples=1):
        
        X,_,_ = zscore(X)
        if not self._cfg['stochastic']:
            return self._predict(X)
        
        samples = []
        for i in range(n_samples):
            samples.append(self._predict(X))
        
        samples = np.array(samples)
        
        return samples

    def _predict(self, X):
        return self._model.predict(X) * self._Ytrain_std + self._Ytrain_mu

    def evaluate(self, X, Y):
        X,_,_ = zscore(X)
        Y = (Y - self._Ytrain_mu) / self._Ytrain_std
        return self._model.evaluate(X, Y)
