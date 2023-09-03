# ------------------------------------------------------------------------------    # <-- NN Class

# Neural Network Class

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter(
    "ignore",
    category=ConvergenceWarning)      # <-- kell a Convergencia Warning miatt
simplefilter("ignore")                # <-- a batch_size > n miatt kell ide

import numpy as np
from sklearn.neural_network import MLPRegressor
from copy import deepcopy

# ------------------------------------------------------------------------------


class NN():

    def __init__(self, x_train, y_train):
        self.mlp = None
        self.x_train = x_train
        self.y_train = y_train
        self.random_seed = 1
        self.prediction = None
        self.sample_size = 10

# ------------------------------------------------------------------------------

    def init_nn(self, _first=15, _second=5, activation='tanh', random_seed=1):
        'Init Scikit Learn MLPRegressor'  # <-- hogy létre jöjjenek a súlyok inicializálni kell
        
        if random_seed != 1:
            self.random_seed = random_seed

        np.random.seed(self.random_seed)
        
        # Take a sample
        idx = np.random.randint(self.x_train.shape[0], size=self.sample_size)
        x_sample = self.x_train[idx,:]
        y_sample = self.y_train[idx,:]

        mlp = MLPRegressor(
            hidden_layer_sizes=(_first, _second),
            activation=activation,                     # -------> ha (MinMax(-1,1) vagy StandardScaler())
            # solver='sgd',
            solver='lbfgs',
            batch_size=987654321,         # <<-- v.017 bug fixed
            max_iter=1,                   # <-- sajnos legalább 1 kell hogy legyen
            shuffle=False,
            random_state=self.random_seed,
            learning_rate_init=
            0.0000000001,                 # >- lehetőleg ne tanuljon semmit GD alapján
            validation_fraction=0.0,
            max_fun=1,
            momentum=0.0,
            nesterovs_momentum=False,
            n_iter_no_change=1)  # ----->   Behoztam ide az első illesztést is, hogy meglegyenek neki a súlyok

        np.random.seed(self.random_seed)

#        y_random = np.zeros(
#            (self.y_train.shape[0])
#        ) * 0.0123                      # --> tök random adaton tanítom, hogy még véletlenül se tanuljon

        y_random = np.random.rand(y_sample.shape[0],)

        mlp.fit(
#            self.x_train, y_random
            x_sample, y_random
        )                               # --> nem akarjuk mi semmire megtanítani csak kell az inithez

        self.mlp = mlp
        
        print('_________________WAU___________________')

        return mlp

# ------------------------------------------------------------------------------

    def create_prediction(self):
        'Saját adati alapján csinája meg a predcitiont'

        self.prediction = self.mlp.predict(self.x_train)

        return self.prediction

# ------------------------------------------------------------------------------

    def mlp_reinitalizer(self, fac: float) -> MLPRegressor:

        mlp = deepcopy(self.mlp)

        for i, c in enumerate(mlp.coefs_):
            _ = np.random.rand(c.shape[0], c.shape[1]) * fac                   # uniform
            _ = np.random.normal(0, 1, size=(c.shape[0], c.shape[1])) * fac    # normal
            mlp.coefs_[i] = _

        for i, c in enumerate(mlp.intercepts_):
            if len(c.shape) == 2:
                _ = np.random.rand(c.shape[0], c.shape[1]) * fac
                _ = np.random.normal(0, 1, size=(c.shape[0], c.shape[1])) * fac
            if len(c.shape) == 1:
                _ = np.random.rand(c.shape[0]) * fac
                _ = np.random.normal(0, 1, size=(c.shape[0])) * fac

            mlp.intercepts_[i] = _
        
        self.mlp = deepcopy(mlp)        # felül csapom az initializált mlp-ta

        return mlp

# ------------------------------------------------------------------------------

    def knn_to_snn(self, kn):
        'keras model to scikit model'

        'természetesen csak regressziós és fc modellekre'

        _weights = []
        _biases  = []

        weights = kn.mlp.get_weights()
        for i in range(len(weights)):
            if i % 2 == 0:
                _w = weights[i].astype('float64')
                _weights.append(_w)
            if i % 2 == 1:
                _b = weights[i].astype('float64')
                _biases.append(_b)
        
        self.mlp.coefs_ = _weights
        self.mlp.intercepts_ = _biases

        return _weights, _biases

# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------    # <-- NN Class

# Neural Network Keras Class

from sklearn.neural_network import MLPRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

# ------------------------------------------------------------------------------


class KerasMLP():

    def __init__(self, x_train, y_train):
        self.mlp = None
        self.x_train = x_train
        self.y_train = y_train
        self.random_seed = 1
        self.prediction = None
        self.sample_size = 10
        tf.random.set_seed(1)

# ------------------------------------------------------------------------------

    def init_nn(self, _first=15, _second=5, activation='tanh', random_seed = 1):
        'Init Keras MLP'  # <-- hogy létre jöjjenek a súlyok inicializálni kell
        
        if random_seed != 1:
            self.random_seed = random_seed

        np.random.seed(self.random_seed)
        
        # Take a sample
        idx = np.random.randint(self.x_train.shape[0], size=self.sample_size)
        x_sample = self.x_train[idx,:]
        y_sample = self.y_train[idx,:]

        _input_shape = self.x_train.shape[1]

        mlp = Sequential()
        mlp.add(Dense(_first, input_shape=(_input_shape, ), activation=activation))
        mlp.add(Dense(_second, activation=activation))
        mlp.add(Dense(1))

        # ----->   Behoztam ide az első illesztést is, hogy meglegyenek neki a súlyok

        np.random.seed(self.random_seed)

        y_random = np.zeros(
            # (self.y_train.shape[0])
            (y_sample.shape[0])
        ) * 0.0123  # --> tök random adaton tanítom, hogy még véletlenül se tanuljon

        mlp.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.0000001),
            loss='mean_absolute_error')

        history = mlp.fit(
            # self.x_train,
            x_sample,
            y_random,  # --> y_random és nem self.y_train, hogy véletlen adatokra illeszen
            epochs=1,
            batch_size=y_random.shape[0],   # az összes adaton egyszerre
            shuffle=False,
            verbose=0,
            validation_split=0)

        self.mlp = mlp

        return mlp


# ------------------------------------------------------------------------------

    def create_prediction(self, batch_size):
        'Saját adatai alapján csinája meg a predcitiont'

        self.prediction = self.mlp.predict(self.x_train, batch_size=batch_size).flatten()
        
        return self.prediction

# ------------------------------------------------------------------------------
