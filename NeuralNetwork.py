"""Neural Network

Ez a modul a Scikit Learn féle Neurális Hálozatot fogja impllementálni.

Lehet, hogy később kiegészítem, hogy a NeuralNetwork osztály absztrakt
osztály legyen és a két féle megoldás (Scikit és Keras) lesz majd
leszármaztatva ebből az absztrakt osztályból.

This script requires that `numpy` and `sklearn` be installed within the
Python environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * __init__ - initialize the object with x_train and y_train from parameters
    * init_nn - sets the mlp with parameter _first and _second
    * create_prediction - return wiht the array of the predicted value(s)
"""

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)                             # <-- kell a Convergencia Warning miatt
simplefilter("ignore")                                                          # <-- a batch_size > n miatt kell ide

import numpy as np
from sklearn.neural_network import MLPRegressor

# ------------------------------------------------------------------------------

class NN():
    """
    Class NN bla bla.
    
    Attributes
    ----------
    mlp : sklearn.neural_network._multilayer_perceptron.MLPRegressor
        the number of rows that is read from the csv
    x_train : np.ndarray
        Need for this only because when we initialize the network the shape of the
        input is mandatory and shold be set properly. In order to be this sync
        with the input data, the best if we pass the x_train np.ndarray as
        an input (as a parameter of the consturctor).
    y_train : int
        Need for this only because when we initialize the network the shape of the
        output is mandatory and should be set properly. In order to be this sync
        with the output data, the best if we pas the y_train np.ndarray as
        an input (as a parameter of the consturctor).
    prediction : numpy.ndarray
        Initialy this variable is None.
        It is created after the create_prediction() method has been called.
        The shape of this np.ndarray is equal wiht the y_train.shape.
        

    Methods
    -------
    __init__(self, x_train: np.ndarray, y_train: np.ndarray)
        Consturctor of the object.
        Initialize the most important attributes via the given parameters.
    init_nn(self, _first: int, _second: int)
        This method must be called in order to initialize the weight of the neural net.
        The Scikit MLPRegressor has been created with empty weight.
        In order to create the weigths the MLPRegressor.fit() method must be invoked.
        This mehtod is responsible for creating the MLPRegressor and initialize
        the weigths via call the MLPRegressor.fit() method.
        The initialized MLPRegressor then is registered as an attribute of the object.
    create_prediction(self)
        This method is not mandatory but if we want to analyze or present the the
        prediction of the MLPRegressor than this method returns with the predicted
        array of values.
        It also assaignes the prediction to the self.prediction attribute.
    """
    
    def __init__(self, x_train, y_train):
        self.mlp = None
        self.x_train = x_train
        self.y_train = y_train
        self.prediction = None

# ------------------------------------------------------------------------------
  
    def init_nn(self, _first = 15, _second = 5, _numpy_random_seed = 1):
        '''
        Init Scikit Learn MLPRegressor
        
        This method must be called to initialize the mlp attirbute of
        the class and its coefs_ and intercept_ attributes.
        
        Parameters
        ----------
        _first : int, optional (default is 15)
            The number of neurons on the first layer
        _second : int, optional (default is 5)
            The number of neurons on the second layer
        _numpy_random_seed : int, optional (default is 1)
            The random seed number for numpy.random.seed(n)
        
        Returns
        -------
        sklearn.neural_network._multilayer_perceptron.MLPRegressor
            The MLPRegressor with the initialized coefs_ and intercept_
         
         Note
         ----
         Do not mess arround with the other parameters of the MLPRegressor
         as we will not 'train' the model and adjust the weight of the
         MLPRegressor at all.
        
        '''
        
        # <-- hogy létre jöjjenek a súlyok inicializálni kell
    
        np.random.seed(_numpy_random_seed)

        mlp = MLPRegressor(hidden_layer_sizes=(_first, _second),
                          activation='tanh',                                # -------> ha (MinMax(-1,1) vagy StandardScaler())
                          solver='sgd',
                          batch_size=100000,                                # <-- legyen nagy, nehogy végig iteráljon
                          max_iter=1,                                       # <-- sajnos legalább 1 kell hogy legyen
                          shuffle=False,
                          random_state=1,
                          learning_rate_init=0.00000001,                    # >- lehetőleg ne tanuljon semmit GD alapján
                          validation_fraction=0.0,
                          n_iter_no_change=99999999)
    
        # ----->                                         Behoztam ide az első illesztést is, hogy meglegyenek neki a súlyok

        np.random.seed(_numpy_random_seed)

        y_random = np.zeros((self.y_train.shape[0])) * 0.01                 # --> tök random adaton tanítom, hogy ne tanuljon

        mlp.fit(self.x_train, y_random)                                     # --> nem akarjuk tanítani csak kell az inithez

        self.mlp = mlp                                                      # --> settelje be a self.mlp
        
        print('Az mlp objectum típusa ebben az esetben: ', type(mlp))       # -- csak ellenőrzés képpen hogy mi a típusa

        return mlp                                                          # --> TODO: nem kell ez ide, ne használja több helyen

# ------------------------------------------------------------------------------

    def create_prediction(self) -> np.ndarray:
        '''
        Create the object's self.prediction attributes.
        
        This method must be called.
        
        Otherwise the self.predicted field stays empty or None.
        
        Returns
        -------
            prediction: np.ndarray: self.mlp.predict(self.x_train)
        '''

        self.prediction = self.mlp.predict(self.x_train)                         
        # --> set the self.prediction field with the predicted values

        return self.prediction


# ------------------------------------------------------------------------------    # <-- NN Class

# ------------------------------------------------------------------------------    # <-- NN Class

# Neural Network Keras Class

import os
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation

# ------------------------------------------------------------------------------

class KerasMLP():
    """
    Class NN bla bla.
    
    Attributes
    ----------
    mlp : sklearn.neural_network._multilayer_perceptron.MLPRegressor
        the number of rows that is read from the csv
    """

    def __init__(self, x_train, y_train, gpu = False):
        self.mlp = None
        self.x_train = x_train
        self.y_train = y_train
        self.prediction = None
        tf.random.set_seed(1)
        if ( gpu == False ):
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'                # <-- Disable GPU, Run with CPU
        if ( gpu == True ):
            os.environ['CUDA_VISIBLE_DEVICES'] = '1'                 # <-- Enable GPU, Run with GPU

# ------------------------------------------------------------------------------
  
    def init_nn(self, _first = 15, _second = 5):
        'Init Keras MLP'                                             # <-- hogy létre jöjjenek a súlyok inicializálni kell
    
        np.random.seed(1)

        # <--------------------------------- ez az x_train.shape-ből kiolvasható
        _input_shape = 2                                                # <----------------------jaj ennek a window-ból tudni kell az értékét
        _input_shape = self.x_train.shape[1]                            # <----------------------ezzel lesznek gondok, ha majd az indiket is hozzáadom

        mlp = Sequential()
        mlp.add(Dense(_first, input_shape=(_input_shape, ), activation='tanh'))
        mlp.add(Dense(_second, activation='tanh'))
        mlp.add(Dense(1))
    
        # ----->                                           Behoztam ide az első illesztést is, hogy meglegyenek neki a súlyok

        np.random.seed(1)

        y_random = np.zeros((self.y_train.shape[0])) * 0.01                 # --> tök random tanítom, hogy ne tanuljon


        mlp.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.0000001),
            # optimizer=tf.keras.optimizers.Adam(learning_rate=0.0000001),  # <---- Az Adam legyen SGD ugyse használom
            loss='mean_absolute_error')

        history = mlp.fit(self.x_train,
                          y_random,                # --> y_random és nem self.y_train, hogy véletlen adatokra illeszen
                          epochs=1,
                          shuffle=False,
                          verbose=0,
                          validation_split = 0)

        self.mlp = mlp                                                      # --> settelje be a self.mlp
        
        print('Az mlp objectum típusa ebben az esetben: ', type(mlp))       # -- csak ellenőrzés képpen hogy mi a típusa
    
        return mlp

# ------------------------------------------------------------------------------

    def create_prediction(self):
        'Saját adatai alapján csinája meg a predcitiont'

        self.prediction = self.mlp.predict(self.x_train).flatten()

        return self.prediction

















