# ------------------------------------------------------------------------------    # <-- Randomer Class

# Segítség

# Korábban ezt már elég jól megírtam, itt taláható a kód
# https://github.com/JoDeMiro/DeepLearningIntroduction/blob/main/Fun_with_NeuralNet_Part_2.ipynb

# Randomer Class

from numpy.random import RandomState
from copy import deepcopy
import numpy as np
import os


class Randomer():

    def __init__(self, _seed):
        self.seed = _seed
        self.prng = RandomState(1234567890)
        self.debug = False
        self.counter = 0

    def randomize_intercepts(self, intercepts, factor=1000, select_ratio=1.0):
        'Get an MLPRegresson, takes its .intercepts_ and randomize'
        'select_ratio: float -> ha 1.0 akkor mindenki be van választva, ha 0.0 akkor senki (nincs mutáció)'
        _factor = factor
        _intercepts = deepcopy(intercepts)
        for i in range(len(intercepts)):

            modifier = (self.prng.randn(intercepts[i].shape[0]) / factor)
            selector = np.random.rand((modifier.shape[0]))       
            modifier[selector>select_ratio] = 0
            _intercepts[i] = intercepts[i] + modifier  # <-- add new random values to the intercepts (all at once)

        return _intercepts

    def randomize(self, coefs, factor=1000, select_ratio=1.0):
        'Get an MLPRegressor, takes its .coefs_ and randomize'
        self.counter += 1
        _factor = factor
        _coefs = deepcopy(coefs)
        for i in range(len(coefs)):

            modifier = (self.prng.randn(coefs[i].shape[0], coefs[i].shape[1]) /
                        _factor)  # <-- create new random values N(0,1)/factor

            if (self.debug == True):
                print('# ------- c =', self.counter, '------------')
                print('# ------- i =', i, 'layer --------')
                print('------- MODIFIER -------')
                print(modifier)
                print('------- COEFS ---------')
                print(coefs[i])

            # Régi Selector nélküli
            # _coefs[i] = coefs[i] + modifier  # <-- add new random values to the weights (all at once)
            
            # Új Selecor
            selector = np.random.rand(modifier.shape[0], modifier.shape[1])
            modifier[selector>select_ratio] = 0
            
            _coefs[i] = coefs[i] + modifier  # <-- add new random values to the weights (all at once)

            if (self.debug == True):
                print('------- MOD COEFS -----')
                print(_coefs[i])

            # _coefs[i] = coefs[i]                                                        # <-- ha nem akarom módosítani akkor legye egyszerűen csak ez

        return _coefs