# ------------------------------------------------------------------------------    # <-- Randomer Class

# Randomer Class

from numpy.random import RandomState
from copy import deepcopy
import numpy as np

class Randomer():

  def __init__(self, _seed):
    self.seed = _seed
    self.prng = RandomState(1234567890)
    self.debug = False
    self.counter = 0


  def randomize(self, coefs, factor = 1000):
    'Get an MLPRegressor, takes its .coefs_ and randomize'
    self.counter += 1
    _factor = factor
    # _coefs = coefs.copy()                               # <-- BUG?? NEM AZ!
    _coefs = deepcopy(coefs)
    for i in range(len(coefs)):

      modifier = (self.prng.randn(coefs[i].shape[0], coefs[i].shape[1]) / _factor)  # <-- create new random values N(0,1)/factor

      if (self.debug == True):
        print('# ------- c =', self.counter, '------------')
        print('# ------- i =', i, 'layer --------')
        print('------- MODIFIER -------')
        print(modifier)
        print('------- COEFS ---------')
        print(coefs[i])

      _coefs[i] = coefs[i] + modifier                                               # <-- add new ranodom values to the weights (all at once)

      if (self.debug == True):
        print('------- MOD COEFS -----')
        print(_coefs[i])

      # _coefs[i] = coefs[i]                                                        # <-- ha nem akarom módosítani akkor legye egyszerűen csak ez

    return _coefs


    