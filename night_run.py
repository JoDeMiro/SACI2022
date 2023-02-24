# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------
Copyright (C) 2023 SZTAKI (Pintye István), Hungary.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
------------------------------------------------------------------------------
 "main.py" - Main file for training fully-connected and convolutional networks using evolution selection (ES).
    Example: use the following command to train Agents on the test set of EURUSD using ES:
         python main.py --dataset EURUSD --train-mode ES --generation 200 --freeze-conv-layers
                        --dropout 0.05 --topology CONV_64_3_1_1_CONV_256_3_1_1_FC_2000_FC_10
                        --loss CE --output-act none --lr 5e-4
 
 Project: SACI2022 - Evolutionary approach train Forex Robot
 Authors:  I. Pintye SZTAKI, 02/2022
 Cite/paper: I. Pintye, R. Lovas and J. Kovacs,
             "Evolutionary approach for neural network based agents applied on time series data in the Cloud"
             IEEE 16th International Symposium on Applied Computational Intelligence and Informatics SACI 2022
             10.1109/SACI55618.2022.9919475
------------------------------------------------------------------------------
"""


import os
import sys
import glob
import keras
import pprint
import sklearn
import ipywidgets
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.neural_network import MLPRegressor

from keras.models import Sequential
from keras.layers import Dense, Activation

from ipywidgets import interact, interactive, fixed, interact_manual, IntSlider, Checkbox
from IPython.display import display

from matplotlib.ticker import StrMethodFormatter
from matplotlib import cm

import time
import talib
import random
import joblib
from sys import maxsize
from datetime import datetime

import importlib

print('Python version:{}'.format(sys.version))
print('Numpy version:{}'.format(np.__version__))
print('Pandas version:{}'.format(pd.__version__))
print('Keras version:{}'.format(keras.__version__))
print('Tensorflow version:{}'.format(tf.__version__))
print('Sci-Kit Learn version:{}'.format(sklearn.__version__))

import logging
log_name = 'night_run_' + datetime.now().strftime('%Y%m%d-%H%M%S') + '.log'

log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger('pyApp')

file_handler = logging.FileHandler('{0}'.format(log_name))
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

logger.setLevel(logging.INFO)

a = 'Hello'
logger.info('%s Ez itt info', a)
logger.debug('%s Ez itt debug', a)
logger.warning('%s Ez itt warning', a)
logger.info('Ez itt egy formázott info {0}'.format(a))


import modules.datareader
from modules.datareader import DataReader

import modules.experiment
from modules.experiment import Experiment

import modules.nn
from modules.nn import NN, KerasMLP

import modules.trader
from modules.trader import Trader

import modules.randomer
from modules.randomer import Randomer

import modules.plotter
from modules.plotter import Plotter

import modules.fuk
from modules.fuk import plot_trade_adv
from modules.fuk import save_animated_mp4
from modules.fuk import create_animated_mp4
from modules.fuk import save_model_trader_results
from modules.fuk import run_info

# --

RELOAD_MODEL = False
RELOADED_MODEL = 'NightRuns/20230221-195640/20230221-195640_model.joblib'

ROW = 4000000
FROM = 0

# ROW = input('Válaszd ki, hogy hányadik sorig töltsük be (max 5 600 000) ')
# ROW = int(ROW)
# print('ROW =', ROW)

DIFF_MULTIPLIER = 1
INPUT_MULTIPLIER = 1

WINDOW = 1

INDICATORS = ['ROC1', 'ROC2', 'ROC3', 'RSI14', 'RSI28',
              'MACD',
              'MA50BIN', 'MA50DIS', 'MA100BIN', 'MA100DIS', 'MA200BIN', 'MA200DIS',
              'LIN30', 'LIN50',
              'RSIHL']

INDICATORS = ['ROC1', 'ROC2', 'ROC3', 'ROC4', 'ROC5', 'ROC6', 'ROC7', 'ROC8', 'ROC9', 'ROC10',
              'ROC11', 'ROC12', 'ROC13', 'ROC14', 'ROC15', 'ROC16', 'ROC17', 'ROC18', 'ROC19', 'ROC20',
              'ROC21', 'ROC22', 'ROC23', 'ROC24', 'ROC25', 'ROC26', 'ROC27', 'ROC28', 'ROC29', 'ROC30',
              'MACD',
              'MA50DIS', 'MA100DIS', 'MA200BIN', 'MA200DIS',
              'LIN30', 'LIN50',
              'RSIHL']





setup = dict()
setup['INDICATORS'] = INDICATORS
setup['WINDOW'] = WINDOW
setup['DIFF_MULTIPLIER'] = DIFF_MULTIPLIER
setup['INPUT_MULTIPLIER'] = INPUT_MULTIPLIER

data_reader = DataReader(nRowsRead=ROW)

data_reader.load_with_pandas(path = './temp/56.csv')
data_reader.df = data_reader.df.rename({data_reader.df.columns[0]: 'BC'}, axis=1)
data_reader.df['Date'] = np.arange(0, data_reader.df['BC'].size, 1) # az elötte lévő sorral együtt kel (!)

data_reader.cut(FROM, ROW)
data_reader.set_target()

# Depricated ->
# data_reader.create_diff(DIFF_MULTIPLIER)

# New
data_reader.create_input()

data_reader.create_indicators(extended=True, indicators=INDICATORS)

# Depricated ->
# data_reader.remove_price()
data_reader.drop_price()

# visszavágtam float 64-ről 16-ra
data_reader.retard()

# data_reader.normalize_values()       # experiment
# data_reader.normalize_values()       # experiment

data_reader.set_window(WINDOW)
data_reader.create_train_set()
data_reader.create_test_set()

# --

FIRST = 15
SECOND = 5

# Scikit
nn = NN(x_train = data_reader.x_train, y_train = data_reader.y_train)
nn.init_nn(_first = FIRST, _second = SECOND, activation='tanh')

# Keras
kn = KerasMLP(x_train = data_reader.x_train, y_train = data_reader.y_train)
kn.init_nn(_first = FIRST, _second = SECOND, activation='tanh')

# Reweight
_ = nn.knn_to_snn(kn)

# --

TRADE_COST = 0.0005                  # jelenleg a minimális költég 5x adtam az igazinak -> erős
TRADE_COST = 0.000160
MINIMALMOV = 0.000010
MINTPROFIT = 0.000160
DYNASPREAD = 0.000080                # a spread a minimal movenak a 8x
THRESHOLD = 0.0
TH = 0.4

trader = Trader(threshold=THRESHOLD,
                data_reader=data_reader,
                trade_cost=TRADE_COST,
                th=TH)

pred = nn.create_prediction()
result = trader.calculator_np(pred)

logger.info('--------------------------')
logger.info(' inicializált eredmény ')
logger.info(' ')
logger.info(f'{result}')
_ = data_reader.y_train.size // 1440
logger.info(f'nap = {_}')
logger.info(f"trade/nap = {result['buy_count'] / _}")
logger.info('--------------------------')

logger.debug(f'{nn.mlp.get_params()}')

# ---------------------------------------------------------------------

# itt lenne hogy megmérem egyként újonan incializált hálóval

# selected = input('Válaszd ki, hogy melyik modelt tanítsuk (0 a legjobb) ')

# selected = int(selected)

# print('Selected =', selected)


# ---------------------------------------------------------------------

# --
      
ex = Experiment(data_reader, nn)

if RELOAD_MODEL == True:
    ex.nn.mlp = joblib.load(RELOADED_MODEL)
    logger.info('RELOAD_MODEL')
    logger.info('OWERWIRTE ex.nn.mlp with {0}'.format(RELOADED_MODEL))

conf = dict()
conf['TRADE_COST'] = 0.000160 # 0.0005
conf['THRESHOLD']  = 0.0
conf['TH']         = 0.4

conf['internal_plotting'] = True
conf['repeatable']        = True
conf['re_learn']          = False

conf['save_interval']     = 100

conf['generation']        = 1000
conf['population']        = 20
conf['factor_weight']     = 0          # nem is használom ha _variable_
conf['factor_intercept']  = 0          # nem is használom ha _variable_
conf['variable_factor']   = True
conf['factor_min']        = 1         # 0.1   _minnél kisebb annál radikálisabban változtat a súlyon (átirn)
conf['factor_max']        = 200       # 200   _minnél nagyobb annál kevésbé változtat a súlyon
conf['keep_best']         = True
conf['select_ratio_weight']   = 1
conf['select_ratio_bias']     = 1
conf['variable_select_ratio'] = True
conf['select_ratio_min']      = 0.1     # 0.1
conf['select_ratio_max']      = 0.2     # 0.9
conf['setup'] = setup

logger.info('--------------------------')
logger.info('----------Experi----------')
logger.info('--------------------------')

ex.exp(conf)

# --

logger.info('----------Trader----------')
logger.info('--------------------------')

trader = ex.trader
trader.result

# save model
save_model_trader_results(ex)

# save
save_animated_mp4('price*.png', framerate=1, quite=True)

# shor ex runtime info
run_info(ex)
