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
 "datareader.py" - Construction of arbitrary network topologies.
 
 Project: SACI2022 - Evolutionary approach train Forex Robot
 Authors:  I. Pintye SZTAKI, 02/2022
 Cite/paper: I. Pintye, R. Lovas and J. Kovacs,
             "Evolutionary approach for neural network based agents applied on time series data in the Cloud"
             IEEE 16th International Symposium on Applied Computational Intelligence and Informatics SACI 2022
             10.1109/SACI55618.2022.9919475
------------------------------------------------------------------------------
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import talib

import logging
logger = logging.getLogger('pyApp')
logger.info('Data Reader modul loaded')

# végig néztem, data_readr-ben csak az x_train, y_train, y_train_originalt használom

class DataReader():
  """
  DataReader is responsible for holding, store the data and the basic
  data manipulation.
  """

  def __init__(self, nRowsRead):
    """
    :param nRowsRead: Number of rows read from the csv file via pandas.
    """
    self.nRowsRead = nRowsRead
    self.window = None

# ------------------------------------------------------------------------------

  def load_with_pandas(self, path = './input/eurusd_minute.csv'):
    """
    Load ./input/eurusd_minute.csv int pandas dataframe.
    """
    self.df = pd.read_csv(path, delimiter=',', nrows = self.nRowsRead)             # <-- read csv to pandas dataframe
    self.df.dataframeName = 'eurusd_minute.csv'

    self.n_row, self.n_col = self.df.shape
    print(f'There are {self.n_row} rows and {self.n_col} columns')                  # <-- print nRow, nCol
    print(self.df.head(5))                                                         # <-- print head
    
    logger.info(f'There are {self.n_row} rows and {self.n_col} columns')
    self.println()

    return self.df

# ------------------------------------------------------------------------------

  def cut(self, start: int, end: int):
    """
    Cut or Crop the dataframe.
    """
    self.df = self.df.iloc[start:end, :]
    self.start = start
    self.end = end
    print(self.df.shape)
        
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
  def info(self):
    """
    Print out the basic information about the loaded dataframe.
    """
    print('Info:')
    print(self.df.BC.values.ctypes)
    print(self.df.BC.values.dtype)
    print(self.df.BC.values.itemsize)
    print(self.df.BC.values.nbytes)

    print('BC.nbytes =', self.df.BC.values.nbytes)
    print('BC.nbytes =', self.df.BC.values.nbytes / 1000, 'Kbyte')
    self.println()

# ------------------------------------------------------------------------------

  def set_target(self):
    """
    Perform the necessery data transormation.
    For example: Keep only the BC and the Date columns.
    """

    # For BC column only
    self.df.set_index(['Date'], inplace=True)              # <-- set index column
    self.target = self.df.filter(['BC'])                   # <-- select only the BC column
    self.array = self.target.values                        # --> from pandas.series -> numpy.ndarray

# ------------------------------------------------------------------------------

  def retard(self):
    # self.array = self.array.astype(np.float16)
    
    self.input = self.input.astype(np.float16)
    pass


# ------------------------------------------------------------------------------

  def create_input(self):
    self.input = self.array.copy()

# ------------------------------------------------------------------------------


  def create_diff(self, emphasize):
    """
    Create diff n=1.
    """
    original = self.array
    diff = np.diff(original[:,0], n=1, prepend=original[0])       # Az array első oszlpán 
    diff = diff * emphasize                                 # mivel ez pici ért és nem akarom normálni
    diff = diff.reshape(-1, 1)
    plt.plot(diff)
    
    print(original.shape)
    print(diff.shape)
    
    self.println()

    hstack = np.hstack((original, diff))
    hstack.shape                              # (10000, 2)
    
    self.input = hstack
    
    del(original)
    del(diff)
    del(hstack)
    
# ------------------------------------------------------------------------------

  def remove_price(self):
    '''
    Remove price
    '''
    self.input = self.input[:, 1]
    self.input = self.input.reshape(-1, 1)
    print(self.input.shape)

# ------------------------------------------------------------------------------

  def drop_price(self):
    '''
    Drop price new
    '''
    self.input = self.input[:, 1:]
    print(self.input.shape)

# ------------------------------------------------------------------------------

  def create_indicators(self, extended = False, indicators = None):
    '''
    Bármit csinálok, állítok itt elő az legyen hozzácsapva a self.inputhoz
    és az akármit adok hozzá az mindíg a self.input hosszával legyen azonos
    :Ha hiányzó adattal kell kipótolnom, mint például a mozgóátlag,
    akkor a hiányzó érték legyen np.nan
    :A self.array az a változó amin valamit mókolunk és azt adjuk hozzá
    a self.inputhoz
    :Bár a self.inputból is lehet levállogatni ha az élet úgy hozza, de
    azzal zavar keletkezhet ha nem a megfelelő sorrandben hajtom végre
    az egyes parancsokat -->> ezért ezt soha ne tegyük.
    :Akkor már inkább paraméterként adjunk át neki valamit --> egy másik fgben
    '''
    original = self.array
    
    original = original.flatten()    # az original és a self.array is mindíg 1D de (n, 1)
    
    
    print(original.shape)
    print(self.input.shape)
    
    print('indcator section')

    #_indicator = talib.SMA(real = original, timeperiod = 30)
    #_indicator[:30] = _indicator[30]
    #_indicator = _indicator.reshape(-1, 1)
    #assert _indicator.shape[0] == self.input.shape[0]
    #self.input = np.hstack((self.input, _indicator))
        
    # ---------------- RSI
    

    
    # ----------------
    # ----------------
    # ----------------
    # ----------------
    # ----------------
    if extended:
        
        rocs = [x for x in indicators if 'ROC' in x]       # -> ROC1, ROC2, stb formában
        
        for i in range(len(rocs)):
            _ = int(rocs[i][3:])
            _indicator = talib.ROC(original, timeperiod=_) # -> ez a diff tulajdonképen
            _indicator[:_] = _indicator[_]
            _indicator = _indicator.reshape(-1, 1)
            
            self.input = np.hstack((self.input, _indicator))
        
#        if 'ROC1' in indicators:
#        
#            _indicator = talib.ROC(original, timeperiod=1) # -> ez a diff tulajdonképen
#            _indicator[:1] = _indicator[1]
#            _indicator = _indicator.reshape(-1, 1)
#
#            self.input = np.hstack((self.input, _indicator))


        if 'RSI14' in indicators:
            
            _indicator = talib.RSI(real = original, timeperiod = 14)
            _indicator = _indicator - 50
            _indicator = _indicator / 100
            _indicator[:14] = _indicator[14]
            _indicator = _indicator.reshape(-1, 1)

            self.input = np.hstack((self.input, _indicator))

        if 'RSI28' in indicators:
            
            _indicator = talib.RSI(real = original, timeperiod = 28)
            _indicator = _indicator - 50
            _indicator = _indicator / 100
            _indicator[:28] = _indicator[28]
            _indicator = _indicator.reshape(-1, 1)

            self.input = np.hstack((self.input, _indicator))
        
        if 'MACD' in indicators:

            _indicator = talib.MACD(original)
            _sig = _indicator[0]
            _mac = _indicator[1]
            _dif = _indicator[2]
            _sig[:33] = _sig[33]
            _mac[:33] = _mac[33]
            _dif[:33] = _dif[33]
            _sig = _sig.reshape(-1, 1)
            _mac = _mac.reshape(-1, 1)
            _dif = _dif.reshape(-1, 1)
            
            _sig = _sig * 300
            _mac = _mac * 900
            _dif = _dif * 1000

            self.input = np.hstack((self.input, _sig))
            self.input = np.hstack((self.input, _mac))
            self.input = np.hstack((self.input, _dif))
            
        if 'STDDEV10' in indicators:
            
            _indicator = talib.STDDEV(real = original, timeperiod = 10)
            _indicator[:10] = _indicator[10]
            _indicator = _indicator.reshape(-1, 1)
            
            _indicator = _indicator * 100

            self.input = np.hstack((self.input, _indicator))

        if 'MA50BIN' in indicators:
            
            _indicator = talib.MA(original, timeperiod = 50)
            _indicator[:50] = _indicator[50]
            _in = _indicator < original
            _indicator = _in.astype('float')
            _indicator = _indicator.reshape(-1, 1)
            
            _indicator = ((_indicator - 0.5) * 2)
            
            self.input = np.hstack((self.input, _indicator))
        
        if 'MA50DIS' in indicators:
            _indicator = talib.MA(original, timeperiod = 50)
            _indicator[:50] = _indicator[50]
            _indicator = _indicator - original
            _indicator = _indicator.reshape(-1, 1)
            
            _indicator = _indicator * 100
            
            self.input = np.hstack((self.input, _indicator))

        if 'MA100BIN' in indicators:
            
            _indicator = talib.MA(original, timeperiod = 100)
            _indicator[:100] = _indicator[100]
            _in = _indicator < original
            _indicator = _in.astype('float')
            _indicator = _indicator.reshape(-1, 1)
            
            _indicator = ((_indicator - 0.5) * 2)
            
            self.input = np.hstack((self.input, _indicator))
        
        if 'MA100DIS' in indicators:
            _indicator = talib.MA(original, timeperiod = 100)
            _indicator[:100] = _indicator[100]
            _indicator = _indicator - original
            _indicator = _indicator.reshape(-1, 1)
            
            _indicator = _indicator * 100
            
            self.input = np.hstack((self.input, _indicator))

        if 'MA200BIN' in indicators:
            
            _indicator = talib.MA(original, timeperiod = 200)
            _indicator[:200] = _indicator[200]
            _in = _indicator < original
            _indicator = _in.astype('float')
            _indicator = _indicator.reshape(-1, 1)
            
            _indicator = ((_indicator - 0.5) * 2)
            
            self.input = np.hstack((self.input, _indicator))

        if 'MA200DIS' in indicators:
            _indicator = talib.MA(original, timeperiod = 200)
            _indicator[:200] = _indicator[200]
            _indicator = _indicator - original
            _indicator = _indicator.reshape(-1, 1)
            
            _indicator = _indicator * 100
            
            self.input = np.hstack((self.input, _indicator))

        if 'LIN50' in indicators:
            _indicator = talib.LINEARREG_ANGLE(original, timeperiod = 50)
            _indicator[:50] = _indicator[50]
            _indicator = _indicator.reshape(-1, 1)
            
            _indicator = _indicator * 100
            
            self.input = np.hstack((self.input, _indicator))

        if 'LIN30' in indicators:
            _indicator = talib.LINEARREG_ANGLE(original, timeperiod = 30)
            _indicator[:30] = _indicator[30]
            _indicator = _indicator.reshape(-1, 1)
            
            _indicator = _indicator * 30
            
            self.input = np.hstack((self.input, _indicator))

        if 'RSIHL' in indicators:
            for i in [20, 50, 80]:
                
                _indicator = talib.MAX(talib.RSI(original, timeperiod=14), timeperiod=i)
                _indicator = _indicator - 50
                _indicator = _indicator / 100
                _z = 15 + i
                _indicator[:_z] = _indicator[_z]
                _indicator = _indicator.reshape(-1, 1)

                self.input = np.hstack((self.input, _indicator))

            for i in [20, 50, 80]:
                
                _indicator = talib.MIN(talib.RSI(original, timeperiod=14), timeperiod=i)
                _indicator = _indicator - 50
                _indicator = _indicator / 100
                _z = 15 + i
                _indicator[:_z] = _indicator[_z]
                _indicator = _indicator.reshape(-1, 1)

                self.input = np.hstack((self.input, _indicator))


        if 'STDDEV' in indicators:
            for i in [200, 1440, 3440]:
                
                _indicator = talib.STDDEV(original, timeperiod=i)
                _z = 1 + i
                _indicator[:_z] = _indicator[_z]
                _indicator = _indicator.reshape(-1, 1)

                self.input = np.hstack((self.input, _indicator))
    
    
    
    del(original)
    del(_indicator)
    del(_in)
    # ----------------
    # ----------------
    # ----------------
    # ----------------
    # ----------------
    self.println()
    pass

# ------------------------------------------------------------------------------

  def show_array_info(self):
    """
    Print out the basic information about the transformed dataframe.
    """
    self.println()
    print(type(self.array))
    print(self.array.shape)
    self.println()

# ------------------------------------------------------------------------------

  def normalize_values(self):
    """
    Normalize the whole input array.
    It transforms the whole dataframe between {-1,+1}.
    """
    
    # Bug:
    # The windowed data should be normalized respectivley, not the whole dataframe.

    self.scaler      = MinMaxScaler(feature_range=(-1, 1))                   # <-- (0, 1) vagy (-1, 1)
    self.input       = self.scaler.fit_transform(self.input)
    
# ------------------------------------------------------------------------------

  def set_window(self, value: int):
    """
    Set the lookback window size.
    :param value: The lookback period.
    """
    if type(value) is int:
      self.window = value
      print('Set window =', self.window)
    else:
      print('Error: Not proper type of parameter')
    self.println()

# ------------------------------------------------------------------------------

  def create_train_set(self):
    """
    Creates the train and the test dataset.
    """

    self.x_train = []
    self.y_train = []    # --> soha nem fogjuk használni emiatt csak az nn weigth inithez kell
    
    self.y_train = self.array[self.window:,] # le kell vágni az y_train-t hogy az x_trainel azonos h. legyen


    for i in range(self.window, len(self.input)):
        # self.x_train.append(self.input[i-self.window:i, 0])               # egy konkrét oszlop -> nem jó
        # self.x_train.append(self.input[i-self.window:i, 0:2])             # új   [, 0:2]  az első két oszlop
        self.x_train.append(self.input[i-self.window:i, :])               # új [, :] az összes oszlop
        # self.y_train.append(self.array[i, 0])                           # felesleges a self.y_train nem használjuk

    print('The train x_train {}, and the labels {}'.format(len(self.x_train), len(self.y_train)))

    self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)   # <-- Convert list to Numpy Array

    print(self.x_train.shape, self.y_train.shape)
    
    print('x_train.nbytes = ', self.x_train.nbytes/1000, 'Kbyte')
    print('x_train.nbytes = ', self.x_train.nbytes/1000/1000, 'Mbyte')


    # Mivel az MLPRegressioin nem képes csak 2D tömöbket fogadni bemenetnek ezért kell ez az átalakítás
    
    _ = []
    for i in range(self.x_train.shape[-1]):
        _.append(self.x_train[:, :, i])
    _ = tuple(_)
    self.x_train = np.hstack(_)

    self.x_train
    
    del(_)

    # <-- így néz ki a helyes összefűzési mód -> így már lehet az MLPRegression bemente
    
    self.println()

# ------------------------------------------------------------------------------

  def create_test_set(self):
    """
    Creates the test and the test dataset.
    """
    
    # a legegyszerűbb az lenne, hogyha még az elején megadnám, hogy mi legyen a
    # train test ratio akkor a cut segítségével már ketté tudnám vágni
    
    
    # s akkor az ami itt lefut az igazából minden ami a cut után lefut
    
    print(self.start)
    print(self.end)
    
    # tulajdonképpen csak az end kell -> innnen indul ugyanis a test adatsorunk
    
    # a cut a df-ből indul de sajnos felül is írja
    
    # -----
    
    # a másik dirty hack megoldásom az lenne, hogy a tanítás végén egész egyszerűen
    # újra olvasom és generálom az adtokat, csak hosszabban mint korábban és
    # egyszerűen ráeresztem a már feltanított modellt.
    
    
    print('create test set section')
    
    pass

# ------------------------------------------------------------------------------

  def sin(self, length, p, m, l, t, e):
    'Dummy generator'

    ll, l = l

    x = np.arange(0, length, 1)
    s0 = np.sin(x/180*np.pi*p)                        # p = phase
    s1 = m*np.sin(x/180*np.pi*p)                      # m = magnitúdó
    s2 = m*np.sin(x/180*np.pi*(1+(l*(x/ll)))*p)       # l = longitude modulation
    s3 = m*np.sin(x/180*np.pi*(1+(l*(x/ll)))*p)+x*t   # t = linear trend
    s4 = (m+(m*e*x))*np.sin(x/180*np.pi*(1+(l*(x/ll)))*p)+x*t

    plt.figure(figsize=(12, 4))
    plt.plot(s0, label='s0')
    plt.plot(s1, label='s1')
    plt.plot(s2, label='s2')
    plt.plot(s3, label='s3')
    plt.plot(s4, label='s4', color='black')    

    plt.legend(frameon=False)
    plt.show()

    return s4

# ------------------------------------------------------------------------------

  def create_dummy(self, length, p, m, l=(179.0, 0.0), t=0.0, e=0.0):
    """
    Create dummy data
    """
    x = np.arange(0, length, 1)
    s = np.sin(x/180*np.pi*p)
    s = self.sin(length, p, m, l, t, e)


    xs = np.dstack((x,s))
    xs = np.squeeze(xs, axis=0)

    df = pd.DataFrame(xs, columns = ['Date','BC'])
    
    self.df = df


# ------------------------------------------------------------------------------

  def println(self):
    print('------------------------------------------------------------------------------')

