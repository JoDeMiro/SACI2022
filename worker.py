# Worker file

print('---------------------------------------------------------------')
print('                         SET VARIABLE                          ')
print('---------------------------------------------------------------')

driver_ip_address = '192.168.0.114'





print('---------------------------------------------------------------')
print('                         CREATE DAO                            ')
print('---------------------------------------------------------------')

class Parameters():
    
    def __init__(self):
        self.driver_ip = 'http://192.168.0.1' #114
        self.worker_id = 123
        self.nRowsRead = 1000
        self.window = 2
        self.threshold = -0.0
    
    def set_driver_ip(self, _driver_ip):
        self.driver_ip = _driver_ip

    def set_worker_id(self, _worker_id):
        self.worker_id = _worker_id

    def set_nRowsRead(self, _nRowsRead):
        self.nRowsRead = _nRowsRead
    
    def set_window(self, _window):
        self.window = _window
    
    def set_threshold(self, _threshold):
        self.threshold = _threshold
    
    def __str__(self):
        return 'Parameters Class(driver_ip=' + str(self.driver_ip) + 'worker_id=' + str(self.worker_id) + 'nRowsRead=' + str(self.nRowsRead) + ' ,window=' + str(self.window) + ', threshold=' + str(self.threshold) + ')'

parameters = Parameters()
print(parameters)


print('---------------------------------------------------------------')
print('                         BCOLORS ENUM                          ')
print('---------------------------------------------------------------')

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



print('---------------------------------------------------------------')
print('                         HELLO WORKER                          ')
print('---------------------------------------------------------------')


print('---------------------------------------------------------------')
print('                         IMPORT                                ')
print('---------------------------------------------------------------')


import gc
import os
import sys
import time
import json
import pprint
import sklearn
import requests
import threading
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.neural_network import MLPRegressor

from flask import Flask,jsonify,request,make_response,url_for,redirect

print('Python version:{}'.format(sys.version))
print('Numpy version:{}'.format(np.__version__))
print('Pandas version:{}'.format(pd.__version__))
print('Sci-Kit Learn version:{}'.format(sklearn.__version__))


print('---------------------------------------------------------------')
print('                       UNZIP CSV DATA                          ')
print('---------------------------------------------------------------')

os.system('gzip -f -d ./Input/eurusd_minute_1000000.csv.gz')
os.system('mv ./Input/eurusd_minute_1000000.csv ./Input/eurusd_minute.csv')

# probl??m??s, hogy itt fel??l csapom mert a git nem fogja a pull hat??s??ra ??jra lehuzni!
# ez??rt az './Input/eurusd_minute_1000000.csv.gz' m??r nincs ott ??s panadszodik.
# nem baj, de j?? ??szben tartani

os.system('rm ./Input/eurusd_minute_2000000.gz')
os.system('cat ./Input/eurusd_minute_2000000.csv.gz.* >./Input/eurusd_minute_2000000.gz')
os.system('gzip -f -d ./Input/eurusd_minute_2000000.gz')
os.system('mv ./Input/eurusd_minute_2000000 ./Input/eurusd_minute.csv')


print('---------------------------------------------------------------')
print('                       DATA READER CLASS                       ')
print('---------------------------------------------------------------')

from matplotlib.image import NonUniformImage
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

class DataReader():

  def __init__(self, nRowsRead):
    self.nRowsRead = nRowsRead
    self.dataset = None
    self.dataset_full = None
    self.window = None                      # <- Check (rosszul volt kor??bban be??ll??tva)

# ------------------------------------------------------------------------------

  def load_with_pandas(self, path = './input/eurusd_minute.csv'):
    self.df2 = pd.read_csv(path, delimiter=',', nrows = self.nRowsRead)             # <-- read csv to pandas dataframe
    self.df2.dataframeName = 'eurusd_minute.csv'

    self.n_row, self.n_col = self.df2.shape
    print(f'There are {self.n_row} rows and {self.n_col} columns')                  # <-- print nRow, nCol
    print(self.df2.head(5))                                                         # <-- print head
    self.println()

    return self.df2

# ------------------------------------------------------------------------------

  def info(self):
    print('Info:')
    print(self.df2.BC.values.ctypes)
    print(self.df2.BC.values.dtype)
    print(self.df2.BC.values.itemsize)
    print(self.df2.BC.values.nbytes)

    print('BC.nbytes =', self.df2.BC.values.nbytes)
    print('BC.nbytes =', self.df2.BC.values.nbytes / 1000, 'Kbyte')
    self.println()

# ------------------------------------------------------------------------------

  def prepare_data(self):
    self.df = self.df2

    # For BC column only
    self.df.set_index(['Date'], inplace=True)              # <-- set index column
    self.data = self.df.filter(['BC'])                     # <-- select only the BC column
    self.dataset = self.data.values                        # --> from pandas.series -> numpy.ndarray

    # Check
    # T??r??lj??k a Pandas Dataframet mert t??bb?? m??r nincs sz??ks??gem r??
    if hasattr(self, 'df'):
      print('----------------> t??r??lj??k a self.df-t mert m??r nincs r?? sz??ks??g')
      del self.df
    if hasattr(self, 'df2'):
      print('----------------> t??r??lj??k a self.df2-t is mert m??r arra sincs sz??ks??g')
      del self.df2
    if hasattr(self, 'data'):
      print('----------------> t??r??lj??k a self.data-t is mert m??r arra sincs sz??ks??g')
      del self.data
    if hasattr(self, 'df'):
      print('----------------> ' + bcolors.WARNING + 'ha ez az ??zenet l??tsz??dik akkor nem t??rl??d??tt a self.df  ' + bcolors.ENDC)
    if hasattr(self, 'df2'):
      print('----------------> ' + bcolors.WARNING + 'ha ez az ??zenet l??tsz??dik akkor nem t??rl??d??tt a self.df2 ' + bcolors.ENDC)
    if hasattr(self, 'data'):
      print('----------------> ' + bcolors.WARNING + 'ha ez az ??zenet l??tsz??dik akkor nem t??rl??d??tt a self.data' + bcolors.ENDC)


    # For all column (I do not use it)
    # data_full = self.df.filter(['BO', 'BH', 'BL', 'BC', 'BCh', 'AO', 'AH', 'AL', 'AC', 'ACh'])
    # self.dataset_full = data_full.values                   # --> from pandas.series -> numpy.ndarray

# ------------------------------------------------------------------------------

  def show_dataset_info(self):
    self.println()
    print(type(self.dataset))
    # print(type(self.dataset_full))
    print(self.dataset.shape)
    # print(self.dataset_full.shape)
    self.println()

# ------------------------------------------------------------------------------

  def normalize_values(self):
    
    self.scaler        = MinMaxScaler(feature_range=(-1, 1))                        # <-- (0, 1) vagy (-1, 1)
    self.scaled_data   = self.scaler.fit_transform(self.dataset)

    # self.scaler_full        = MinMaxScaler(feature_range=(-1, 1))                   # <-- (0, 1) vagy (-1, 1)
    # self.scaled_data_full   = self.scaler_full.fit_transform(self.dataset_full)

# ------------------------------------------------------------------------------

  def set_window(self, value: int):
    if type(value) is int:
      self.window = value
      print('Set window =', self.window)
    else:
      print('Error: Not proper type of parameter')
    self.println()


# ------------------------------------------------------------------------------

  def create_train_set(self):
    self.x_train_list = []                                                               # <-- Create list of Windows
    self.y_train_list = []

    for i in range(self.window, len(self.scaled_data)):
        self.x_train_list.append(self.scaled_data[i-self.window:i, 0])
        self.y_train_list.append(self.scaled_data[i, 0])

    print('The train dataset {}, and the labels {}'.format(len(self.x_train_list), len(self.y_train_list)))

    self.x_train, self.y_train = np.array(self.x_train_list), np.array(self.y_train_list)     # <-- Convert list to Numpy Array

    if hasattr(self, 'x_train_list'):
      print('----------------> t??r??lj??k a self.x_train_list-t is mert m??r nincs r?? sz??ks??g')
      del self.x_train_list
    if hasattr(self, 'y_train_list'):
      print('----------------> t??r??lj??k a self.y_train_list-t is mert m??r nincs r?? sz??ks??g')
      del self.y_train_list



    print(self.x_train.shape)
    print(self.y_train.shape)

    print('x_train.nbytes = ', self.x_train.nbytes/1000, 'Kbyte')
    print('x_train.nbytes = ', self.x_train.nbytes/1000/1000, 'Mbyte')

    self.x_train_reshaped = np.reshape(self.x_train,                                # <-- Nagyon fontos az adatokat az al??bbi form??ban v??rjuk
                                       newshape=(self.x_train.shape[0],
                                                 self.x_train.shape[1], 1))         # <-- Reshape (970, 30) -> (970, 30, 1)
    
    print(self.x_train_reshaped.shape)
    print(self.x_train.shape)
    print(self.y_train.shape)
    self.println()

    gc.collect()
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)


    # --------------------------------------------------------------------------    # <-- ToDo: ezt itt hagyom de nem haszn??lom

    # x_train_full = []                                                               # <-- Create list of Windows
    # y_train_full = []

    # for i in range(self.window, len(self.scaled_data_full)):
    #     x_train_full.append(self.scaled_data_full[i-self.window:i, :])
    #     y_train_full.append(self.scaled_data_full[i, 0])

    # print('The train dataset {}, and the labels {}'.format(len(x_train_full), len(y_train_full)))

    # x_train_full, y_train_full = np.array(x_train_full), np.array(y_train_full)     # <-- Convert list to Numpy Array

    # print(x_train_full.shape)
    # print(y_train_full.shape)

    # print('x_train_full.nbytes = ', x_train_full.nbytes/1000, 'Kbyte')
    # print('x_train_full.nbytes = ', x_train_full.nbytes/1000/1000, 'Mbyte')

# ------------------------------------------------------------------------------

  def clean(self):
          deletable_variables = ['df', 'df2', 'data', 'dataset', 'dataset_full', 'sclaer', 'scaled_data', 'window',
                                 'x_train_list', 'y_train_list', 'x_train', 'y_train', 'x_train_reshaped']
          print(deletable_variables)
          for var_str in deletable_variables:
              print(var_str)
              # Check
              # T??r??lj??k a Pandas Dataframet mert t??bb?? m??r nincs sz??ks??gem r??
              if hasattr(self, var_str):
                  print('----------------> t??r??lj??k a ', var_str, ' mert m??r nincs r?? sz??ks??g')
                  delattr(self, var_str)

# ------------------------------------------------------------------------------

  def println(self):
    print('------------------------------------------------------------------------------')


    
print('---------------------------------------------------------------')
print('                       TEST READER CLASS                       ')
print('---------------------------------------------------------------')


# ------------------------------------------------------------------------------

data_reader = DataReader(nRowsRead=1000)                                        # <-- instantiate DataReader (set number of rows for data)

df2 = data_reader.load_with_pandas(path = './Input/eurusd_minute.csv')          # <-- read a particular files from disk (return with df, but also set self)

data_reader.info()                                                              # <-- my own info() function

data_reader.df2.info()                                                          # <-- call pandas built-in info() function

data_reader.prepare_data()                                                      # <-- prepare is setter convert pandas to numpy and set target variable

data_reader.show_dataset_info()

data_reader.normalize_values()                                                  # <-- transform data between range (-1,1)

data_reader.set_window(2)                                                       # <-- set window size

data_reader.create_train_set()


print('---------------------------------------------------------------')
print('                       INITIALIZE DATA_READER                  ')
print('---------------------------------------------------------------')

def initialize_data_reader(_nRowsRead=1000, _window=2):
    '''
    L??trehoz egy data readder objektumot el??k??sz??ti ??s azt adja vissza
    '''

    data_reader = DataReader(nRowsRead=_nRowsRead)                              # <-- instantiate DataReader (set number of rows for data)
    df2 = data_reader.load_with_pandas(path = './Input/eurusd_minute.csv')      # <-- read files disk (return with df, but also set self)

    data_reader.info()                                                          # <-- my own info() function
    # data_reader.df2.info()                                                      # <-- call pandas built-in info() function
    data_reader.prepare_data()                                                  # <-- prepare is setter convert pd to np, set target var.

    # data_reader.show_dataset_info()
    data_reader.normalize_values()                                              # <-- transform data between range (-1,1)
    data_reader.set_window(_window)                                             # <-- set window size
    data_reader.create_train_set()
    
    return data_reader





print('---------------------------------------------------------------')
print('                       NN CLASS                                ')
print('---------------------------------------------------------------')

# ------------------------------------------------------------------------------    # <-- NN Class

# Neural Network Class

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)                                 # <-- kell a Convergencia Warning miatt
simplefilter("ignore")                                                              # <-- a batch_size > n miatt kell ide

import numpy as np
from sklearn.neural_network import MLPRegressor

# ------------------------------------------------------------------------------

class NN():

  def __init__(self, x_train, y_train):
    self.mlp = None
    self.x_train = x_train
    self.y_train = y_train
    self.prediction = None

# ------------------------------------------------------------------------------
  
  def init_nn(self, _first = 15, _second = 5):
    'Init Scikit Learn MLPRegressor'                                                 # <-- hogy l??tre j??jjenek a s??lyok inicializ??lni kell
    
    np.random.seed(1)

    mlp = MLPRegressor(hidden_layer_sizes=(_first, _second),
                      activation='tanh',                                             # -------> ha (MinMax(-1,1) vagy StandardScaler())
                      solver='sgd',
                      batch_size=100000,                                             # <<-- v.017 bug fixed
                      max_iter=1,                                                    # <-- sajnos legal??bb 1 kell hogy legyen
                      shuffle=False,
                      random_state=1,
                      learning_rate_init=0.00000001,                                 # >- lehet??leg ne tanuljon semmit GD alapj??n
                      validation_fraction=0.0,
                      n_iter_no_change=99999999)
    
    # ----->                                                         Behoztam ide az els?? illeszt??st is, hogy meglegyenek neki a s??lyok

    np.random.seed(1)

    y_random = np.zeros((self.y_train.shape[0])) * 110.01                           # --> t??k random adaton tan??tom, hogy m??g v??letlen??l se tanuljon

    mlp.fit(self.x_train, y_random)                                                 # --> nem akarjuk mi semmire megtan??tani csak kell az inithez

    self.mlp = mlp
    
    return mlp

# ------------------------------------------------------------------------------

  def create_prediction(self):
    'Saj??t adati alapj??n csin??ja meg a predcitiont'

    self.prediction = self.mlp.predict(self.x_train)

    return self.prediction

print('---------------------------------------------------------------')
print('                       NN CLASS TEST                           ')
print('---------------------------------------------------------------')

# ------------------------------------------------------------------------------

nn = NN(x_train = data_reader.x_train, y_train = data_reader.y_train)

nn.init_nn(_first = 15, _second = 5)

# nn.mlp.coefs_                                                                     # <-- ha debuggolni kell


# ------------------------------------------------------------------------------

# Create Prediction

mlp = nn.init_nn(2, 2)

test_pred = mlp.predict(data_reader.x_train)                                        # teszt pred --> semmire nem fogjuk haszn??lni

print(test_pred[0:5])

test_pred = nn.mlp.predict(data_reader.x_train)                                     # init ut??n direktben is el lehet ??rni az mlp.predict() f??ggv??nyt

print(test_pred[0:5])

test_pred = nn.create_prediction()                                                  # csin??ltam neki egy saj??t f??gv??nyt ami elv??gzi az eg??sz predictiont

print(test_pred[0:5])

# ------------------------------------------------------------------------------

# A scikit-t??l elt??r?? s??lyincializ??c??ra van lehet??s??g l??tre hoztam r?? egy
# elj??r??st de nem haszn??lom. Ha m??gis haszn??lni szeretn??m akkor a
# SACI22 - 018.ipynb-ben megtal??lhat??

# ------------------------------------------------------------------------------







print('---------------------------------------------------------------')
print('                       TRADER CLASS                            ')
print('---------------------------------------------------------------')

# ------------------------------------------------------------------------------    # <-- Trader Class

# Trader Calculator Class

import numpy as np

class Trader():

  def __init__(self, threshold, data_reader, debug = False):
    self.threshold = threshold
    self.data_reader = data_reader
    self.debug = debug
    print('__init__ Trader')

  def calculator(self, pred):
    buy   = pred > self.threshold
    sell  = pred < self.threshold

    sunique, scounts = np.unique(sell, return_counts=True)
    sell_stat = dict(zip(sunique, scounts))

    bunique, bcounts = np.unique(buy, return_counts=True)
    buy_stat = dict(zip(bunique, bcounts))

    lenght = pred.size

    is_in_trade = False
    is_in_buy = False
    buy_count = 0
    sell_count = 0
    buy_price = []
    sell_price = []
    buy_index = []
    sell_index = []
    for i in range(lenght):
      if buy[i] == True and is_in_trade == False:
        buy_count += 1
        buy_price.append(self.data_reader.y_train[i])
        buy_index.append(i)
        is_in_trade = True
      
      if sell[i] == True and is_in_trade == True:
        sell_count += 1
        sell_price.append(self.data_reader.y_train[i])
        sell_index.append(i)
        is_in_trade = False

      if i == lenght - 1 and is_in_trade == True:                                       # <-- le kell z??rni az utols??n??l a v??telt ha nyitva van
        sell_count += 1
        sell_price.append(self.data_reader.y_train[i])
        sell_index.append(i)
        is_in_trade = False
    
    gains = np.array(sell_price) - np.array(buy_price)
    # print(gains)

    gain = gains.sum()
    # print(gain)

    if (self.debug == True):
      print('Summary :')
      print('buy_stat = ', buy_stat)
      print('sell_stat = ', sell_stat)
      print('buy_count = ', buy_count)
      print('sell_count = ', sell_count)
      print('len(buy_price) = ', len(buy_price))
      print('len(sell_price) = ', len(sell_price))
      print('buy_price  = ', buy_price)
      print('sell_price = ', sell_price)
      print('buy_index  = ', buy_index)
      print('sell_index = ', sell_index)
      # print('gains      = ', gains)
      print('gain       = ', gain)

    self.result = {'buy_price': buy_price, 'sell_price': sell_price, 'buy_index': buy_index, 'sell_index': sell_index}

    result = {'buy_stat': buy_stat.get(True), 'sell_stat': sell_stat.get(True), 'buy_count': buy_count, 'sell_count': sell_count, 'gain': gain}

    return result

# ------------------------------------------------------------------------------

  def clean(self):
          deletable_variables = ['data_reader']
          print(deletable_variables)
          for var_str in deletable_variables:
              print(var_str)
              # Check
              # T??r??lj??k az al??bbi v??ltoz??kat
              if hasattr(self, var_str):
                  print('----------------> t??r??lj??k a ', var_str, ' mert m??r nincs r?? sz??ks??g')
                  delattr(self, var_str)

# ---------------------------------------------------------------------------


print('---------------------------------------------------------------')
print('                       TRADER CLASS TEST                       ')
print('---------------------------------------------------------------')


# ------------------------------------------------------------------------------    # <-- Test Trader Class

new_trader = Trader(threshold = -0.0, data_reader = data_reader)                    # <-- Ebben a form??ban kell majd haszn??lni

pred = nn.create_prediction()

print(pred[0:5])

result = new_trader.calculator(pred)                                                # <-- Ebben a form??ban kell majd haszn??lni

result.keys()                           # <- get dict.keys
result.get('gain')                      # <- get a value by a given key
result['gain']                          # <- get a value by a given key             # <-- Ebben a form??ban kell majd haszn??lni

# ------------------------------------------------------------------------------    # <-- Test has successed


# A Trader oszt??lyon m??g sz??mos keresked??si eredm??nyt lehet hadsznlni,
# de ezeket most nem haszn??lom. Ha ??rdekel akkor a SACI22 - 018.ipynb
# f??jlban meg lehet tal??lni


# ------------------------------------------------------------------------------



print('---------------------------------------------------------------')
print('                       DATA SENDER CLASS                       ')
print('---------------------------------------------------------------')

from matplotlib.image import NonUniformImage

class DataSender():

  def __init__(self):
    self.driver_ip_address = None

  def send_to_driver(self, data = None, model_id = None):
    '''
    Elk??ldi az Trader eredm??ny??t (result) a Drivernek
    Ez egy sima GET Request lesz amit a tuloldalon v??r a Driver
    '''
    print('------------>>    send_to_driver(self, data)    data is sending to the driver')
    print('------------>>    self.driver_ip_address', self.driver_ip_address)
    print('------------>>    sended from worker to drive model_id', model_id)
    print('zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz')

    # aaaaa
    # na itt k??ne megh??vni egy sima requestes szar
    # ??tk??ld??k egy ??rt??ket a Drivernek
    resp = requests.get(self.driver_ip_address + '/receiveresult?gain=' + (str)(data) + '&model_id=' + str(model_id))
    print('receiveresult  ', resp)
    
    pass

  def initialize_data_sender(self, driver_ip_address):
    self.driver_ip_address = driver_ip_address



print('---------------------------------------------------------------')
print('                       RANDOMER CLASS                          ')
print('---------------------------------------------------------------')

# Na ezt a Driver fogja csin??lni









print('---------------------------------------------------------------')
print('                       F??GGV??NYEK A FLASKHEZ                   ')
print('---------------------------------------------------------------')

from sklearn.neural_network import MLPRegressor
import numpy as np
import pickle
import joblib
import json
import os

# Van egy parameters objektum az elej??n azt adjuk ??t ??s olvassuk ki.

def initialize_worker(_parameters):
    '''
    Be??ll??tunk a Workeren n??h??ny objektum??nak a param??tereit.
    ??gy mint a Trader(threshod), vagy mint a DataReader(nRowsRead, window), vagy mint az NN(arch)
    '''
    
    print('-------------------------------INITIALIZE WORKER---------------------')
    
    # A parameters objektumb??l olvassuk ki a parametereket
    
    print(_parameters)
    print('\n')
    
    _worker_id = _parameters.worker_id
    _nRowsRead = _parameters.nRowsRead
    _window = _parameters.window
    _threshold = _parameters.threshold
    
    global data_reader                                                                    # <-- hogy fel??l csapja a glob??lisat
    data_reader = initialize_data_reader(_nRowsRead=_nRowsRead, _window=_window)          # <-- Initialize data_reader
        
    global data_sender
    data_sender = DataSender()                                                            # <-- Initialize data_sender
    data_sender.initialize_data_sender(_parameters.driver_ip)                             # <-- set driver_ip_address

    
    print('-------------------------------INITIALIZE WORKER DONE----------------')

    

def load_model():
    '''
    A s??lyok bet??lt??s????rt felel
    '''
    clf = joblib.load('model.joblib')                                                     # <-- bet??ltj??k a modlet a fil??b??l
    print('# Model bet??ltve a joblib-b??l')
    print(clf.get_params())
    
    return clf



def evaluate_model(mlp, model_id):
    '''
    Az evaluate_model() f??ggv??nynek akkor kell lefutnia amikor kap egy modelt a worker
    a drivert??l k??v??lr??l, m??sk??pp nem h??vhat??.
    
    A data_reader ??s a parameters objektumok amiket haszn??l ez a f??ggv??ny globalisok
    '''
    
    print('-------------------------------GC COLLECT----------------------------')

    gc.collect()
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)


    print('-------------------------------EVALUATE MODEL------------------------')
    
    # A globalis data_readert l??tja ez a f??ggv??ny
    # data_reader.show_dataset_info()
    # Ellen??rizni szoktam vele, hogy az ??t??l??otott ??s ??j data_reader objektumot l??tja-e az nRowsRead alapj??n tudom, hogy j??
    
    
    # Az initialize() megcsin??lta nek??nk a data_reader ??s a trader objektumokat
    
    # A data_reader objektumot csak egyszer kell l??trehozni, de a trader-t azt minden fut??sn??l, ugyhogy p??ld??nyos??tsunk egyet
    
    # A globalis data_readert l??tja ez a f??ggv??ny
    print('---------IN EVALUATE MODEL, CREATES TRADER WITH PARAMETER------------')
    print(parameters)

    threshold = parameters.threshold
    trader = Trader(threshold = threshold, data_reader = data_reader)
    
    
    # el??sz??r is az NN alapj??n csin??lunk egy predictiont
    
    # honnan j??n a modell ami alapj??n a becsl??st csin??ljuk.?
    # h??t a f??ggv??nyb??l ami ezt a f??ggv??ynt h??vja meg!
    # ezt a f??ggv??nyt az upload() h??vja
    # el??tte lefut a load_model()
    # ami visszaad egy modelt
    # a modell ennek a f??ggv??nynek a bemenete

    print('----------------------< itt szokott meghasalni >---------------------')
    
    # sz??molja ki a becsl??st
    start_time_pred = time.time()
    pred = mlp.predict(data_reader.x_train)
    # print(pred)
    end_time_pred   = time.time()
    eval_time_pred  = end_time_pred - start_time_pred

    print('----------------------< ' + bcolors.WARNING + 'eval_time_pred '  + str(round(eval_time_pred, 3)) + bcolors.ENDC + '   >---------------------')

    print('----------------------< k??l??n sz??lon fut most  >---------------------')

    print('----------------------< t??l??lte a predcitiont  >---------------------')

    print('----------------------< trader elkezdte m??rni  >---------------------')
    
    # m??rje vissza a hib??t, sz??molja ki a kereseked??seket
    start_time_trader = time.time()
    result = trader.calculator(pred)
    print(result)
    end_time_trader   = time.time()
    eval_time_trader  = end_time_trader - start_time_trader

    print('----------------------< ' + bcolors.WARNING + 'eval_time_trader '  + str(round(eval_time_trader, 3)) + bcolors.ENDC + ' >---------------------')

    print('----------------------< trader befejezte m??rni >---------------------')

    del(trader)                                                                       # <-- Check

    del(pred)                                                                         # <-- Check
    
    # ki k??ne venni a resultb??l, csak a 'gain' ??rt??ket
    gain = result.get('gain')

    del(result)                                                                       # <-- Check

    gc.collect()                                                                      # <-- Check


    # a k??ldend?? csomagbe be kell tenni a model_id-t is
    print('\n\n\n\n\n !!!!!!!!!!!!!!ezzel a model_id-el fogjuk visszak??ldeni a csomagot', model_id, '\n\n\n\n\n\n')
    
    # k??ldj??k el az eredm??nyt a Drivernek, Dev: tov??bb lehet fejleszteni, hogy az eg??sz resultot k??ldje el
    data_sender.send_to_driver(data=gain, model_id=model_id)
    # Most az van, hogy mivel a 'send_to_driver' m??g ??res, meg kell vizsg??lnom, hogy lehet ??tk??ldeni egy rest apinak egy ??rt??ket
    # ez most egy kis olvas??s.
    # Ha ez megvan akkor itt csin??lni a Workeren egy olyan v??gponotot amire tudok ??rt??keket k??ldeni ??s ki is tudja onnan olvasni
    # letesztelni, hogy megy egy, majd azt bent hagyni mint test eset
    # ??s ut??na implement??lni ide, hogy tudjon k??ldeni adatot a drivernek
    
    
    
    print('-------------------------------EVALUATE MODEL DONE-------------------')
    
    
    
  
    
    

    
    
    
    
    

print('---------------------------------------------------------------')
print('                       FLASK                                   ')
print('---------------------------------------------------------------')

# https://flask.palletsprojects.com/en/2.0.x/patterns/fileuploads/



from flask import Flask
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename


app = Flask(__name__)

# ------------

@app.route('/')
def index():
    return 'Web App with Python Flask!'


def restart_waitress():
    print("Waitress is going to kill")
    time.sleep(1)
    os.system('kill -9 $(pgrep waitress) ; waitress-serve --port=8080 --call worker:create_app')


@app.route('/update')
def update():
    print('-------------------------------GIT PULL------------------------------')
    os.system('ls -la')
    os.system('git pull')
    # os.system('kill -9 $(pgrep waitress) ; waitress-serve --port=8080 --call worker:create_app')
    t1 = threading.Thread(target=restart_waitress, args=[])
    t1.start()

    # ps -aux
    # ubuntu /home/ubuntu/worker/bin/python /home/ubuntu/worker/bin/waitress-serve --port=8080 --call worker:create_app

    print('-------------------------------GIT PULL DONE-------------------------')
    # return 'Web App with Python Flask!'
    return '', 204


@app.route('/testpoint', methods=['GET'])
def testpoint():
    received_value = request.args.get('value')
    print('---------------------------------')
    print('received_value =', received_value)
    print('---------------------------------')
    return 'Web App with Python Flask!'


@app.route('/setup', methods=['GET'])
def initialize_params(_parameters=parameters, _worker_id=123, _nRowsRead=3000, _window=20, _threshold = -1000.0):
    '''
    Bemenete az a Parameters objektum amit a program elej??n l??trehoztunk,
    illetve azok az ??rt??kek amelyekre be akarjuk ezt ??ll??tani
    '''
    print('-------------------------------SETUP---------------------------------')

    received_driver_ip = (str)(request.args.get('driver_ip'))
    received_worker_id = (int)(request.args.get('worker_id'))
    received_nRowsRead = (int)(request.args.get('nRowsRead'))
    received_window    = (int)(request.args.get('window'))
    received_threshold = (float)(request.args.get('threshold'))

    print('received_driverip  =', received_driver_ip)
    print('received_workerid  =', received_worker_id)
    print('received_nRowsRead =', received_nRowsRead)
    print('received_window    =', received_window)
    print('received_threshold =', received_threshold)

    _driver_ip = received_driver_ip
    _worker_id = received_worker_id
    _nRowsRead = received_nRowsRead
    _window = received_window
    _threshold = received_threshold

    
    print('-------------------------------SETUP +-------------------------------')
    
    print('_driver_ip =', _driver_ip)
    print('_worker_id =', _worker_id)
    print('_nRowsRead =', _nRowsRead)
    print('_window    =', _window)
    print('_threshold =', _threshold)

    print(type(_driver_ip))
    print(type(_worker_id))
    print(type(_nRowsRead))
    print(type(_window))
    print(type(_threshold))
    
    _parameters.set_driver_ip(_driver_ip)
    _parameters.set_worker_id(_worker_id)
    _parameters.set_nRowsRead(_nRowsRead)
    _parameters.set_window(_window)
    _parameters.set_threshold(_threshold)
    
    print(_parameters)
    
    print('-------------------------------SETUP DONE----------------------------')
    
    # Nem kell return??lnie nem akark visszakapni semmit ez egy setter
    # De ez a  be??ll??tott params legyen glob??lis hogy mindenki l??ssa
    global parameters 
    parameters = _parameters
    
    return 'initialize_params method has been called'


@app.route('/initilaize')
def initialize(_parameters=parameters):
    initialize_worker(_parameters=_parameters)
    return 'Worker initilize method has been called'


@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      m = request.files['model_id']
      file_name = f.filename
      model_id = '9876'
      if( file_name != 'model.joblib'):  # todo vissza??rni
        print('received_file_name = ', f.filename)
        print(type(f.filename))
        cut_file_name = file_name[5:]      # model.joblib -> .joblib
        print('jobbr??l = ', cut_file_name)
        cut_file_name = cut_file_name[:-7] # .joblib -> ''
        print('ballr??l = ', cut_file_name)
        print('len(cut_file_name = ', len(cut_file_name))
        model_id = cut_file_name
        if(len(cut_file_name) == 0):
          print(' A cut_file_name hossza 0')
          model_id = '8765'
      print('??gy k??ne, hogy mindig egy adott n??ven metse le. model.joblib mondjuk')
      f.filename = 'model.joblib'
      print(f.filename)
      f.save(secure_filename(f.filename))
      
      # load model
      mlp = load_model()
      
      # el lehetne kezdeni kisz??molni ez alapj??n az eredm??nyt
      # evaluate_model(mlp, model_id)
      # ha ??gy h??vom akkor megv??rja a kisz??m??t??st ??s fogja a process-t ez??rt amikor a Driver r??h??v akkor v??r a v??laszra
      # megold??s az al??bbi k??d amire kicser??ltem.
      
      # na ezt most kiteszem egy k??l??n sz??lra
      thread = threading.Thread(target=evaluate_model, args=(mlp,model_id,))
      thread.start()
      # https://zoltan-varadi.medium.com/flask-api-how-to-return-response-but-continue-execution-828da40881e7

      return 'file uploaded successfully'




@app.route('/clean')
def clean_api():
  global data_reader
  print('-------------------------------CLEAN --------------------------------')
  print(locals())
  print('---------------------------------------------------------------------')
  print(globals())
  print('-------------------------------CLEAN --------------------------------')
  if 'data_reader' in globals():
    print('Ahoz k??pest, hogy l??tnie k??ne m??gsem l??tja')
    data_reader.clean()
  # data_reader.clean()
  gc.collect()
  print('-------------------------------CLEAN +-------------------------------')
  if 'data_reader' in globals():
    # global data_reader
    del data_reader
    print('----> del data_reader, ----> data_reader has been deleted')
  gc.collect()
  print('-------------------------------CLEAN ++------------------------------')
  return 'Worker clean method has been called'







from waitress import serve
# serve(app, host="0.0.0.0", port=8080)
# serve(app, host="127.0.0.1", port=8080)
# serve(app, host="192.168.0.247", port=8080)

# app.run(host='127.0.0.1', port=8080, DEBUG=True)
# app.run(host='192.168.0.247', port=8080, DEBUG=True)

def create_app():
   return app

# -------------------------------------------
# ezt majd ??gy kell elind??tani
# waitress-serve --port=8080 --call hello:create_app
#
# https://stackoverflow.com/questions/51025893/flask-at-first-run-do-not-use-the-development-server-in-a-production-environmen
#
# igy lehet ellen??rizni, hogy fut-e a flask
# nc -vz 192.168.0.247 8080
# -------------------------------------------

