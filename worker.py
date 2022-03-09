# Worker file

class Parameters():
    
    def __init__(self):
        self.nRowsRead = 1000
        self.window = 2
        self.threshold = -0.0
    
    def set_nRowsRead(self, _nRowsRead):
        self.nRowsRead = _nRowsRead
    
    def set_window(self, _window):
        self.window = _window
    
    def set_threshold(self, _threshold):
        self.threshold = _threshold
    
    def __str__(self):
        return 'Parameters Class(nRowsRead=' + str(self.nRowsRead) + ' ,window=' + self.y + ', threshold=' + self.threshold + ')'

parameters = Parameters()
print(parameters)



print('---------------------------------------------------------------')
print('                         HELLO WORKER                          ')
print('---------------------------------------------------------------')


print('---------------------------------------------------------------')
print('                         IMPORT                                ')
print('---------------------------------------------------------------')


import os
import sys
import pprint
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.neural_network import MLPRegressor

print('Python version:{}'.format(sys.version))
print('Numpy version:{}'.format(np.__version__))
print('Pandas version:{}'.format(pd.__version__))
print('Sci-Kit Learn version:{}'.format(sklearn.__version__))


print('---------------------------------------------------------------')
print('                       UNZIP CSV DATA                          ')
print('---------------------------------------------------------------')

os.system('gzip -f -d ./Input/eurusd_minute_1000000.csv.gz')
os.system('mv ./Input/eurusd_minute_1000000.csv ./Input/eurusd_minute.csv')


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
    self.window = NonUniformImage

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

    # For all column (I do not use it)
    data_full = self.df.filter(['BO', 'BH', 'BL', 'BC', 'BCh', 'AO', 'AH', 'AL', 'AC', 'ACh'])
    self.dataset_full = data_full.values                   # --> from pandas.series -> numpy.ndarray

# ------------------------------------------------------------------------------

  def show_dataset_info(self):
    self.println()
    print(type(self.dataset))
    print(type(self.dataset_full))
    print(self.dataset.shape)
    print(self.dataset_full.shape)
    self.println()

# ------------------------------------------------------------------------------

  def normalize_values(self):
    
    self.scaler        = MinMaxScaler(feature_range=(-1, 1))                        # <-- (0, 1) vagy (-1, 1)
    self.scaled_data   = self.scaler.fit_transform(self.dataset)

    self.scaler_full        = MinMaxScaler(feature_range=(-1, 1))                   # <-- (0, 1) vagy (-1, 1)
    self.scaled_data_full   = self.scaler_full.fit_transform(self.dataset_full)

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
    self.x_train = []                                                               # <-- Create list of Windows
    self.y_train = []

    for i in range(self.window, len(self.scaled_data)):
        self.x_train.append(self.scaled_data[i-self.window:i, 0])
        self.y_train.append(self.scaled_data[i, 0])

    print('The train dataset {}, and the labels {}'.format(len(self.x_train), len(self.y_train)))

    self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)     # <-- Convert list to Numpy Array

    print(self.x_train.shape)
    print(self.y_train.shape)

    print('x_train.nbytes = ', self.x_train.nbytes/1000, 'Kbyte')
    print('x_train.nbytes = ', self.x_train.nbytes/1000/1000, 'Mbyte')

    self.x_train_reshaped = np.reshape(self.x_train,                                # <-- Nagyon fontos az adatokat az alábbi formában várjuk
                                       newshape=(self.x_train.shape[0],
                                                 self.x_train.shape[1], 1))         # <-- Reshape (970, 30) -> (970, 30, 1)
    
    print(self.x_train_reshaped.shape)
    print(self.x_train.shape)
    print(self.y_train.shape)
    self.println()


    # --------------------------------------------------------------------------    # <-- ToDo: ezt itt hagyom de nem használom

    x_train_full = []                                                               # <-- Create list of Windows
    y_train_full = []

    for i in range(self.window, len(self.scaled_data_full)):
        x_train_full.append(self.scaled_data_full[i-self.window:i, :])
        y_train_full.append(self.scaled_data_full[i, 0])

    # print('The train dataset {}, and the labels {}'.format(len(x_train_full), len(y_train_full)))

    x_train_full, y_train_full = np.array(x_train_full), np.array(y_train_full)     # <-- Convert list to Numpy Array

    # print(x_train_full.shape)
    # print(y_train_full.shape)

    # print('x_train_full.nbytes = ', x_train_full.nbytes/1000, 'Kbyte')
    # print('x_train_full.nbytes = ', x_train_full.nbytes/1000/1000, 'Mbyte')

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
    Létrehoz egy data readder objektumot előkészíti és azt adja vissza
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
    'Init Scikit Learn MLPRegressor'                                                 # <-- hogy létre jöjjenek a súlyok inicializálni kell
    
    np.random.seed(1)

    mlp = MLPRegressor(hidden_layer_sizes=(_first, _second),
                      activation='tanh',                                             # -------> ha (MinMax(-1,1) vagy StandardScaler())
                      solver='sgd',
                      batch_size=100000,                                             # <<-- v.017 bug fixed
                      max_iter=1,                                                    # <-- sajnos legalább 1 kell hogy legyen
                      shuffle=False,
                      random_state=1,
                      learning_rate_init=0.00000001,                                 # >- lehetőleg ne tanuljon semmit GD alapján
                      validation_fraction=0.0,
                      n_iter_no_change=99999999)
    
    # ----->                                                         Behoztam ide az első illesztést is, hogy meglegyenek neki a súlyok

    np.random.seed(1)

    y_random = np.zeros((self.y_train.shape[0])) * 110.01                           # --> tök random adaton tanítom, hogy még véletlenül se tanuljon

    mlp.fit(self.x_train, y_random)                                                 # --> nem akarjuk mi semmire megtanítani csak kell az inithez

    self.mlp = mlp
    
    return mlp

# ------------------------------------------------------------------------------

  def create_prediction(self):
    'Saját adati alapján csinája meg a predcitiont'

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

test_pred = mlp.predict(data_reader.x_train)                                        # teszt pred --> semmire nem fogjuk használni

print(test_pred[0:5])

test_pred = nn.mlp.predict(data_reader.x_train)                                     # init után direktben is el lehet érni az mlp.predict() függvényt

print(test_pred[0:5])

test_pred = nn.create_prediction()                                                  # csináltam neki egy saját fügvényt ami elvégzi az egész predictiont

print(test_pred[0:5])

# ------------------------------------------------------------------------------

# A scikit-től eltérő súlyincializácóra van lehetőség létre hoztam rá egy
# eljárást de nem használom. Ha mégis használni szeretném akkor a
# SACI22 - 018.ipynb-ben megtalálható

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

      if i == lenght - 1 and is_in_trade == True:                                       # <-- le kell zárni az utolsónál a vételt ha nyitva van
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


print('---------------------------------------------------------------')
print('                       TRADER CLASS TEST                       ')
print('---------------------------------------------------------------')


# ------------------------------------------------------------------------------    # <-- Test Trader Class

new_trader = Trader(threshold = -0.0, data_reader = data_reader)                    # <-- Ebben a formában kell majd használni


pred = nn.create_prediction()

print(pred[0:5])

result = new_trader.calculator(pred)                                                # <-- Ebben a formában kell majd használni

result.keys()                           # <- get dict.keys
result.get('gain')                      # <- get a value by a given key
result['gain']                          # <- get a value by a given key             # <-- Ebben a formában kell majd használni

# ------------------------------------------------------------------------------    # <-- Test has successed


# A Trader osztályon még számos kereskedési eredményt lehet hadsznlni,
# de ezeket most nem használom. Ha érdekel akkor a SACI22 - 018.ipynb
# fájlban meg lehet találni


# ------------------------------------------------------------------------------







print('---------------------------------------------------------------')
print('                       RANDOMER CLASS                          ')
print('---------------------------------------------------------------')

# Na ezt a Driver fogja csinálni




print('---------------------------------------------------------------')
print('                       FÜGGVÉNYEK A FLASKHEZ                   ')
print('---------------------------------------------------------------')

from sklearn.neural_network import MLPRegressor
import numpy as np
import pickle
import joblib
import json
import os

# inkább úgy kéne megcsinálni, hogy van egy Parameters Class amit példányosítunk a prgoram elején
# Ezt úgy hozzuk létre, hogy csinálunk egy külön REST API-t a Parameters példányosítására és az értékeire
# Ezt a példányt beállítjuk a programnak és utána csak settelni kell


def initialize_worker(parameters):
    # Ide kerül minden amit egyszer meghív a driver aminek a hatására lefut ez a rész és létjrejönnek
    # azok az objektumok amiket a teljes kisérlet során fog használni a worker.
    # A Trader, Az NN, A DataReader
    
    
    print('# Initialize Worker')
    
    # A parameters objektumból olvassuk ki a parametereket
    
    _nRowsRead = parameters._nRowsRead
    _window = parameters._window
    _threshold = parameters._threshold
    
    data_reader = initialize_data_reader(_nRowsRead=_nRowsRead, _window=_window)                # <-- Initialize data_reader
    
    trader = Trader(threshold = _threshold, data_reader = data_reader)                    # <-- Initialize trader
    # A tradert egyébként minden egyes számításnál inicializálni kell szóval nem ide kerül, de most itt hagyom




    
    

def load_model():
    clf = joblib.load('model.joblib')                                   # <-- betöltjük a modlet a filéből
    print('# Model betöltve a joblib-ből')
    print(clf.get_params())

# most az jön, hogy ki kéne számolnia a workernek, hogy milyen eredményt adna az adott modell a belolvasott adatokon

def evaluate_model():
    '''
    Az evaluate_model() függvénynek akkor kell lefutnia amikor kap egy modelt a worker a drivertől
    kívűlről, másképp nem hívható.
    '''
    print('# Model evaluation on Worker')
    
    # ki kell keresnem, hogy mi kerül ide
    # hogy zajlik az eredeti folyamat
    # milyen osztályokat és példányokat olvas be a worker, hogy ez a folyamat megvalósuljon
    # data_reader, trader, ...
    # ellenőrizni, hogy valóban létre jöttek-e ezek a példányok, és valóban tárolják-e a hozzájuk tartozó információt
    
    # Az initialize() megcsinálta nekünk a data_reader és a trader objektumokat
        
    # A data_reader objektumot csak egyszer kell létrehozni, de a trader-t azt minden futásnál, ugyhogy példányosítsunk egyet
    

    # Kapja meg a paramétereket a Parameters objektumból
    
    trader = Trader(threshold = -1.0, data_reader = data_reader)                        # <-- create a Trader (0.0 just a random choise)

    
    
    
    
    

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


@app.route('/setup')
def initialize_params(parameters=parameters, _nRowsRead=1000, _window=2, _threshold = -0.0):
    '''
    Bemenete az a Parameters objektum amit a program elején létrehoztunk,
    illetve azok az értékek amelyekre be akarjuk ezt állítani
    '''
    
    print('-------------------------------SETUP---------------------------------')
    
    parameters.set_nRowsRead = _nRowsRead
    parameters.set_window = _window
    parameters.set_threshold = _threshold
    
    print(parameters._nRowsRead)
    print(parameters._window)
    print(parameters._threshold)
    
    print('-------------------------------SETUP DONE----------------------------')
    
    # Nem kell returnölnie nem akark visszakapni semmit ez egy setter
    return 'initialize_params method has been called'

@app.route('/initilaize')
def initialize(parameters=parameters):
    initialize_worker()
    return 'Worker initilize method has been called'

# ------------

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      
      # load model
      load_model()
      
      # el lehetne kezdeni kiszámolni ez alapján az eredményt
      
      return 'file uploaded successfully'

# ------------

# Feltöltöttük a filét valamit kezdeni kéne vele mondjuk töltsük be.










from waitress import serve
# serve(app, host="0.0.0.0", port=8080)
# serve(app, host="127.0.0.1", port=8080)
# serve(app, host="192.168.0.247", port=8080)

# app.run(host='127.0.0.1', port=8080, DEBUG=True)
# app.run(host='192.168.0.247', port=8080, DEBUG=True)

def create_app():
   return app

# -------------------------------------------
# ezt majd így kell elindítani
# waitress-serve --port=8080 --call hello:create_app
#
# https://stackoverflow.com/questions/51025893/flask-at-first-run-do-not-use-the-development-server-in-a-production-environmen
#
# igy lehet ellenőrizni, hogy fut-e a flask
# nc -vz 192.168.0.247 8080
# -------------------------------------------

