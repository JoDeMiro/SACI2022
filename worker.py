# Worker file

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
# os.system('wget https://raw.githubusercontent.com/JoDeMiro/Micado-Research/main/MLPPlot.py')

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



