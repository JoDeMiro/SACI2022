"""Data Reader

This script allows the user to read the data from a Comma Sep Value
in the Pandas DataFrame. It is assumed that the first row of the 
CSV is the location of the columns.

This tool accepts comma separated value files (.csv) as well as excel
(.xls, .xlsx) files.

This script requires that `pandas` be installed within the Python
environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * read_csv - sets and returns the loaded pandas.DataFrame
    * info_pdf - print out information about the loaded DataFrame
    * create_dataset - sets the Close price and creates and sets a new DataFrame
    * info_dataset
    * scale_dataset - transform the data between a range
    * create_training_dataset - this is where the lagged columns are created
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------------------

class DataReader():
  """
  DataReader is responsible for holding, store the data and the basic
  data manipulation.
  
  The data flow is the following:
  ------------------------------
  
  csv -> pdf -> dataset -> scaled_dataset -> (x_train, y_train, x_train_reshaped)
  
  ------------------------------------------------------------------------------
  
  Attributes
  ----------

  nRowsRead : int
      the number of rows that is read from the csv
  window : int
      the number of the lookback period the dataset has (default 10)
  dataset : pandas.DataFrame
      a dataframe object that stores the data readed from csv
  pdf : pd.DataFrame
      The data frame which stores the read data from csv file
  df : pd.DataFrame
      The data frame which stores only Dates and BC values as pandas.series
  dataset : pd.DataFrame
      The data frame which stores only Dates and BC values as numpy.ndarray
  dataset_full : pd.DataFrame
      * For future use
  n_row : int
      Number of rows in the loaded csv file
  n_col : int
      Number of columns in the loaded csv file
  scaler : sklearn.preprocessing._data.MinMaxScaler
      MinMaxScaler transforms the whole dataset into a range by column wise.
  scaler_full : sklearn.preprocessing._data.MinMaxScaler
      * For future use
  scaled_dataset : pd.DataFrame
      Scaled data frame by column wise
  scaled_dataset_full : pd.DataFrame
      * For future use
  x_train : np.ndarray
      As far as constitue number of the input variables from the window_size,
      the shape of this numpy array is (number of cases, window_size)
  y_train : np.ndarray
      As far as keep the number of the output variable one, the shape of
      this numpy array is (number of cases, )
  x_train_reshaped : np.ndarray
      Transform the x_train to another nd.array in apropirate format
      for Scikit and Keras (number of cases, window_size, 1)

  Methods
  -------
  read_csv(path='./input/eurusd_minute.csv')
      Create and set a new dataframe from csv file
  info_pdf()
      Print out the basic properties of the csv file
  create_dataset()
      Create and set a new dataset from the csv file
  info_dataset()
      Print out the basic properties of the dataset
  scale_dataset()
      Create and set a new scaled_dataset from dataset
  set_window()
      Set the lookback window size
  create_training_dataset()
      Create and set x_train and y_train datasets
      Create and set x_train_train_reshaped dataset
      Until this point we have only one (or more columns)
      But this method creates the lag columns (or arrays) as well
  
  """

  def __init__(self, nRowsRead: int):
    """
    Parameters
    ----------
    nRowsRead : int
        The number of rows will be read from the csv file
    """
    self.nRowsRead    = nRowsRead
    
    self.pdf          = None
    self.df           = None
    self.dataset      = None
    self.dataset_full = None
    self.window       = None
    self.n_row        = None
    self.n_col        = None
    self.scaler       = None
    self.scaler_full  = None
    self.scaled_dataset = None
    self.scaled_dataset_full = None
    self.x_train      = None
    self.y_train      = None
    self.x_train_reshaped = None
    

# ------------------------------------------------------------------------------

  def read_csv(self, path = './input/eurusd_minute.csv') -> pd.DataFrame:
    """
    Read and store csv as Pandas DataFrame.
    
    If the argument `path` isn't passed in, the default path
    is used.
        
    Parameters
    ----------
    path : str, optional
        The path of the csv file which will be loaded.

    Raises
    ------
    NotImplementedError
        If no sound is set for the animal or passed in as a
        parameter.

    Returns
    -------
    pd.DataFrame
        Description of return value
        Date   Time   BO    BH    BL    BC    BCh   AO    AH    AL    AC    ACh
        
        ==========  ==============================================================
        Date        Date (2005-01-02)
        Time        Time (18:29)
        BO          Bid Open Price (1.3555)
        BH          Bid High Price (float)
        BL          Bid Low Price (float)
        BC          Bid Close Price (float)
        BCh         (fogalmam sincs)
        AO          Ask Open Price (float)
        AH          Ask High Price (float)
        AL          Ask Low Price (float)
        AC          Ask Close Price (float)
        ACh         (fogalmam sincs)
        ==========  ==============================================================

    """
    
    self.pdf = pd.read_csv(path, delimiter=',', nrows = self.nRowsRead)             # <-- read csv to pandas dataframe
    self.pdf.dataframeName = 'eurusd_minute.csv'                                    # <-- set dataframeName

    self.n_row, self.n_col = self.pdf.shape
    print(f'There are {self.n_row} rows and {self.n_col} columns')                  # <-- print nRow, nCol
    print(self.pdf.head(5))                                                         # <-- print head
    self.println()

    return self.pdf

# ------------------------------------------------------------------------------

  def info_pdf(self):
    """
    Print out the basic information about the 'BC' column of loaded dataframe.
    """
    print('Info:')
    print('pdf.BC.values.ctypes   =\t', self.pdf.BC.values.ctypes)
    print('pdf.BC.values.dtype    =\t', self.pdf.BC.values.dtype)
    print('pdf.BC.values.itemsize =\t', self.pdf.BC.values.itemsize)
    print('pdf.BC.values.nbytes   =\t', self.pdf.BC.values.nbytes, 'Byte')
    print('pdf.BC.values.nbytes   =\t', self.pdf.BC.values.nbytes / 1024, 'Kbyte')
    self.println()

# ------------------------------------------------------------------------------

  def create_dataset(self):
    """
    Create dataset and dataset_full.
    The dataset contains only the Date as index of pd.DataFrame. <pandas.core.frame.DataFrame>
    and df['BC'] column as the Bid Closing Price.
    The dataset_full contans all variable (or columns) comes from the csv file.
    
    In both cases the data must be transformet from <pandas.core.frame.DataFrame>
    into <numpy.ndarray>.
    
    This method creates two new class variable namely the dataset and dataset_full
    which are numpy.ndarray.
    
    The column 'Time' will be stored as 'object', the others will be 'float64'.
    """
    
    self.df = self.pdf                                     # <-- copy pdf to a new df to preserve the original pdf

    # For BC column only
    self.df.set_index(['Date'], inplace=True)              # <-- set index column
    # self.data = self.df.filter(['BC'])                     # <-- select only the BC column
    # self.dataset = self.data.values                        # --> from pandas.series -> numpy.ndarray
    self.dataset = self.df.filter(['BC']).values           # <-- the above two steps in one

    # For all column (I do not use it)
    self.dataset_full = self.df.filter(['BO', 'BH', 'BL', 'BC', 'BCh', 'AO', 'AH', 'AL', 'AC', 'ACh']).values
                                                           # --> from pandas.series -> numpy.ndarray

# ------------------------------------------------------------------------------

  def info_dataset(self):
    """
    Print out the basic information about the transformed dataframe.
    """
    self.println()
    print('Info Dataset:')
    print('Type(self.dataset       =', type(self.dataset))
    print('Type(self.dataset_full  =', type(self.dataset_full))
    print('Shape(self.dataset)     =', self.dataset.shape)
    print('Shape(self.dataset_full =', self.dataset_full.shape)
    self.println()

# ------------------------------------------------------------------------------

  def scale_dataset(self, minimum = -1, maximum = 1):
    """
    This method creates two <sklearn.preprocessing._data.MinMaxScaler>
    `scaler` is for the ['BC']
    `sclaer_full` is for all columns not just for ['BC']
    
    The columns are scaled into the range independently.

    This method also transforms and stores the scaled columns (numpy.ndarrays)
    into a new class variable:
    - `scaled_dataset`
    - `scaled_dataset_full`
    
    Parameters
    ----------
    minimum : float, optional
        The minimum of the new minimum value.
    maximum : float, optional
        The maximum of the new minimum value.
    """
    
    # Bug:
    # The windowed data should be normalized respectivley, not the whole dataframe.

    self.scaler           = MinMaxScaler(feature_range=(minimum, maximum))           # <-- (0, 1) vagy (-1, 1)
    self.scaled_dataset   = self.scaler.fit_transform(self.dataset)

    self.scaler_full         = MinMaxScaler(feature_range=(minimum, maximum))        # <-- (0, 1) vagy (-1, 1)
    self.scaled_dataset_full = self.scaler_full.fit_transform(self.dataset_full)

# ------------------------------------------------------------------------------

  def set_window(self, window_size: int):
    """
    Set the lookback window size.
    - param window_size (int): The lookback period.
    """
    if type(window_size) is int:
      self.window = window_size
      print('Set window =', self.window)
    else:
      print('Error: Not proper type of parameter')
    self.println()

# ------------------------------------------------------------------------------

  def create_training_dataset(self):
    """
    Creates the train and the test dataset as `x_train` and `y_train` and `x_train_reshaped`.
    """
    
    print('Create Training Dataset:')
    
    self.x_train = []                                                               # <-- Create list of Windows
    self.y_train = []

    for i in range(self.window, len(self.scaled_dataset)):
        self.x_train.append(self.scaled_dataset[i-self.window:i, 0])
        self.y_train.append(self.scaled_dataset[i, 0])

    print('The train dataset {}, and the labels {}'.format(len(self.x_train), len(self.y_train)))

    self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)     # <-- Convert list to Numpy Array

    print('self.x_train.shape  = ', self.x_train.shape)
    print('self.y_train.shape  = ', self.y_train.shape)

    print('self.x_train.nbytes = ', self.x_train.nbytes/1000, 'Kbyte')
    print('self.x_train.nbytes = ', self.x_train.nbytes/1000/1000, 'Mbyte')

    self.x_train_reshaped = np.reshape(self.x_train,                                # <-- Fontos adatokat alábbi formában várjuk
                                       newshape=(self.x_train.shape[0],
                                                 self.x_train.shape[1], 1))         # <-- Reshape (970, 30) -> (970, 30, 1)
    
    print('self.x_train_reshaped.shape =', self.x_train_reshaped.shape)
    print('self.x_train.shape          =', self.x_train.shape)
    print('self.y_train.shape          =', self.y_train.shape)
    self.println()

    # --------------------------------------------------------------------------    # <-- ToDo: ezt itt hagyom de nem használom

    x_train_full = []                                                               # <-- Create list of Windows
    y_train_full = []

    for i in range(self.window, len(self.scaled_dataset_full)):
        x_train_full.append(self.scaled_dataset_full[i-self.window:i, :])
        y_train_full.append(self.scaled_dataset_full[i, 0])

    # print('The train dataset {}, and the labels {}'.format(len(x_train_full), len(y_train_full)))

    x_train_full, y_train_full = np.array(x_train_full), np.array(y_train_full)     # <-- Convert list to Numpy Array

    # print(x_train_full.shape)
    # print(y_train_full.shape)

    # print('x_train_full.nbytes = ', x_train_full.nbytes/1000, 'Kbyte')
    # print('x_train_full.nbytes = ', x_train_full.nbytes/1000/1000, 'Mbyte')

# ------------------------------------------------------------------------------

  @classmethod
  def println(self):
    "Print separation line."
    print('------------------------------------------------------------------------------')

# ------------------------------------------------------------------------------

