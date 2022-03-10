# Driver file

# a funkciók amiket meg kell valósítania

# init() lehet később

# nn generálás (a csv-t nem kell tudnia beolvasnia)

# mutáció() lehet később

# küldés a workerenek

# fogadás a workereketől

# evaluation() később

print('---------------------------------------------------------------')
print('                         SET VARIABLE                          ')
print('---------------------------------------------------------------')

driver_ip_address = '192.168.0.114'


print('---------------------------------------------------------------')
print('                         HELLO DRIVER                          ')
print('---------------------------------------------------------------')


print('---------------------------------------------------------------')
print('                         IMPORT                                ')
print('---------------------------------------------------------------')


import os
import sys
import time
import pprint
import joblib
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

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

print('Python version:{}'.format(sys.version))
print('Numpy version:{}'.format(np.__version__))
print('Pandas version:{}'.format(pd.__version__))
print('Sci-Kit Learn version:{}'.format(sklearn.__version__))



print('---------------------------------------------------------------')
print('                         INIT                                  ')
print('---------------------------------------------------------------')


# beállítunk néhány paraméter ami amit később használunk
# ezeket az értékeket sajnos a Driver oldalon is ki kell számolnunk, ha úgy csináljuk, hogy a modelt küldjük át
# fontos, hogy számos paraméter szinkronizálva legyen, különben ami model itt előáll az a Worker oldalon eltörik
arch = (2,2)                                # <-- nn(arch)
window = 21                                 # <-- bementi változók száma
nRowsRead = 7899                            # <-- ez lehet alapján csinálunk modelt, de lehet más is mint odaát
threshold = -1.0                            # <-- a trader parametére


# elő kell állítani a túloldali adatsorral azonos szart np.[970, 20] és np[970]
x_train = np.ones((970, window))
y_train = np.zeros(970)
print(x_train.shape)
print(y_train.shape)
clf = MLPRegressor(hidden_layer_sizes=arch, max_iter=3, shuffle=False, activation='tanh', random_state=1)
clf.fit(x_train, y_train)

# ---------------------------------------------------------------------------
# A joblib segítségével lementdem a diskre a modelt, hogy küldjem a workernek
# from joblib import dump, load
joblib.dump(clf, 'model.joblib')         # <-- elmenjük
clf = joblib.load('model.joblib')        # <-- betöltjük

# ---------------------------------------------------------------------------


# Valamiért újra indítás után nem müködnek de futnak a workeren mégis kill ---> újrainditás kellett terminálból
# worker_address = 'http://192.168.0.54:8080'
# worker_address = 'http://192.168.0.32:8080'
worker_address = 'http://192.168.0.247:8080'



# ---------------------------------------------------------------------------

print('---------------------------------------------------------------')
print('                         DISTRIBUTE                            ')
print('---------------------------------------------------------------')


# Setuppolni kell a paramétereket
request_params = 'nRowsRead=' + (str)(nRowsRead) + '&' + 'window=' + (str)(window) + '&' + 'threshold=' + (str)(threshold)
print(request_params)
# resp = requests.get('http://192.168.0.247:8080/setup?nRowsRead=2998&window=20&threshold=-1000')
# resp = requests.get('http://192.168.0.247:8080/setup?' + request_params)
resp = requests.get(worker_address + '/setup?' + request_params)

# Ezzel simán meghívjuk a Worker INITIALIZE REST API Végpontját
resp = requests.get(worker_address + '/initilaize')
print('initialize ', resp)

# Ezzel a módszerrel lehet átküldeni neki a joblib model filét
uploadResultUrl = worker_address + '/uploader'
filePath = "model.joblib"
fileFp = open(filePath, 'rb')
fileInfoDict = {
    "file": fileFp,
}
resp = requests.post(uploadResultUrl, files=fileInfoDict)
print('uploader   ', resp)                                # <-- ez nagyon jó ha <Response [200]> mert akkor átment a file

# Átküldök egy értéket a Workernek
resp = requests.get(worker_address + '/testpoint?value=123456789')
print('testpoint  ', resp)








print('---------------------------------------------------------------')
print('                       FLASK                                   ')
print('---------------------------------------------------------------')



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
    os.system('kill -9 $(pgrep waitress) ; waitress-serve --port=8080 --call driver:create_app')


@app.route('/update')
def update():
    print('-------------------------------GIT PULL------------------------------')
    os.system('ls -la')
    os.system('git pull')
    t1 = threading.Thread(target=restart_waitress, args=[])
    t1.start()

    # ps -aux
    # ubuntu /home/ubuntu/worker/bin/python /home/ubuntu/worker/bin/waitress-serve --port=8080 --call driver:create_app

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
def initialize_params(_parameters=parameters, _nRowsRead=3000, _window=20, _threshold = -1000.0):
    '''
    Bemenete az a Parameters objektum amit a program elején létrehoztunk,
    illetve azok az értékek amelyekre be akarjuk ezt állítani
    '''
    print('-------------------------------SETUP---------------------------------')

    received_nRowsRead = (int)(request.args.get('nRowsRead'))
    received_window    = (int)(request.args.get('window'))
    received_threshold = (float)(request.args.get('threshold'))

    print('received_nRowsRead =', received_nRowsRead)
    print('received_window    =', received_window)
    print('received_threshold =', received_threshold)

    _nRowsRead = received_nRowsRead
    _window = received_window
    _threshold = received_threshold

    
    print('-------------------------------SETUP +-------------------------------')
    
    print('_nRowsRead =', _nRowsRead)
    print('_window    =', _window)
    print('_threshold =', _threshold)

    print(type(_nRowsRead))
    print(type(_window))
    print(type(_threshold))
    
    _parameters.set_nRowsRead(_nRowsRead)
    _parameters.set_window(_window)
    _parameters.set_threshold(_threshold)
    
    print(_parameters)
    
    print('-------------------------------SETUP DONE----------------------------')
    
    # Nem kell returnölnie nem akark visszakapni semmit ez egy setter
    # De ez a  beállított params legyen globális hogy mindenki lássa
    global parameters 
    parameters = _parameters
    
    return 'initialize_params method has been called'

@app.route('/initilaize')
def initialize(_parameters=parameters):
    initialize_worker(_parameters=_parameters)
    return 'Worker initilize method has been called'

# ------------











from waitress import serve

def create_app():
   return app

# -------------------------------------------
# ezt majd így kell elindítani
# waitress-serve --port=8080 --call driver:create_app
#
# https://stackoverflow.com/questions/51025893/flask-at-first-run-do-not-use-the-development-server-in-a-production-environmen
#
# igy lehet ellenőrizni, hogy fut-e a flask
# nc -vz 192.168.0.247 8080
# -------------------------------------------

