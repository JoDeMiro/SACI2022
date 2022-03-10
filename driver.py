# Driver file

# a funkciók amiket meg kell valósítania

# init() lehet később

# nn generálás (a csv-t nem kell tudnia beolvasnia)

# mutáció() lehet később

# küldés a workerenek

# fogadás a workereketől

# evaluation() később

# Fut a Flask, képes fogadni adatot a testpoint végponton
# Most meg kéne tudnom szólítani valami mást is



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
print('                         CREATE DAO                            ')
print('---------------------------------------------------------------')

class Parameters():
    
    def __init__(self):
        self.generation = 1000
        self.factor = 2
        self.dummy = -0.0
    
    def set_generation(self, _generation):
        self.generation = _generation
    
    def set_factor(self, _factor):
        self.factor = _factor
    
    def set_dummy(self, _dummy):
        self.dummy = _dummy
    
    def __str__(self):
        return 'Parameters Class(generation=' + str(self.generation) + ' ,factor=' + str(self.factor) + ', dummy=' + str(self.dummy) + ')'

parameters = Parameters()
print(parameters)




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


def call_worker_setup(nRowsRead, window, threshold):

	# Setuppolni kell a paramétereket
	request_params = 'nRowsRead=' + (str)(nRowsRead) + '&' + 'window=' + (str)(window) + '&' + 'threshold=' + (str)(threshold)
	print(request_params)
	# resp = requests.get('http://192.168.0.247:8080/setup?nRowsRead=2998&window=20&threshold=-1000')
	# resp = requests.get('http://192.168.0.247:8080/setup?' + request_params)
	resp = requests.get(worker_address + '/setup?' + request_params)
	print(resp)
	print('_______call_worker_setup_______')


def call_worker_initialize():

	# Ezzel simán meghívjuk a Worker INITIALIZE REST API Végpontját
	resp = requests.get(worker_address + '/initilaize')
	print('initialize ', resp)
	print('_______call_worker_initialize_______')


def call_worker_uploader():

	# Ezzel a módszerrel lehet átküldeni neki a joblib model filét
	uploadResultUrl = worker_address + '/uploader'
	filePath = "model.joblib"
	fileFp = open(filePath, 'rb')
	fileInfoDict = {
	    "file": fileFp,
	}
	resp = requests.post(uploadResultUrl, files=fileInfoDict)
	print('uploader   ', resp)
	print('_______call_worker_uploader_______')


def call_worker_testpoint():
	# Átküldök egy értéket a Workernek
	resp = requests.get(worker_address + '/testpoint?value=123456789')
	print('testpoint  ', resp)
	print('_______call_worker_testpoint_______')








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


# ez egy bejövő testpoint, kap egy értéket akkor kiírja
@app.route('/testpoint', methods=['GET'])
def testpoint():
    received_value = request.args.get('value')
    print('---------------------------------')
    print('received_value =', received_value)
    print('---------------------------------')
    return 'Web App with Python Flask!'


# call_worker_testpoint -> ha erre jön kérés akkor ez tovább hív a worker testpoint-jára
@app.route('/calltestpoint', methods=['GET'])
def call_worker_testpoint_api():
    call_worker_testpoint()
    print('______ráhívtunk a worker testpoinjára ott kell hogy lefusson valami______')
    return 'Called woreker testpoint'


# a Driver programot setupoljuk vele
@app.route('/setup', methods=['GET'])
def initialize_params(_parameters=parameters, _generation=3000, _factor=20, _dummy = -1000.0):
    '''
    Bemenete az a Parameters objektum amit a program elején létrehoztunk,
    illetve azok az értékek amelyekre be akarjuk ezt állítani
    '''
    print('-------------------------------SETUP---------------------------------')

    received_generation = (int)(request.args.get('generation'))
    received_factor     = (int)(request.args.get('factor'))
    received_dummy      = (float)(request.args.get('dummy'))

    print('received_generation =', received_generation)
    print('received_factor     =', received_factor)
    print('received_dummy      =', received_dummy)

    _generation = received_generation
    _factor = received_factor
    _dummy = received_dummy

    
    print('-------------------------------SETUP +-------------------------------')
    
    print('_generation =', _generation)
    print('_factor     =', _factor)
    print('_dummy      =', _dummy)

    print(type(_generation))
    print(type(_factor))
    print(type(_dummy))
    
    _parameters.set_generation(_generation)
    _parameters.set_factor(_factor)
    _parameters.set_dummy(_dummy)
    
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

