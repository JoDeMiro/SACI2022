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

driver_ip_address = 'http://192.168.0.114:8080'

workers_addresses = ['http://192.168.0.54:8080/', 'http://192.168.0.32:8080/', 'http://192.168.0.247:8080/']

workers = []

for i,v in enumerate(workers_addresses):
  print(i, v)
  dic = {'id': i,'add': v}
  workers.append(dic)
  # print(dic.get('id'))
  # print(dic.get('add'))
workers



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

from randomer import Randomer
from copy import deepcopy


###
new_randomer = Randomer(1)
##


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







# A Driver program néhány változóját hozzuk létre vele, pl az első neurális hálót amit el küld majd
def initialize_driver():

	print('---------------------------------------------------------------')
	print('                         INITIALIZE_DRIVER()                   ')
	print('---------------------------------------------------------------')


	# beállítunk néhány paraméter ami amit később használunk
	# ezeket az értékeket sajnos a Driver oldalon is ki kell számolnunk, ha úgy csináljuk, hogy a modelt küldjük át
	# fontos, hogy számos paraméter szinkronizálva legyen, különben ami model itt előáll az a Worker oldalon eltörik
	arch = (2,2)                                # <-- nn(arch)
	global window 
	window = 21  
	global nRowsRead                            # <-- bementi változók száma
	nRowsRead = 7899
	global threshold                            # <-- ez lehet alapján csinálunk modelt, de lehet más is mint odaát
	threshold = -1.0                            # <-- a trader parametére


	# elő kell állítani a túloldali adatsorral azonos szart np.[970, 20] és np[970]
	x_train = np.ones((970, window))
	y_train = np.zeros(970)
	print(x_train.shape)
	print(y_train.shape)

	global clf
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

	# Itt kell majd megadnom egy listában a workerek címeit amin majd a küldésnél végig itereál

	# ---------------------------------------------------------------------------

	# A mutációhoz szükséges példányosítanom egy Randomer-t (randomer.py)

	global randomer
	randomer = Randomer(1)                                         # <-- create a Randomer to controll the mutation

	# - van egy randomize(self, coefs, factor = 1000) függvénye ezt kell majd hivogatni a mutációhoz

	# ---------------------------------------------------------------------------





# Egy konkrét workeren a setup rest hivása
def call_worker_setup(driver_ip, worker_id, worker_address, nRowsRead, window, threshold):

	print('---------------------------------------------------------------')
	print('                      CALL_WORKER_SETUP()                      ')
	print('---------------------------------------------------------------')

	# Setuppolni kell a paramétereket
	request_params = 'driver_ip=' + (str)(driver_ip) + '&worker_id=' + (str)(worker_id) + '&' + 'nRowsRead=' + (str)(nRowsRead) + '&' + 'window=' + (str)(window) + '&' + 'threshold=' + (str)(threshold)
	print(request_params)
	resp = requests.get(worker_address + '/setup?' + request_params)
	print(resp)
	print('_______call_worker_setup_______')


# Az összes workeren lefut a setup
def setup_workers():
	print('---------------------------------------------------------------')
	print('                      SETUP_WORKERS()                          ')
	print('---------------------------------------------------------------')
	print('workers_addresses = ', workers_addresses)
	for worker in workers:
		print('---------------------------------------------------------------------')
		print('diver_ip =', driver_ip_address)
		print('worker_id =', worker.get('id'))
		print('worker_address =', worker.get('add'))
		print('nRowsRead =', nRowsRead)
		print('window    =', window)
		print('threshold =', threshold)
		print('---------------------------------------------------------------------')

		call_worker_setup(driver_ip_address, worker.get('id'), worker.get('add'), nRowsRead, window, threshold)



# Egy konkrét worker inicializálása
def call_worker_initialize(worker_address):

	print('---------------------------------------------------------------')
	print('                      CALL_WORKER_INITIALIZE()                 ')
	print('---------------------------------------------------------------')

	# Ezzel simán meghívjuk a Worker INITIALIZE REST API Végpontját
	resp = requests.get(worker_address + '/initilaize')
	print('initialize ', resp)
	print('_______call_worker_initialize_______')

	return resp


# Az összes worker inicializálása
def initialize_workers():

	print('---------------------------------------------------------------')
	print('                      INITIALIZE_WORKERS()                     ')
	print('---------------------------------------------------------------')

	# Mindegyik Workernek initialize API-ját meg kell hívni
	print('workers_addresses = ', workers_addresses)
	result = []
	for worker_address in workers_addresses:
		print('---------------------------------------------------------------------')
		print('worker_address = ', worker_address)

		resp = call_worker_initialize(worker_address)

		msg = None
		if(resp.status_code == 200):
			msg = ('[OK] worker', worker_address, 'setup', resp.status_code)
		else:
			msg = ('[ERROR] worker', worker_address, 'setup', resp.status_code)

		print(msg)
		result.append(msg)

	return result



# Az összes workeren lefutat egy test packaget
def test_all_worker_with_test_package():

	print('---------------------------------------------------------------')
	print('                      TEST_ALL_WORKER_WITH_TEST_PCK()          ')
	print('---------------------------------------------------------------')

	print('workers_addresses = ', workers_addresses)
	for worker_address in workers_addresses:
		print('---------------------------------------------------------------------')
		print('worker_address = ', worker_address)
		print('---------------------------------------------------------------------')
		# ha van már elkészült teszt modellünk akkor mindenkinek küljük el kiszámításra
		# ehez meg kellene írni a modelkülést is -> amit szerintem már megírtam valahol
		# az egy olyan valami ami a Workerek upload API-ját hivogatják!

		# Van már egy call_worker_uploader() ami fixen a model.joblib filét küldi át
		# Ez most itt csak testz.

		call_worker_uploader(worker_address)


# Egy konkrét workernek küldi át a modelt
def call_worker_uploader(worker_address):

	print('---------------------------------------------------------------')
	print('                      CALL_WORKER_UPLOADER()                   ')
	print('---------------------------------------------------------------')

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


# Egy konkrét workernek küldi át egy konkrét modelt
def call_worker_sender(worker_address, new_clf_file_name):

	print('---------------------------------------------------------------')
	print('                      CALL_WORKER_SENDER()                   ')
	print('---------------------------------------------------------------')

	# Ezzel a módszerrel lehet átküldeni neki a joblib model filét
	uploadResultUrl = worker_address + '/uploader'
	filePath = new_clf_file_name
	fileFp = open(filePath, 'rb')
	fileInfoDict = {
	    "file": fileFp,
	}
	resp = requests.post(uploadResultUrl, files=fileInfoDict)
	print('uploader   ', resp)
	print('_______call_worker_uploader_______')





# Teszt: A workert tesztponjára hív rá assertel egy 200-as választ
def call_workers_testpoint():

	print('---------------------------------------------------------------')
	print('                      CALL_WORKER_TESTPOINT()                  ')
	print('---------------------------------------------------------------')

	# Mindegyik Workernek küldünk a testpoint API-jára egy értéket
	print('workers_addresses = ', workers_addresses)
	for worker_address in workers_addresses:
		print('---------------------------------------------------------------------')
		print('worker_address = ', worker_address)
		# Átküldök egy értéket a Workernek
		resp = requests.get(worker_address + '/testpoint?value=123456789')
		print('testpoint  ', resp)
		print(type(resp))
		print('_______call_worker_testpoint_______')
		print('---------------------------------------------------------------------')







print('---------------------------------------------------------------')
print('                            FLASK REST                         ')
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


# ezen keresztül fogadja a workerektől az eredményeket
# ezt most elöször csak úgy írom meg, hogy egy konkrét értéket legyen képes fogadni
# azonban ez nem lesz jó, mert tudni kell azt is, hogy kitől kapta.
# ráadásul, ezzel kéősőb még lesz gond, mert majd ki kell találnom, hogyan kezeljem le azt, hogy amit visszakap
# azt amikor elküldte le kell tárolnia. Valahogy úgy, hogy kinek és mit küldött,
# amikor visszakapja az eredényt akkor meg kell néznie, hogy melyik a legjobb eredmény megkersei az azonosítot
# én ennek az azonosítónak a nyomán előveszi azt a megoldást amelyik az adott azonosítóhoz tartozik.
# a küldés techinkailag úgy néz ki, hogy csinál egy modelt, és azt elmenti, (egyenlőre nem csinál semmit)
# illetve egyszer megcsinálta és azt küldözgeti.
# amikor majd lesz evolválva, akkor az adott modelt amit megcsinált, abból ki kell vennie a 'coefs_' listát,
# azt a listát eltárolni, és mellette legyen ott, hogy a küldésnél melyik gépnek külde.
# azt a változót ami most a gépek neveit tartalmazza át kell alakítanom úgy, hogy ne csak a címet tartalmazza,
# de egy azonosítot is ami egy sima szám. Fontos, hogy át kell írnom az összes olyan függvéynt ami ezt a változót
# használja.
# A lényeg az evolve() metoduson fog történni. Ott csinál több modelt(de nem egyszerre, hanem szekvenciálisan a mutátor)
# segíétségével. Lementi a modelt joblib fájlba. Kiolvassa a modellből a 'coefs_' értéket, ezt leteszi egy dict-be
# ahol az egyik a worker azonosítoja, a másik maga a lista( amiben a coefs_ van)
# Amikor visszakapja kiszámolt értéket akkor az is olyan formában jön, hogy a worker amikor fogadta a csomagot,
# akkor kapott egy azonosítot is. Ez lenne a jó megoldás, de ehelyett az lesz, hogy amikor a workereket létrehozom,
# vagy a setup, vagy az init-ben átadok neki egy olyan értéket, hogy mi a te azonosítód. Ezt egyébként a workeres_address-ből
# fogja kiolvasni a dirver. Ezt az értéket letárolja a worker és végig meg is örzi a számításai során.
# Ezzel az azonosítóval fogja visszaküldeni az fitness scooret.
# Miután látjuk, hogy melyik adta a legjobb fitness scoret, a csomagküldésnél használt dict-ből (amit tartalmazza a coef-eket)
# ki tudjuk majd olvasni a coefs_t ami a legjobb és amit mutálni akarunk
# 1...
#
# 2...Na most a driverrel kéne egy kicsit foglalkoznom.
# -- létrehoz egy modelt, eddig jó, de most meg kéne írnom a mutátor függvényt.
# -- ráadásul most már az egészet be kéne csomagolnom egy ciklusba (generation)
# -- a recievieresult REST-et megírni hogy az az adatokat irja ki egy DAO-ba
# -- ezt a DAO- az evolve mindíg resetelje.
# -- A receive ellenőrizee a dao méretét ha ez elért egy szintet (akkor visszakapta mindenkiőt a számítást) akkor hívjon rá
#    az evolvra() (az evolve nem kell hogy REST legyen)
@app.route('/receiveresult', methods=['GET'])
def receiveresult():
    received_value = request.args.get('value')
    received_gain = request.args.get('gain')
    print('---------------------------------')
    print('received_value from worker =', received_value)
    print('received_gain from worker =', received_gain)
    print('---------------------------------')
    return 'Recieve value from Worker!'


def empty_func():
	print('a')
	return 'abc'


# evolution DEV
@app.route('/evolution2')
def evolution_dev2():
	# resp = initialize_workers()
	new_clf = deepcopy(clf)
	old_coefs_ = deepcopy(new_clf.coefs_)
	print('--------------- OLD COEFS -------------')
	print(old_coefs_)
	print('for ciklus start')
	print(workers_addresses)
	for i in range(len(workers_addresses)):
		new_coefs_ = randomer.randomize(coefs = old_coefs_, factor = parameters.factor)
		print('--------------- NEW COEFS -------------')
		print(new_coefs_)
		new_clf.coefs_ = new_coefs_        # el kéne küldeni a workereknek az új modelt.
		new_clf_file_name = 'model3.joblib'           # ezt is váltogatni kell kérdés, hogy a tuloldalon milyen néven menti el?
		joblib.dump(new_clf, new_clf_file_name)         # el kéne küldeni egy adott workingernek (speckó nevet kell adni neki)
		# worker_address = 'http://192.168.0.247:8080' # ezt majd mindíg váltogatni kell
		worker_address = workers_addresses[i]
		print('az aktuális worker címe akinek küldünk:', worker_address)
		call_worker_sender(worker_address, new_clf_file_name)
	new_clf = 10
	abc = empty_func()
	print('______végig mentünk az össezs worker initializejan______')
	print('--------------------------------------------------------')
	my_result = 'mumbapa'
	return my_result








# call_worker_testpoint -> ha erre jön kérés akkor ez tovább hív a worker testpoint-jára
@app.route('/calltestpoint', methods=['GET'])
def call_workers_testpoint_api():
    call_workers_testpoint()
    return 'Called woreker testpoint'


# setup_workers
@app.route('/setupworkers')
def setup_all_worker():
    setup_workers()
    print('______végig mentünk az össezs worker setupján______')
    return 'Setup all worker'


# initialize_workers
@app.route('/initializeworkers')
def initialize_all_worker():
    resp = initialize_workers()
    print('______végig mentünk az össezs worker initializejan______')
    print('--------------------------------------------------------')
    my_result = ''.join(map(str,resp))  # <-- list to str
    return my_result



# test_workers_with_test_package
@app.route('/testworkerscalc')
def test_all_worker_with_test_package_api():
    test_all_worker_with_test_package()
    print('______végig mentünk az össezs worker tesztjén______')
    return 'Test all worker with test package'



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

# a Driver programot inicializáljuk vele
@app.route('/initilaize')
def initialize():
    initialize_driver()
    return 'Driver initilize method has been called'

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

