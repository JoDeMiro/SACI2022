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

workers_addresses = ['http://192.168.0.54:8080/', 'http://192.168.0.32:8080/', 'http://192.168.0.247:8080/', 'http://192.168.0.231:8080/']

workers_addresses = ['http://192.168.0.54:8080/', 'http://192.168.0.32:8080/', 'http://192.168.0.247:8080/', 'http://192.168.0.231:8080/', 'http://192.168.0.202:8080/', 'http://192.168.0.198:8080/']


population_size = len(workers_addresses)

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
def initialize_driver(_nRowsRead):

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
	nRowsRead = 98765
	nRowsRead = (int)(_nRowsRead)
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
	    "model_id": 99999,													# <-- dummy értéket kap
	}
	print('uuuuuuuuuuuuuuuuuuuuuuuu1uuuuuuuuuuuuuuuuuuuuuuu')
	print('mit küld át a fileInfoDict??')
	print('fileInfoDict')
	print('uuuuuuuuuuuuuuuuuuuuuuuu3uuuuuuuuuuuuuuuuuuuuuuuu')
	resp = requests.post(uploadResultUrl, files=fileInfoDict)
	print('<<< response >>> uploader   ', resp)
	print('<<< sended model_id was === ', fileInfoDict.get('model_id'))
	print('_______call_worker_uploader_______')


# Egy konkrét workernek küldi át egy konkrét modelt
def call_worker_sender(worker_address, new_clf_file_name, model_id):

	# --> ennek a hatására a worker vissza fog hívni a receive_result() api-ra

	print('---------------------------------------------------------------')
	print('                      CALL_WORKER_SENDER()                     ')
	print('---------------------------------------------------------------')

	# Ezzel a módszerrel lehet átküldeni neki a joblib model filét
	uploadResultUrl = worker_address + '/uploader'
	filePath = new_clf_file_name
	fileFp = open(filePath, 'rb')
	fileInfoDict = {
	    "file": fileFp,
	    "model_id": model_id,
	}
	resp = requests.post(uploadResultUrl, files=fileInfoDict)
	print('<<< response >>> uploader   ', resp)
	print('<<< sended model_id was ===', model_id)
	print('_______call_worker_sender_______')





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
from flask import Flask, flash, request, redirect, url_for, jsonify
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




global received_response_count
received_response_count = 0
print('0  >> received_response_count =', received_response_count)
def check_received_responses_count():
	global received_response_count
	received_response_count += 1
	print('1  >> received_response_count =', received_response_count)
	print('Az eddig beérkezett válaszok száma =', received_response_count)
	# if( received_response_count >= 3 ):
	if( received_response_count >= population_size ):								# ToDo: Ellenőrizni
		print('2  >> received_response_count =', received_response_count)
		# reseteljük az számlálót
		# nem ne ő resetelje, hanem a while ciklus akkor amikor kiakad
		# received_response_count = 0
		print('3  >> received_response_count =', received_response_count)
		# itt akasztjuk ki a másik megakasztott while ciklust, programot
		# global enough
		# enough = True
		# WARNING
		# jó kérdés, hogy vajon itt vagy a while ciklusban állítsuk-e át ezt az értéket.
		# két helyen nem biztos, hogy jó ötlet
		print('X >> elegendő számú válaszunk van ki fogjuk léptetni a while loopból, mert átállítom az enough értékét')

global generation_holder
generation_holder = []
def add_result(result):
	generation_holder.append(result)
	print(generation_holder)



@app.route('/receiveresult', methods=['GET'])
def receive_result():
	received_value = request.args.get('value')
	received_gain = request.args.get('gain')
	received_model_id = request.args.get('model_id')
	print('----------------------------------------------------------------------------------------------')
	print('                                       RECEIVED VALUE FROM WORKER                             ')
	print('----------------------------------------------------------------------------------------------')
	# print('received_value from worker =', received_value)
	print('received_gain from worker  =', received_gain)
	print('received_model_id from worker =', received_model_id)
	print('---------------------------------------TEGYÜK LISTÁBA AZ EREDMÉNYT----------------------------')
	result = []
	result = [received_model_id, received_gain]
	add_result(result)
	print('---------------------------------------CHECK_RECEIVED_RESPONSE_COUNT--------------------------')
	check_received_responses_count()
	print('---------------------------------------END CHECK_RECEIVED_RESPONSE_COUNT----------------------')
	print('---------------------------------------END RECEIVED VALUE FROM WORKER-------------------------')
	return 'Recieve value from Worker!'



def empty_func():
	print('    ')
	return 'abc'


# evolution DEV
@app.route('/evolution2')
def evolution_dev2():
	global generation_best_score
	generation_best_score = []
	global generation_find_better_solution_holder
	generation_find_better_solution_holder = []
	global global_best_score
	global_best_score = -9999.0
	# --- ami e fölött van az kívül esik majd a generációs iteráción
	start_evolution_time = time.time()
	print('--------------------------------------------------------------------------------------------------')
	print('-------------------------------------------START EVOLUTION2---------------------------------------')
	print('--------------------------------------------------------------------------------------------------')
	basic_clf = deepcopy(clf)
	for g in range(parameters.generation):						# generációk száma, ki kell majd vezetni
		print(bcolors.WARNING + str(parameters) + bcolors.ENDC)
		print('--------------------------------------------------------------------------------------------------')
		print(bcolors.OKBLUE + '                                           GENERATION ' + str(g) + '' + bcolors.ENDC)
		print('--------------------------------------------------------------------------------------------------')

		new_clf = deepcopy(basic_clf)
		old_coefs_ = deepcopy(new_clf.coefs_)
		# print('--------------- OLD COEFS -------------')
		# print(old_coefs_)
		print(workers_addresses)
		print('--------------------------------------------------------------------------------------------------')
		print('                                           START FOR LOOP                                         ')
		print('--------------------------------------------------------------------------------------------------')
		global received_response_count
		received_response_count = 0 			# be kell állítani 0-ra, hogy tényleg azt számolja amit kell
		# Feltételezzük, hogy ezen a ponton még nem érkezett egyetlen válasz sem, de ez egy hibás feltételezés
		# nem így kéne ezt itt lekezelni, hogy simán felül csapom ezt az értéket.
		print('received_response_count:', received_response_count)


		print('--------------------------------------------------------------------------------------------------')
		print('-------------------------------------------RESET GLOBAL generation_holder-------------------------')
		print('--------------------------------------------------------------------------------------------------')
		global generation_holder
		generation_holder = []
		print('--------------------------------------------------------------------------------------------------')
		print('-------------------------------------------RESET GLOBAL received_response_count-------------------')
		print('--------------------------------------------------------------------------------------------------')
		global enough
		enough = False					# ez az érték legyen valahogy elérhető a másik REST számára is, hogy át tudja állítani
		print('------------------------------------------------------->>>>>>>>>>>>> 1_____ enough = ', enough)
		print('--------------------------------------------------------------------------------------------------')
		print('                                           START FOR LOOP NEW CLF AND SENDING MODEL               ')
		print('--------------------------------------------------------------------------------------------------')
		for i in range(len(workers_addresses)):
			print('_________a cikluson belül itt tartunk : ', i)
			new_coefs_ = randomer.randomize(coefs = old_coefs_, factor = parameters.factor)
			# print('--------------- NEW COEFS -------------')
			# print(new_coefs_)
			new_clf.coefs_ = new_coefs_        # el kéne küldeni a workereknek az új modelt.
			# A ciklus számláló alapján azonosítható legyen a filé
			#new_clf_file_name = 'model3.joblib'           # ezt is váltogatni kell kérdés, hogy a tuloldalon milyen néven menti el?
			new_clf_file_name = 'model' + str(i) + '.joblib'           # ezt is váltogatni kell kérdés, hogy a tuloldalon milyen néven menti el?
			joblib.dump(new_clf, new_clf_file_name)         # el kéne küldeni egy adott workingernek (speckó nevet kell adni neki)
			# worker_address = 'http://192.168.0.247:8080' # ezt majd mindíg váltogatni kell
			worker_address = workers_addresses[i]
			# hagyjuk a worker_id-t mivel itt hozzuk létre és mentjük le filébe a modelt, egyszerűen csatoljuk a küldött filé mellé
			# a ciklus számláló értékét, ez lesz az azonosító, amely alapján azonosítjuk a lementett filét és a küldést is.
			# a call_worker_sender(worker_address, new_clf_file_name, model_id)
			model_id = i
			print('\n model_id:', model_id, '\n', 'worker_address:', worker_address)
			print(' az aktuális worker címe akinek küldünk:', worker_address)
			print('------------------------------------------------------->>>>>>>>>>>>> 3_____ enough = ', enough)
			call_worker_sender(worker_address, new_clf_file_name, model_id)
			print('------------------------------------------------------->>>>>>>>>>>>> 4_____ enough = ', enough)
			if( enough == True ):
				print('\n\n\n\n\n\n valami hiba történt \n\n\n\n\n\n\n')
				raise Exception("\n\n\n\n\nEzen  a ponton az enough érétéke nem lehet True, mert ha igen akkor valahol hiba van\n\n\n\n\n")
			# A lényeg, hogy cikluson belül figyeljünk oda, hogy az enough értéke soha sem lehet True,
			# mert akkor valamit rosszul csinálunk, azt jelentené, hogy már megérkezeett az összes válasz miközben
			# még nem küldük el az összes kérést.
			# az van, hogy a fenti metodus elküldi az anyagot a workernek-> a tuloldalon ez a hívás azt eredményezi, hogy számol és
			# vissza is küldi ugyan erre a gépre, de egy másik végpontra az eredményt.
		print('--------------------------------------------------------------------------------------------------')
		print('                                           END FOR LOOP NEW CLF AND SENDING MODEL                 ')
		print('--------------------------------------------------------------------------------------------------')
		# Megvizsálom, hogy visszajött-e az összes eredmény
		# Közben a fenti for ciklus még küldözgethet
		#
		# A Worker úgy van megírva, hogy külön szálon elindítja a kiszámolást,
		# amikor végez akkor tovább lépne és meghívná a Driver REST API-t amin az erdményeket küldi.
		# Ezzel viszont egy pontenciális hiba forrást nyitok, ugyanis ha sokáig tart a válasz, és függetlenül müködik az később bezavarhat.
		print('---------------------------------------------------------------------------------------------')
		print('-------------------------------------------START WHILE---------------------------------------')
		print('---------------------------------------------------------------------------------------------')
		tmp = 0
		prev_received_response_count = 0
		start_time = time.time()
		while (enough == False):
			# bizonyos ideig fogjuk ezt a ciklust különben nagyon gyorsan lefut
			time.sleep( 0.001 )
			# csak akkor irjuk ki az eredményt, ha változás van a beérkezett válaszok számában
			# látjuk egyáltalán a received_response_count globális változó értékét? Hogy tudjuk azt innen elérni?
			if( prev_received_response_count != received_response_count or received_response_count is None ):
				print('>>>> Beérkezett egy érték a Workertől hurráááááááááááááááááááááááááá')
				print('>>>> Ekkor a tmp értéke a következő volt = ', tmp)
				print('>>>> Ekkor a received_response_count = ', received_response_count)
				print('>>>> Ennyi idő telt el a while indítása óta: ', time.time() - start_time)
				prev_received_response_count = received_response_count
			# vizgálja meg, hogy megvan-e a három eredmény.
			# ha igen akkor engedje tovább futni a programot.
			# ha nem akkor tartsa ebben a ciklusban
			tmp += 1
			if(tmp > 20000):
				enough = True
			# if(received_response_count >= 3):
			if(received_response_count >= population_size):							# ToDo: Ellenőrizni
				enough = True
				received_response_count = 0
				print('>>>> Ebben a körben érkezett be az utolsó válasz is: ', tmp)
				print('>>>> Ennyi idő telt el a while indítása óta: ', time.time() - start_time)
		print('-----------------------------------------------------')
		print('#### Ebben a körben jöttünk ki a while loopból: ', tmp)
		print('vége kijöttünk a while loopból.......................')
		print('#### Ennyi időbe telt kijönni a while loopból: ', time.time()-start_time)
		#
		# Rendben, nos akkor most az van, hogy visszakaptuk az eredményket ki kéne választani, hogy melyik volt a legjobb
		# 1) A model küldésénél a file névből tudjuk, hogy melyik modelt küldtük, visszafelé is ezt küldi a Worker ez alapján azonosítom
		#
		# Rendben, a generation_holder listahoz adogatom az a generáción belül az egyes eredményeket
		# Kiiratni
		print('vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv')
		print('generation_holder       ', generation_holder)
		# Növekvő sorba rendezni a fitness score alapján (második elem a listában)
		sorted_generation_holder = sorted(generation_holder, key = lambda x:(x[1], x[0]))
		print('sorted_generation_holder', sorted_generation_holder)
		print('wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww')
		#
		# Ki kell választani a legjobb értéket, ami a listában az utolsó elem lesz
		best_result = sorted_generation_holder[-1]
		print('best_result ', best_result)
		print('best_result[0] ', best_result[0])
		print('best_result[1] ', best_result[1])
		best_model = best_result[0]
		best_score = best_result[1]
		print('best_model ', best_model)
		print(bcolors.WARNING + 'best_model ' + best_model + bcolors.ENDC)
		print('best_score ', best_score)
		print(bcolors.WARNING + 'best_score ' + best_score + bcolors.ENDC)
		#
		# Letárolni az adott generáció legjobbját egy globális változóba
		# global generation_best_score
		generation_best_score.append(best_score)
		# print(bcolors.WARNING + 'generation_best_score' + bcolors.ENDC, generation_best_score)		# <-- debuggoláshoz
		#
		# Rendben most már tudjuk, hogy ki a legjobb model az adott generációban. Töltsük be, hogy aztán mutálni tudjuk
		reloaded_model_name = 'model' + str(best_model) + '.joblib'
		print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
		print('ezt a model filét fogjuk betölteni, remélem van ilyen valahol és elmentettem amikor készült!')
		print(bcolors.WARNING + 'ezt töltjük be ' + reloaded_model_name + bcolors.ENDC)
		reloaded_best_model = joblib.load(reloaded_model_name)
		print(bcolors.OKBLUE + '# Model betöltve a joblib-ből' + bcolors.ENDC)
		print(reloaded_best_model.get_params())
		print('ggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg')
		princ('ááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááá')
		#
		#
		# Ha ez megvan akkor a legjobb modelt ismét előkaparni -> reloaded_best_model
		# Betölteni mint parent_model, vagy parnet_clf, vagy parnet_mlp
		#
		# Meg kell mutálni őkegyelmességét
		# Most itt tartok.
		# A lényeg, hogy mutációt elvileg fent elvégezzük, ezért itt nem mutálni kell, hanem ezt a modelt valahogy
		# kiszervezni egy globális változóba.
		# Továbbá úgy kell megírni ennek a forciklusnak az elejét, hogy ott már legyen egy ilyen model,
		# amit itt ebben a szent pillanatban felül csap.
		# Tehát csak az elején kell valahogy elérhetővé tennem utána folyamatosan felül csapom generácioról generációra
		#
		# Egy olyan modelt kell felülcsapnom, amit a generációs iteráción kívül inicializálok. Ezért csináltam egy ilyet basic_clf
		# Ebből olvassa be az individumoknak modelt, vagyis ezt másolja le.. Ezért ha ezt felülcsapom az jó. Akkor ez generációról
		# generációra változni fog.
		# basic_clf = deepcopy(reloaded_best_model)
		# kérdés, hogy globális-e? Illetve ki fogja legközelebb használni, ha itt felülcsapom?
		# most több generációt engedek neki.
		#
		# Be kell vezetni a keep_best kapcsolót.
		# Csak akkor csapom felül a globális modelt, ha a generáióban elértünk jobb eredményt mint amlyet korábban bármelyik model.
		#
		print('best_score        ->>>> best score in this genereation = ', best_score)
		print('global_best_score ->>>> curent all time best score     = ', global_best_score)
		find_better_solution = 0
		if( float(best_score) > global_best_score):
			basic_clf = deepcopy(reloaded_best_model)			# <-- csapjuk felül a globális modelt amiből a gyerköcök készülnek
			find_better_solution = 1
			global_best_score = float(best_score)				# <-- csapjuk felül a globális best score-t
		generation_find_better_solution_holder.append(find_better_solution)
		# print('generation_find_better_solution_holder', generation_find_better_solution_holder)			# <-- debuggoláshoz
		#
		#
		# Vége a generáción belüli itárációnak (az egyedek iterációjának)
		print('_____Vége egy adott generációnak generációnak____')
		print('_____A generáció száma amely véget ért: ', g)
	#
	# Végi generációk iterálásáank
	print('_____Vége a generációnak____')
	#
	# Majd az egész eddigi cuccot amit írtam ebbe az API-ba azt egyel beljebb tenni és berakni egy for cikluba ami a generation
	new_clf = 10
	abc = empty_func()
	print('_____Vége az Evolution2nek_____')
	evo_time = time.time() - start_evolution_time
	package = {'generation_best_score': generation_best_score, 'evolution_time': evo_time}
	# my_result =  jsonify(generation_best_score)
	my_result =  jsonify(package)
	#
	# Na most kitalálhatom, hogy mivel akarok visszatérni a juypter notebookba
	return my_result



# test pont -> törölhető
@app.route('/tmp')
def tmp():
	print('____tmp has been called____')
	lst = ['0.0', '0.1', '0.2']
	rsp_time = time.time()
	package = {'lista': lst, 'response_time':rsp_time}
	return jsonify(package)



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
    received_factor     = (float)(request.args.get('factor'))
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
	_nRowsRead = (int)(request.args.get('_nRowsRead'))
	initialize_driver(_nRowsRead)
	return 'Driver initilize method has been called'

# ------------






def princ(text: str):
	output = '\033[38;5;208m' + str(text) + '\033[0;0m'
	print(output)

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

