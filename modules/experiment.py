import logging
logger = logging.getLogger('pyApp')
logger.info('Experiment modul loaded')

import modules.trader
from modules.trader import Trader

import modules.randomer
from modules.randomer import Randomer

from modules.fuk import plot_trade_adv

import numpy as np
import time
import os
import joblib

from sys import maxsize
from copy import deepcopy
from datetime import datetime

from rich.progress import track


def t():
    return time.time()

class Experiment():
    
    def __init__(self, data_reader, nn):
        self.data_reader = data_reader
        self.nn = nn
        self.result = dict()
        self.attach = dict()
        self.exec_times = dict()
        self.conf = None
        
        logger.info('__init__ Experiment')
        
    def exp(self, conf):
        'Experiment'
        
        self.conf = conf

        TRADE_COST = conf['TRADE_COST']
        THRESHOLD  = conf['THRESHOLD']
        TH         = conf['TH']

        internal_ploting = conf['internal_plotting']
        repeatable       = conf['repeatable']
        re_learn         = conf['re_learn']
        
        save_interval    = conf['save_interval']

        generation = conf['generation']              # <----------- genetration (200)
        population = conf['population']              # <----------- population  (20)
        factor = conf['factor_weight']               # <----------- randomization factor (25) (10)
        factor_intercept = conf['factor_intercept']  # <----------- randomization factor for intercepts_ (1000)
        variable_factor  = conf['variable_factor']
        factor_min = conf['factor_min']              # <----------- if variable_fac then
        factor_max = conf['factor_max']              # <------------if variable_vac then
        keep_best = conf['keep_best']                # <----------- keep_best (False)
        print_generation = (False, 1)  # <----------- hiba kereséshez

        select_ratio_weight = conf['select_ratio_weight']        # <-- 1 = minden súlyt beválaszt a mutációba 0 = egyiket sem
        select_ratio_bias = conf['select_ratio_bias']          # <-- 1 = mindent biast beválaszt a mutációba 0 = egyiket sem
        variable_select_ratio = conf['variable_select_ratio']
        select_ratio_min = conf['select_ratio_min'] 
        select_ratio_max = conf['select_ratio_max'] 
        
        trader = Trader(data_reader=self.data_reader,
                        trade_cost=TRADE_COST,
                        threshold=THRESHOLD,
                        th=TH,
                        debug=False)
        
        if repeatable == True:         # <-- ha azt akarom, hogy mindíg ugyan azt az eredményt kapjam
            np.random.seed(2)

        randomer = Randomer(1)         # <-- create a Randomer to controll the mutation


        # ------------------------------------------------------------------------------
        # Töröljük a korábbi képket
        _ = os.system('rm *.png 2> /dev/null')

        # Hova tegyüka a thresholdot?
        # self.data_reader.x_train[:,0].mean()

        # ------------------------------------------------------------------------------    # <-- Test The Program

        # ----------------------------------- Azt hiszem az egészre már csináltam egy új osztályt de most ezt leteszteljük
        
        logger.info('---------- Exp -----------')
        logger.info('--------------------------')
        

        start_time = time.time()
        
        mlp = self.nn.mlp
        
        coefs = deepcopy(mlp.coefs_)
        intercepts = deepcopy(mlp.intercepts_)

        backup_mlp = deepcopy(mlp)
        working_mlp = deepcopy(mlp)

        # ------------------------------------ Dirty hack - ha tovább akarom tanítani akkor

        if re_learn == True:
            coefs = deepcopy(self.result['best_generation_solution'].coefs_)
            intercepts = deepcopy(self.result['best_generation_solution'].intercepts_)
            backup_mlp = deepcopy(self.result['best_generation_solution'])
            working_mlp = deepcopy(self.result['best_generation_solution'])
        
        # ------------------------------------

        best_generation_score = -maxsize
        best_generation_solution = None

        generation_holder = []
        population_holder = []
        best_indiv_holder = []
        best_score_holder = []

        have_found = 0

        _gt_ss  = []
        _rt_ss  = []
        _prt_ss = []
        _tt_ss  = []
        

        for i in track(range(generation)):  # <-- generation part

            # Randomize the factor
            if variable_factor:
                factor_weight = (np.random.uniform(low=factor_min, high=factor_max))
                factor_intercept = (np.random.uniform(low=factor_min, high=factor_max))

            # Randomize the selection ratio
            if variable_select_ratio:
                select_ratio_weight = (np.random.uniform(low=select_ratio_min, high=select_ratio_max))
                select_ratio_bias = (np.random.uniform(low=select_ratio_min, high=select_ratio_max))

            # Mutatja az ido mulasat
            if (i % 100 == 0):
                logger.info(f'Generation: {i}')

            best_individual_solution = None
            best_individual_score = best_generation_score

            population_holder = []
            best_indiv_holder = []

            _rt_s  = []
            _prt_s = []
            _tt_s  = []

            g_rt_s = time.time()
            for j in range(population):  # <-- population part
                _r = t()
                if (keep_best == True):
                    if (j == 0):
                        a = coefs
                        b = intercepts
                        # pass
                    else:
                        a = randomer.randomize(coefs,
                                               factor=factor_weight,
                                               select_ratio=select_ratio_weight)
                        b = randomer.randomize_intercepts(intercepts,
                                                          factor=factor_intercept,
                                                          select_ratio=select_ratio_bias)
                else:
                    a = randomer.randomize(coefs,
                                           factor=factor_weight,
                                           select_ratio=select_ratio_weight)
                    b = randomer.randomize_intercepts(intercepts,
                                                      factor=factor_intercept,
                                                      select_ratio=select_ratio_bias)
                # --> vége a súlymutációnak

                working_mlp.coefs_ = a  # <-- assigne randomized coefs to the working_model
                working_mlp.intercepts_ = b  # <-- assigne randomized intc to the working_model

                _rt_s.append((t() - _r))

                
                # számolja ki a becslést
                _p = t()
                pred = working_mlp.predict(self.data_reader.x_train)
                _prt_s.append((t() - _p))


                # mérje vissza a hibát, számolja ki a keresekedéseket
                _t = t()
                # loop vagy np
                # result = trader.calculator_ff(pred)      # <-- for loop (slow) (legacy)
                result = trader.calculator_np(pred)        # <-- numpy (fast)
                _tt_s.append((t() - _t))

                if result['buy_count'] != 0:
                    score = result[
                        'gain']                     # <-- csak akkor kapja meg az értéket ha volt vétel            
                else:
                    score = -maxsize                # <-- egyébkként -maxsize (-9223372036854775807)

                # mindíg tartsa észbe, hogy ki volt a legjobb -> ezt tárolja le
                if (score > best_individual_score
                    ):                              # mivel maximalizálni akarunk ezért 'score > best_score'
                    best_individual_score = score
                    best_individual_solution = deepcopy(working_mlp)
                    logger.info(f'new best_solution find individual = {j}')
                    logger.info(f'i = {i}, j = {j}, score = {score}')
                    # print('new best_solution find individual = ', j)
                    # print('i =', i, 'j =', j, '\tscore = ', score)

                # az éppen aktuális egyed score értékét tegyük el
                population_holder.append(score)

            _tt_sum = sum(_tt_s)
            _tt_ss.append(_tt_sum)

            _rt_sum = sum(_rt_s)
            _rt_ss.append(_rt_sum)

            _prt_sum = sum(_prt_s)
            _prt_ss.append(_prt_sum)

            g_rt_e = time.time()
            g_rt = g_rt_e - g_rt_s
            _gt_ss.append(g_rt)

            # generációnként tároljuk le a legjobb egyed score érétkét
            best_score_holder.append(best_individual_score)

            # nyomonkövetés céljából tegyük le egy adott generáció összes egyedének scorját is
            generation_holder.append(population_holder)

            if (print_generation[0]):
                if (i % print_generation[1] == 0):
                    print('best_score in generation i = ', i, ' = ',
                          best_individual_score)

            # ha a generáció legjobbja jobb mint a korábbi generáció legjobbja akkor csapja felül
            if (best_individual_score > best_generation_score):
                best_generation_score = best_individual_score
                best_generation_solution = deepcopy(best_individual_solution)
                found_better_in_generation = True
                # A generáció legjobbját tegyük le a coef változóba -> ezáltal a következő generációban ő lesz az ami alapján elindul a mutáció
                coefs = deepcopy(best_generation_solution.coefs_)
                intercepts = deepcopy(best_generation_solution.intercepts_)
                
                # now = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
                # logger.info(now)

                logger.info(f'best_score in generation i = {i} = {best_individual_score}')
                # print('best_score in generation i = ', i, ' = ', best_individual_score)


                # ------------------------------------------------------------------->
                # Az Eq animációhoz

                if internal_ploting:
                    # --------------------------------------------------------------->
                    # Az Eq animációhoz

                    replayed_pred = best_generation_solution.predict(self.data_reader.x_train)
                    _ = trader.calculator_np(replayed_pred) # szükséges, felülcsaja a traderben a dolgokat

                    have_found += 1
                    plot_trade_adv(trader, save = True, j = have_found, g=i, window=trader.window)

            # Save periodicaly
            if (i % save_interval == 0):
                file_name = 'model_' + str(i) + '.joblib'
                joblib.dump(best_generation_solution, file_name)       # <-- elmenjük


            # A kötések hosszának eloszlásat akarom időnként látni
            if (i % 500 == 0):
                trd_time = trader.result['sell_index'] - trader.result['buy_index']
                trd_time = trd_time.astype('int')
                _ = np.unique(trd_time, return_counts=True)
                ellapsed_time = time.time() - start_time
                logger.info(f'ellapsed_time = {ellapsed_time}')
                logger.info('A tradek hosszának alakulása:')
                logger.info(f'{_}')

        # A globalisan legjobb megoldás alapján csináljuk meg a becslést
        self.test_pred = best_generation_solution.predict(
            self.data_reader.x_train)  # <-- kiértékeléshez és vizualizációhoz

        # A backup_mlp alapján kiszámolhatom, hogy imlyen volt a becslés eredetileg
        self.test_pred_initial = backup_mlp.predict(
            self.data_reader.x_train
        )  # > ha később össze akarom vetni valamelyik mutációval

        # Végül minden esetben updateljük a trader.reult-ot azáltal, hogy meghívjuk a következőt
        self.replayed_pred = best_generation_solution.predict(self.data_reader.x_train)
        self.replayed_result = trader.calculator_np(self.replayed_pred)
        
        # Trader-t írjuk vissza self-be
        self.trader = trader                         # ???????????? deepcopy?

        logger.info(f'\n\t {self.replayed_result}')
        # print('\n', self.replayed_result)

        # Mérjük meg, hogy mennyi ideig fut
        self.running_time = time.time() - start_time
        logger.info('\nexperiment (takes) running_time = {:.2f}\n'.format(self.running_time))
        
        # A mért időket is becsomagoljuk             --> ezeknek rendes neveket adni ToDo.:
        self.exec_times['randomer_time'] = _rt_ss
        self.exec_times['prediction_time'] = _prt_ss
        self.exec_times['trader_time'] = _tt_ss
        self.exec_times['generation_time'] = _gt_ss
        
        # Nincs return de van setter
        self.result['best_generation_score'] = best_generation_score
        self.result['best_generation_solution'] = best_generation_solution
        
        self.result['best_score_holder'] = best_score_holder
        self.result['generation_holder'] = generation_holder
        self.result['population_holder'] = population_holder
        self.result['best_indiv_holder'] = best_indiv_holder
        
        self.attach['backup_mlp'] = backup_mlp
