# ----------------------------------------------------------------------------------

# Segéd függvények a plottoláshoz

# ----------------------------------------------------------------------------------

import os
import glob
import json
import pickle
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from datetime import datetime

import logging
logger = logging.getLogger('pyApp')
logger.info('Fuk modul loaded')

def plot_trade_adv(trader, trader_signal=None, save=False, j=1, g='', _from=0, _back=0, window=0, step=False):
    '''
    Csak a plottolásért felelős, minden adatot paraméterként kap.
    Semmit nem lát a függvényen kívűl.
    '''
    
    size = trader.data_reader.y_train.shape[0]
    start = _from
    end   = size - _back

    fig, ax1 = plt.subplots(figsize=(20, 5))

    ax2 = ax1.twinx()

    # Plot 1
    ax1.plot(trader.data_reader.y_train[start:end,])
    

    # Plot 2
    __eq_raw, __eq_cost = trader.calculate_equity()

    if step == True:
        ax2.plot(__eq_raw[start:end,], color='C1', drawstyle='steps-post', label='Cum Equity raw')
        ax2.plot(__eq_cost[start:end,], color='C2', drawstyle='steps-post', label='Cum Equity cost')
    else:
        ax2.plot(__eq_raw[start:end,], color='C1', label='Cum Equity cost')
        ax2.plot(__eq_cost[start:end,], color='C2', label='Cum Equity cost')

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close price USD ($)', color='C0')
    ax2.set_ylabel('Cum. Equity', color='C1')

    plt.title('Close price history with Equity at gen = ' + str(g))
    plt.gca().yaxis.set_major_formatter(
        StrMethodFormatter('{x:,.4f}'))  # 4 decimal places

    chart_y_max = trader.data_reader.y_train[start:end].max()
    chart_y_min = trader.data_reader.y_train[start:end].min()
    
    
    # Buy price
    buy_price_array = np.zeros((size))
    buy_price_array[:] = np.nan
    for i in range(len(trader.result['buy_index'])):
        buy_idx  = int(trader.result['buy_index'][i])
        sell_idx = int(trader.result['sell_index'][i])

        # vertical lines
        if buy_idx >= start and buy_idx <= end:
            ax1.vlines(trader.result['buy_index'][i] - start,
                       chart_y_min,
                       chart_y_max,
                       lw=0.5,
                       ls='dashed',
                       color='black')
        if sell_idx >= start and sell_idx <= end:
            ax1.vlines(trader.result['sell_index'][i] - start,
                       chart_y_min,
                       chart_y_max,
                       lw=0.5,
                       color='black')
        
        # buy_price_array
        buy_price_array[buy_idx:sell_idx+0] = trader.result['buy_price'][i]

    # buy_price_array
    ax1.plot(buy_price_array[start:end], color='black', lw=2, drawstyle='steps-post')


    # Signal
    if isinstance(trader_signal, (np.ndarray, np.generic) ):
        ax3 = ax1.twinx()
        ax3.plot(trader_signal[start:end], lw=0.15, color='C2')
        ax3.hlines(trader.threshold, xmin=0, xmax=end-start, lw=1.5, linestyle='dashed', color='black')
        ax3.hlines(trader.th_up, xmin=0, xmax=end-start, lw=1.5, color='black')
        ax3.hlines(trader.th_dn, xmin=0, xmax=end-start, lw=1.5, color='black')
        ax3.set_yticks([-1.0, 0, 1.0])
        
    if save == False:
        plt.show()
    
    if save == True:
        plt.savefig('price_{0:04}'.format(j) + '.png')
        plt.close('all')


# ----------------------------------------------------------------------------------


def create_animated_mp4(filter='price*.png', prefix='ani_', framerate=10, quite=True):
    postfix = ' 2> /dev/null' if quite else ''
    output = prefix + filter[0:filter.find('*')] + '.mp4'
    os.system('rm ' + output)
    if (len(glob.glob(filter)) > 0):
        os.system('ffmpeg -r ' + str(framerate) + ' -pattern_type glob -i "' +
                  filter + '" -vcodec libx264 -crf 25 -pix_fmt yuv420p ' +
                  output + postfix)

        print('ok')
    else:
        print('skipped')
        pass

# ----------------------------------------------------------------------------------

def save_animated_mp4(filter='price*.png', prefix='ani_', framerate=10, quite=True):

    print(temp)
    print(temp.dir_name)
    postfix = ' 2> /dev/null' if quite else ''
    output = temp.dir_name + '/' + prefix + filter[0:filter.find('*')] + '.mp4'
    print(output)
    os.system('rm ' + output)
    if (len(glob.glob(filter)) > 0):
        os.system('ffmpeg -r ' + str(framerate) + ' -pattern_type glob -i "' +
                  filter + '" -vcodec libx264 -crf 25 -pix_fmt yuv420p ' +
                  output + postfix)

        print('ok')
    else:
        print('skipped')
        pass

# ----------------------------------------------------------------------------------

# Nos ha már az ember rááldozott egy csomó időt, hogy tanítsa a modelt, akkor talán illene azt elmenteni

def save_model_trader_results(ex):
    
    file_base = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    dir_name = os.path.join('NightRuns', file_base)

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        print("Directory " , dir_name ,  " Created ")
    else:    
        print("Directory " , dir_name ,  " already exists")
    
    # --
    
    file_name = file_base + '_model.joblib'
    file_path = os.path.join(dir_name, file_name)
    joblib.dump(ex.result['best_generation_solution'], file_path)       # <-- elmenjük
    backup_model = joblib.load(file_path)                           # <-- betöltjük

    print(type(backup_model))

    file_name = file_base + '_trader.joblib'
    file_path = os.path.join(dir_name, file_name)
    joblib.dump(ex.trader, file_path)
    backup_trader = joblib.load(file_path)

    print(type(backup_trader))

    file_name = file_base + '_generation_holder.joblib'
    file_path = os.path.join(dir_name, file_name)
    joblib.dump(ex.result['generation_holder'], file_path)
    backup_generation_holder = joblib.load(file_path)

    print(type(backup_generation_holder))
    
    file_name = file_base + '_result.joblib'
    file_path = os.path.join(dir_name, file_name)
    joblib.dump(ex.result, file_path)
    backup_result = joblib.load(file_path)
    
    print(type(backup_result))
    
    file_name = file_base + '_conf.joblib'
    file_path = os.path.join(dir_name, file_name)
    joblib.dump(ex.conf, file_path)
    backup_conf = joblib.load(file_path)
    
    print(type(backup_conf))
    
    global temp
    temp = Temp(dir_name)

class Temp():
    
    def __init__(self, dir_name):
        
        self.dir_name = dir_name
        
        print('\n\n\n', self.dir_name, '\n\n\n')



# ----------------------------------------------------------------------------------

# a trader.calculator zabálja az időt

def run_info(ex):
    print('{:0.4f} sec \t total time'.format(sum(ex.exec_times['generation_time'])))   # total time
    print('{:0.4f} sec \t mlp pred'.format(sum(ex.exec_times['prediction_time'])))  # mlp pred
    print('{:0.4f} sec \t randomer'.format(sum(ex.exec_times['randomer_time'])))   # randomer
    print('{:0.4f} sec \t trader'.format(sum(ex.exec_times['trader_time'])))     # trader
    
    logger.info('--------------------------')
    logger.info('{:0.4f} sec \t total time'.format(sum(ex.exec_times['generation_time'])))   # total time
    logger.info('{:0.4f} sec \t mlp pred'.format(sum(ex.exec_times['prediction_time'])))  # mlp pred
    logger.info('{:0.4f} sec \t randomer'.format(sum(ex.exec_times['randomer_time'])))   # randomer
    logger.info('{:0.4f} sec \t trader'.format(sum(ex.exec_times['trader_time'])))     # trader