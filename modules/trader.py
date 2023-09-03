# ------------------------------------------------------------------------------    # <-- Trader Class

# Trader Calculator Class

import numpy as np
import time

import logging
logger = logging.getLogger('pyApp')
logger.info('Trader modul loaded')

class Trader():
    '''
    Trader Class is responsible for ..
    '''

    def __init__(self, threshold, data_reader, trade_cost=0.0005, th=0.0, debug=False):
        '''
        Initialize the class
        :param threshold: Bellow this value the trader sell the security
          Above this value the trader buy the security.
        '''
        self.threshold = threshold
        self.th_up = 0 + th                       # New
        self.th_dn = 0 - th                       # New
        
        self.data_reader = data_reader
        self.window = data_reader.window
        self.debug = debug
        
        self.trade_cost = trade_cost               # New
        
        self.t1 = []
        self.t2 = []
        self.ex = dict()

        self.signal = None
        self.result = dict()
        
        self.hossz = data_reader.y_train.shape[0]
        # self.xx = np.arange(0, self.hossz, 1, dtype='uint32')
        # self.buy_idx = np.empty((self.hossz))
        # self.sell_idx = np.empty((self.hossz))
        
        logger.info('__init__ Trader')

    def calculator_ff(self, pred: np.ndarray) -> dict:
        '''
        Calculete each trade on the data. It works as a Backtest Engine.
        :param pred: ndarray Predicted value of the Agent. Based on this the function
          calculates the trades and the Equity.
        '''
        begin = time.time()
        
        # -------------------------------
        
        buy = pred > self.th_up
        sell = pred < self.th_dn

        sunique, scounts = np.unique(sell, return_counts=True)
        sell_stat = dict(zip(sunique, scounts))

        bunique, bcounts = np.unique(buy, return_counts=True)
        buy_stat = dict(zip(bunique, bcounts))

        lenght = pred.size

        is_in_trade = False
        is_in_buy = False
        buy_count = 0
        sell_count = 0
        buy_price  = []
        sell_price = []
        buy_index  = []
        sell_index = []
        for i in range(lenght):
            if buy[i] == True and is_in_trade == False:
                buy_count += 1
                buy_price.append(self.data_reader.y_train[i]  )
                buy_index.append(i)
                is_in_trade = True

            if sell[i] == True and is_in_trade == True:
                sell_count += 1
                sell_price.append(self.data_reader.y_train[i])
                sell_index.append(i)
                is_in_trade = False

            if i == lenght - 1 and is_in_trade == True:  # <-- le kell zárni az utolsónál a vételt ha nyitva van
                sell_count += 1
                sell_price.append(self.data_reader.y_train[i])
                sell_index.append(i)
                is_in_trade = False
                
        
        elapsed_time = time.time() - begin
        self.t1.append(elapsed_time)
        
        # sell_idx és buy_idx alapján megállapítom a trade hosszát
        trade_length = np.array(sell_index) - np.array(buy_index)
        # print('trade_length: ', trade_length)
        
        # gains = np.array(sell_price) - np.array(buy_price)                 # Régi
        gains = np.array(sell_price) - np.array(buy_price) - self.trade_cost # New
        # print(gains)
        
        gain = gains.sum()
        # print(gain)
        
        self.elapsed_time_ff = time.time() - begin

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

        self.result = {
            'buy_price': buy_price,
            'sell_price': sell_price,
            'buy_index': buy_index,
            'sell_index': sell_index,
            'trade_length': trade_length
        }

        result = {
            'buy_stat': buy_stat.get(True),
            'sell_stat': sell_stat.get(True),
            'buy_count': buy_count,
            'sell_count': sell_count,
            'gain': gain
        }
        
        return result


    def calculator_np(self, pred: np.ndarray, debug=False) -> dict:
        
        'ha bizonyos feltéel teljesül akkor az első időpillanatba akkor legyen vétel'
        
        begin = time.time()
        
        self.ex[1] = (time.time()-begin)
        self.ex[2] = (time.time()-begin)        

#        plt.figure(figsize=(20, 3))
#        plt.plot(buy_sig, label='buy_sig')
#        plt.plot(sell_sig, label='sell_sig')
#        plt.legend(frameon=False)
#        plt.show()
        
        # sunique, scounts = np.unique(sell_sig, return_counts=True)
        # sell_stat = dict(zip(sunique, scounts))
        self.ex[3] = (time.time()-begin)

        # bunique, bcounts = np.unique(buy_sig, return_counts=True)
        # buy_stat = dict(zip(bunique, bcounts))
        self.ex[4] = (time.time()-begin)

        buy_o = False
        sell_o = False
        self.ex[5] = (time.time()-begin)
        signal = np.zeros(pred.shape)
        self.ex[6] = (time.time()-begin)
        

        for i, p in enumerate(pred):
            if (buy_o == False and p > self.th_up) or (buy_o == True and p > self.th_dn):
                signal[i] = 1
                buy_o = True
                sell_o = False
            if (sell_o == False and buy_o == True and p < self.th_dn) or (sell_o == True and p < self.th_up):
                signal[i] = -1
                sell_o = True
                buy_o = False
        # ha van nyiott buy akkor zárja le az utolsóval
        if buy_o == True:
            signal[-1] = -1
            buy_o = False
        self.ex[7] = (time.time()-begin)
        
        # self.signal = signal
                
#        print(signal)
#        plt.figure(figsize=(20,3))
#        plt.plot(signal, label='signal___')
#        plt.legend(frameon=False)
#        plt.show()
        
        diff = np.diff(signal, prepend=signal[0])
        self.ex[8] = (time.time()-begin)
#       print(diff)

#        plt.figure(figsize=(20,3))
#        plt.plot(diff, label='diff___')
#        plt.legend(frameon=False)
#        plt.show()
        
        buy = np.zeros(pred.shape)
        sell = np.zeros(pred.shape)
        self.ex[9] = (time.time()-begin)
        
        buy.fill(np.nan)
        sell.fill(np.nan)
        # buy[:] = np.nan
        # sell[:] = np.nan
        self.ex[10] = (time.time()-begin)
        
        buy[diff>0] = 1
        sell[diff<0] = 1
        self.ex[11] = (time.time()-begin)
        
        # Ha az utlsó buy volt de le is zárta akkor kell ezt megcsinálni
        if (np.any(signal==1) != True and np.any(signal==-1) == True):
            # nem biztos, hogy az utolsó a ludas de most így hagyom
            logger.error('KUUUUUUUUURVA --> signalban hiba van')
            sell[-1] = np.nan
        
        # Ha az első ponton már van vételi jel akkor legyen a buy ott vétel
        if signal[0] == 1:
            buy[0] = 1
        
#        plt.figure(figsize=(20,3))
#        plt.plot(buy, label='buy')
#        plt.plot(sell, label='sell')
#        plt.legend(frameon=False)
#        plt.show()
        
#        print('buy.shape', buy.shape)
#        print('sell.shape', sell.shape)
#        print('buy.min,max', buy.min(), buy.max())
#        print('sell.min,max', sell.min(), sell.max())
        
        # ___eddig csak a signal (buy,sell) előállításán fáradoztam__most jöhet a price___
        
        # ________________________brand__________________new
        
        xx = np.arange(0, self.hossz, 1)

        # Hol van az, hogy a buy == 1 azaz buy signal
        # buy_idx = np.empty((self.hossz))                  # <-- ki lett szervezve az initbe
        buy_idx = np.empty((self.hossz))
        buy_idx[:] = np.nan
        buy_idx[buy==1] = xx[buy==1]
        buy_idx = buy_idx[~np.isnan(buy_idx)]
        buy_idx = buy_idx.astype('int')
        # print('buy_idx: ', buy_idx)
        self.ex[12] = (time.time()-begin)
        
        # Hol van az, hogy a sell == 1 azaz sell signal
        # sell_idx = np.empty((self.hossz))                  # <-- ki lett szervezve az initbe
        sell_idx = np.empty((self.hossz))
        sell_idx[:] = np.nan
        sell_idx[sell==1] = xx[sell==1]  
        sell_idx = sell_idx[~np.isnan(sell_idx)]
        sell_idx = sell_idx.astype('int')
        # print('sell_idx: ', sell_idx)
        self.ex[13] = (time.time()-begin)
        
        # buy_idx alapján megállapítom a buy_price értékeket
        buy_price = np.empty((self.hossz))
        buy_price[:] = np.nan
        # buy_price[buy==1] = self.data_reader.y_train[buy==1]    # <-- eredeti ár
        buy_price[buy==1] = self.data_reader.y_train[:, 0][buy==1]    # <-- eredeti ár
        buy_price = buy_price[~np.isnan(buy_price)]
        # print('buy_price: ', buy_price)
        self.ex[14] = (time.time()-begin)

        # sell_idx alapján megállapítom a sell_price értékeket
        sell_price = np.empty((self.hossz))
        sell_price[:] = np.nan
        # sell_price[sell==1] = self.data_reader.y_train[sell==1] # <-- eredeti ár
        sell_price[sell==1] = self.data_reader.y_train[:, 0][sell==1] # <-- eredeti ár
        sell_price = sell_price[~np.isnan(sell_price)]
        # print('sell_price: ', sell_price)
        self.ex[15] = (time.time()-begin)
        
        # print('buy_idx  ', buy_idx)
        # print('sell_idx ', sell_idx)
        
        # sell_idx és buy_idx alapján megállapítom a trade hosszát
        trade_length = sell_idx - buy_idx
        # print('trade_length: ', trade_length)
        self.ex[16] = (time.time()-begin)
        
        buy_count = buy_price.shape[0]
        sell_count = sell_price.shape[0]
        self.ex[17] = (time.time()-begin)
        

        # van gond, ha a két tömb nem egyenlő hosszú 
        if ( buy_price.shape != sell_price.shape):
            print('      buy_price, ', buy_price)
            print('      sell_price,', sell_price)
            print('buy_count = ', buy_count)
            print('sell_count = ', sell_count)
            print('len(buy_price) = ', len(buy_price))
            print('len(sell_price) = ', len(sell_price))
            print('buy_price  = ', buy_price)
            print('sell_price = ', sell_price)
            print('buy_index  = ', buy_idx)
            print('sell_index = ', sell_idx)
            print(self.signal)
            print(pred)
            print(diff)
            print(buy)
            print(sell)
            raise Exception('buy_price.shape != sell_price.shape')

            
        # _______________________________________________end
        
        #____ha a price megvan jöhet a gain kiszámolása
                    
        # gains = np.array(sell_price) - np.array(buy_price)                 # Régi
        gains = np.array(sell_price) - np.array(buy_price) - self.trade_cost # New
        self.ex[18] = (time.time()-begin)
        # print('gains: ', gains)

        gain = gains.sum()
        self.ex[19] = (time.time()-begin)
        
        self.elapsed_time_np = time.time() - begin

        if (self.debug == True):
            print('Summary :')
            # print('buy_stat = ', buy_stat)
            # print('sell_stat = ', sell_stat)
            print('buy_count = ', buy_count)
            print('sell_count = ', sell_count)
            print('len(buy_price) = ', len(buy_price))
            print('len(sell_price) = ', len(sell_price))
            print('buy_price  = ', buy_price)
            print('sell_price = ', sell_price)
            print('buy_index  = ', buy_idx)
            print('sell_index = ', sell_idx)
            # print('gains      = ', gains)
            print('gain       = ', gain)

        self.result = {
            'buy_price': buy_price,
            'sell_price': sell_price,
            'buy_index': buy_idx,
            'sell_index': sell_idx,
            'trade_length': trade_length
        }
        self.ex[20] = (time.time()-begin)

        result = {
            # 'buy_stat': buy_stat.get(1.0),
            # 'sell_stat': sell_stat.get(-1.0),
            'buy_count': buy_count,
            'sell_count': sell_count,
            'gain': gain
        }
        self.ex[21] = (time.time()-begin)
        
        del(buy_o)
        del(sell_o)
        del(signal)
        del(diff)
        del(buy)
        del(sell)
        del(buy_idx)
        del(sell_idx)
        del(buy_price)
        del(sell_price)
        del(trade_length)
        del(gains)

        return result


    def calculate_equity(self):
        
        '''
        Elvégzi az Equity kiszámítását a Trader.result alapján
        '''
        # a régi oldsuk ff modelben listáként tér vissza a sell_index meg minden ezért konv ha nem np
        if isinstance(self.result['sell_index'], np.ndarray):
            pass
        else:
            self.result['buy_index'] = np.array(self.result['buy_index'])
            self.result['sell_index'] = np.array(self.result['sell_index'])
            self.result['buy_price'] = np.array(self.result['buy_price'])
            self.result['sell_price'] = np.array(self.result['sell_price'])

        trades_profit_raw = self.result['sell_price'] - self.result['buy_price']
        trades_profit_costs = self.result['sell_price'] - self.result['buy_price'] - self.trade_cost
                
        
        # ------------------------------------------------------------------------------
        # cum profit (equity)

        cum_profit_raw = [0]
        for i in range(len(trades_profit_raw)):
            cum_profit_raw.append(cum_profit_raw[i] + trades_profit_raw[i])       # [0, ., .,]
        
        cum_profit_raw = np.cumsum(trades_profit_raw)                             # [., .,]

        # ------------------------------------------------------------------------------

        # cum profit (equity)

        cum_profit_costs = [0]
        for i in range(len(trades_profit_costs)):
            cum_profit_costs.append(cum_profit_costs[i] + trades_profit_costs[i]) # [0, ., .,]
        
        cum_profit_costs = np.cumsum(trades_profit_costs)                         # [., .,]

        # ------------------------------------------------------------------------------

        eq_raw = np.zeros(self.data_reader.y_train.shape[0])   # <-- override
        eq_cost = np.zeros(self.data_reader.y_train.shape[0]) # <-- override

        # utó indexelt ( csak akkor ugrik az equity amikor lezárta )
        for i in range(len(self.result['sell_price'])):
            sell_index = self.result['sell_index'][i]
            eq_raw[sell_index:] = cum_profit_raw[i]
            eq_cost[sell_index:] = cum_profit_costs[i]

        self.eq_cost = eq_cost
        self.eq_raw = eq_raw
        
        del(trades_profit_raw)
        del(trades_profit_costs)
        del(cum_profit_raw)
        del(cum_profit_costs)
        
        return (eq_raw, eq_cost)