# ------------------------------------------------------------------------------    # <-- Plotter Class

import os
import importlib
import MLPPlot

from sys import maxsize

import numpy as np
import matplotlib.pyplot as plt

class Plotter():

    def __init__(self, result):
        self.result = result

        os.system('rm MLPPlot.py')
        os.system('wget https://raw.githubusercontent.com/JoDeMiro/Micado-Research/main/MLPPlot.py 2> /dev/null')
        importlib.reload(MLPPlot)



    def plot_generation_scatter(self, generation_holder):

        m = np.zeros((len(generation_holder), len(generation_holder[0])))
        # print(m.shape)

        for i in range(len(generation_holder)):
            for j in range(len(generation_holder[i])):
                m[i, j] = generation_holder[i][j]

        # --- kell csinálni egy [0, 1, 2, ..., 0, 1, 2, ..., 0, 1, 2] vektort is az lesz majd az x -tengely
        n = np.arange(1, len(generation_holder) + 1, 1)
        a = []
        for _ in range(len(generation_holder[0])):
            a.append(n)
        b = np.array(a)
        c = b.flatten()

        # --- ki kell teríteni a mátrixba rakott score-okat és kész is
        s = m.flatten(order='C')
        s = m.flatten(order='F')

        # --- kiszűrni a végelenűl nagyokat (amelyknél nem volt értlemezhető scorre)      # <-- new in v.014

        f = np.zeros(
            s.shape)  # <-- nincs jobb ötletem, minthogy 0-ra cserélem őket
        f[s > -maxsize + 1] = s[s > -maxsize + 1]

        # --- plot
        # plt.scatter(c, s)

        return c, s, f


# ------------------------------------------------------------------------------

    def plot_results(self, _max=None, _show_mid=False):
        if (_show_mid == True):
            fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(18, 5))
        else:
            fig, (a0, a2) = plt.subplots(1, 2, figsize=(12, 5))

        _number_of_generation = len(self.result['best_score_holder'])
        if (_max == None):
            _number_of_generation = _number_of_generation
        elif (_max > _number_of_generation):
            _number_of_generation = _number_of_generation
        elif (_max < _number_of_generation):
            _number_of_generation = _max

        _x_axis = np.arange(1, _number_of_generation + 1, 1)
        a0.scatter(_x_axis, self.result['best_score_holder'][:_number_of_generation])
        a0.set_ylabel('Means Square Error')
        a0.set_xlabel('Generation')

        c, s, f = self.plot_generation_scatter(
            self.result['generation_holder'][:_number_of_generation])

        if (_show_mid == True):
            #a1.plot(generation_holder[:_number_of_generation])
            a1.plot(f[:_number_of_generation])

            a1.set_xlabel('Generation')

        # a2.scatter(c, s)    # <-- nem szűrt adatok
        a2.scatter(c, f)  # <-- szűrt adatok

        a2.set_xlabel('Generation')
        plt.show()

# ------------------------------------------------------------------------------

    def vshow(self, _net):
        net = _net                                                     # <- a best_solution
        num_input_varialbe = ['X']
        num_input_varialbe = ['X'+str(i) for i in range(30)]
        num_input_varialbe = ['X'+str(i) for i in range(net.coefs_[0].shape[0])] # <- mlp.coefs_[0].shape[0] a bemenetek száma

        # Define the structure of the network
        network_structure = np.hstack(([len(num_input_varialbe)], np.asarray(net.hidden_layer_sizes), [1]))

        print(network_structure)

        # Draw the Neural Network with weights
        network = MLPPlot.DrawNN(network_structure, net.coefs_, num_input_varialbe)
        network.draw(line_width=1)

# ------------------------------------------------------------------------------


    def plot_int(self, ex, start=0, end=-1, p=True, i=True):

        fig, ax1 = plt.subplots(figsize=(20,5))
        ax2 = ax1.twinx()
        ax2.plot(ex.data_reader.y_train[start:end], c='black', label = 'y_train')
        if p:
            ax1.plot(ex.test_pred[start:end], label = 'test_pred')
        if i:
            ax1.plot(ex.test_pred_initial[start:end], label = 'initial')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Y value')
        ax1.hlines(ex.trader.th_up, 0, ex.data_reader.y_train.size-(start-end), lw = 1, linestyles='dashed')
        ax1.hlines(ex.trader.th_dn, 0, ex.data_reader.y_train.size-(start-end), lw = 1, linestyles='dashed')
        ax1.hlines(ex.trader.threshold, 0, ex.data_reader.y_train.size-(start-end), lw = 1)
        ax1.legend(frameon=False)
        plt.show()

# ------------------------------------------------------------------------------

    def plot_res(self, trader):
        
        trades_profit = trader.result['sell_price'] - trader.result['buy_price']
        trades_profit_net = trader.result['sell_price'] - trader.result['buy_price'] - trader.trade_cost
        
        plt.hist(trades_profit)
        plt.title('Distribution of trades gains')
        plt.show()
        
        # ---
        
        cum_profit = [0]
        cum_profit_net = [0]

        for i in range(len(trades_profit)):
            cum_profit.append(cum_profit[i] + trades_profit[i])
            cum_profit_net.append(cum_profit_net[i] + trades_profit_net[i])

        plt.plot(cum_profit, label='Cum trades profits')
        plt.plot(cum_profit_net, label='Cum trades net profits')
        plt.title('Cummulative trades profits')
        plt.xlabel('Number of trades')
        plt.legend(frameon=False)
        plt.show()
        
        # ...
        
        # cum profit (equity) az idősoron - csak az equity

        eq = np.zeros(trader.data_reader.y_train.shape[0])
        eq_net = np.zeros(trader.data_reader.y_train.shape[0])
        
        # Az eq kiszámolása itt történik meg
        # erre már nincs is szükség a traderben van
        
        for i in range(len(trader.result['sell_price'])):
            sell_index = trader.result['sell_index'][i]
            sell_index = int(sell_index)
            eq[sell_index:] = cum_profit[i + 1]
            eq_net[sell_index:] = cum_profit_net[i + 1]

        plt.plot(eq, color='C1', label='Cum Equity')
        plt.plot(eq_net, color='C2', label='Cum Net. Equity')
        plt.title('Cum Equity')
        plt.xlabel('data_reader.y_train (not full data)')
        plt.legend(frameon=False)
        plt.show()
