#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Griffin Staples
Date Created: Fri Oct 02 2020
License:
The MIT License (MIT)
Copyright (c) 2020 Griffin Staples

"""

import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras import Input, Model, optimizers
from keras.layers import Dense
import keras.backend as K
import json
import keras
from scipy.stats import binom
import tensorflow as tf
tf.config.run_functions_eagerly(True)

sys.path.append(os.path.abspath("../data_management/"))
sys.path.append(os.path.abspath("../local_packages/python_binance_master/binance/"))
sys.path.append(os.path.abspath("./"))
sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("./data"))

from Trader import Trader
from local_packages.python_binance_master.binance.client import Client

class SimpleNetworkTrader(Trader):

    def __init__(self, client, config, *args, **kwargs):
        #Initialize trader
        for key in config["specific_config"].keys():
            setattr(self,key,config["specific_config"][key])


        # Default instantiation
        super().__init__(client, config)

    def train_network_v2(self, filepath, network_path, *args, **kwargs):
        
        #get data
        now = int(time.time()//60*60*1000)
        # start = now-2000*60*1000 #time 10000 minutes prior
        start = 0

        data = self.DataManager.load_data(filepath,start,cols=[0,4])
        
        unmixed_formatted_input, unmixed_answers = self._format_data(data)

        # randomize inputs and answers in unison
        p = np.random.permutation(len(unmixed_answers))
        answers = unmixed_answers[p]
        formatted_input = unmixed_formatted_input[p]
        last_closes = formatted_input[:,-1]
        answers_data = np.column_stack((answers,last_closes))

        #define custom loss function

        def loss_fcn(data, y_pred):

            #parse input data
            y_true = data[:,0]
            last_close = data[:,1]
            y_pred = y_pred[:,0]
            trans_amt = 1
            commission = self.taker_commission*trans_amt
            earnings = 0
            loss = 100000

            # print('YPRED',y_pred)
            # print('LASTCLOSE',last_close)

            if(loss<20000):
                pred_up_mask = K.greater(y_pred,last_close)
                pred_down_mask = tf.math.logical_not(pred_up_mask)
                # print('MASK',pred_up_mask)

                pred_up = tf.boolean_mask(y_pred,pred_up_mask)
                last_up = tf.boolean_mask(last_close,pred_up_mask)
                true_up = tf.boolean_mask(y_true,pred_up_mask)
                pred_down = tf.boolean_mask(y_pred,pred_down_mask)
                last_down = tf.boolean_mask(last_close,pred_down_mask)
                true_down = tf.boolean_mask(y_true,pred_down_mask)
                

                trans_price_up = (pred_up-self.threshold)
                buy_up_mask = K.greater(trans_price_up,last_up+commission)
                # print(trans_price_up)
                earnings += K.sum((tf.boolean_mask(true_up,buy_up_mask)/tf.boolean_mask(trans_price_up,buy_up_mask)-1)*trans_amt - commission)

                trans_price_down = (pred_down+self.threshold)
                sell_down_mask = K.greater(trans_price_down,last_down+commission)
                earnings += K.sum((1-tf.boolean_mask(true_down,sell_down_mask)/tf.boolean_mask(trans_price_down,sell_down_mask))*trans_amt - commission)
                if(earnings<=0):
                    loss = 1 - earnings
                else:
                    loss = tf.math.exp(-earnings)
            else:
                loss = K.mean(abs(y_true-y_pred))
            # print(trans_price_down)


            # for i in range(K.int_shape(y_pred)[0]):

            #     if(y_pred[i]>last_close[i]):
            #         trans_price = y_pred[i]-self.threshold
            #         if(trans_price>last_close[i]+commission):
            #             if(y_true[i]>trans_price):
            #                 # wins+=1
            #                 earnings += (y_true[i]/trans_price-1)*trans_amt - commission
            #             else:
            #                 # losses += 1
            #                 earnings += (y_true[i]/trans_price-1)*trans_amt - commission
            #         # else:
            #             # no_trades += 1
            #     elif(y_pred[i]<last_close[i]):
            #         trans_price = y_pred[i]+self.threshold
            #         if(trans_price<last_close[i]-commission):
            #             if(y_true[i]<trans_price):
            #                 # wins+=1
            #                 earnings += (1-y_true[i]/trans_price)*trans_amt - commission
            #             else:
            #                 # losses+=1
            #                 earnings += (1-y_true[i]/trans_price)*trans_amt - commission
            # loss = 1/(100*abs(earnings)+0.1)

            
            
            
            # print(earnings)
            # loss*=100
                    # else:
                        # no_trades += 1
                # else:
                    # no_trades += 1
            # num_points = K.int_shape(y_true)[0]
            # loss=0
            # for i in range(num_points):
            #     x = y_pred[i] - last_close[i]
            #     d = y_true[i] - last_close[i]
            #     lr = abs(0.1)
            #     if(tf.math.count_nonzero(d)>0):
            #         loss+= lr*(((x/d)**4)-2*((x/d)**2)+1)
            #     else:
            #         loss += lr*(1-tf.math.exp())

             
            # loss = K.mean(100*abs(1-tf.math.exp(-((y_pred-y_true)/y_true)**2)))

            return loss

        # build network structure
        net_input = Input(shape=(self.timeframe,))

        hidden = Dense(self.timeframe,activation="relu",kernel_initializer="random_uniform")(net_input)
        output = Dense(1,activation='linear',kernel_initializer="random_uniform")(hidden)
        model = Model(net_input,output)

        sgd = optimizers.Adam()
        model.compile(optimizer=sgd,loss=loss_fcn,metrics=["mean_absolute_error"])

        # fit model
        fit = model.fit(x=formatted_input, y=answers_data, validation_split=0.2, epochs=5,batch_size=500,shuffle=False)
        
        # save network stats
        self.network_stats = {}
        self.network_stats["history"] = fit.history
        print("MAE: ", fit.history["mean_absolute_error"])
        print("MAE value: ", fit.history["val_mean_absolute_error"])
        
        # plt.plot(fit.history["mean_absolute_error"])
        # plt.plot(fit.history["val_mean_absolute_error"])
        # plt.title("model accuracy")
        # plt.ylabel("accuracy")
        # plt.xlabel("epoch")
        # plt.legend(["train","eval"],loc="upper left")

        # calculated = model.predict(unmixed_formatted_input)
        # plt.plot(calculated,'r')
        # plt.plot(unmixed_answers,'b')
        # plt.show()


        self._save_network(model,network_path)

    def train_network(self, filepath, network_path, *args, **kwargs):
        
        #get data
        now = int(time.time()//60*60*1000)
        # start = now-2000*60*1000 #time 10000 minutes prior
        start = 0

        data = self.DataManager.load_data(filepath,start,cols=[0,4])
        
        unmixed_formatted_input, unmixed_answers = self._format_data(data)

        # randomize inputs and answers in unison
        p = np.random.permutation(len(unmixed_answers))
        answers = unmixed_answers[p]
        formatted_input = unmixed_formatted_input[p]

        # build network structure
        net_input = Input(shape=(self.timeframe,))

        hidden = Dense(self.timeframe,activation="relu",kernel_initializer="random_uniform")(net_input)
        output = Dense(1,activation='linear',kernel_initializer="random_uniform")(hidden)
        model = Model(net_input,output)

        sgd = optimizers.Adam()
        model.compile(optimizer=sgd,loss="mean_absolute_error",metrics=["mean_absolute_error"])

        # fit model
        fit = model.fit(x=formatted_input, y=answers, validation_split=0.2, epochs=3,batch_size=1000)
        
        # save network stats
        self.network_stats = {}
        self.network_stats["history"] = fit.history
        print("MAE: ", fit.history["mean_absolute_error"])
        print("MAE value: ", fit.history["val_mean_absolute_error"])


        # calculated = model.predict(unmixed_formatted_input)
        # plt.plot(calculated,'r')
        # plt.plot(unmixed_answers,'b')
        # plt.show()

        self._save_network(model,network_path)


    
    def evaluate_trader(self, filepath, network_path, *args , **kwargs):

        #get data
        now = int(time.time()//60*60*1000)
        # start = now-2000*60*1000 #time 2000 minutes prior
        start = 0
        data = self.DataManager.load_data(filepath,start,cols=[0,4])

        formatted_input, answers = self._format_data(data)

        #get network
        model = self._load_network(network_path)
        
        #get predictions
        y_pred = model.predict(formatted_input)
        y_pred = y_pred[:,0]

        trans_amt = 1
        earnings = 0
        wins=0
        losses=0
        total_earnings = np.zeros((len(answers),))
        commission = self.taker_commission

        for i, ans in enumerate(answers):
            last_close = formatted_input[i,-1]
            if(ans>=last_close):
                #stock is going up
                if(y_pred[i]>last_close+trans_amt*commission+self.threshold):
                    wins+=1
                    earnings += (ans/last_close-1)*trans_amt - trans_amt*commission
                elif(y_pred[i]<last_close-trans_amt*commission-self.threshold):
                    losses+=1
                    earnings += (1-ans/last_close)*trans_amt-trans_amt*commission
            else:
                #stock is going down
                if(y_pred[i]>last_close+trans_amt*commission+self.threshold):
                    losses+=1
                    earnings += (ans/last_close-1)*trans_amt - trans_amt*commission
                elif(y_pred[i]<last_close-trans_amt*commission-self.threshold):
                    wins+=1
                    earnings += (1-ans/last_close)*trans_amt+trans_amt*commission
            total_earnings[i] = earnings
    

        N = wins+losses
        diffs = answers-formatted_input[:,-1] #find changes in value between last close price and next close price
        diffs_mean = np.mean(diffs) #find mean of diffs
        num_times_goes_down = np.count_nonzero(diffs<0)
        num_times_goes_up = len(diffs)-num_times_goes_down
        prob = binom.cdf(losses,N,0.5)

        #print evaluation info
        print("wins: ", wins)
        print("losses: ", losses)
        print("num times goes down: ", num_times_goes_down)
        print("num times goes up: ", num_times_goes_up)
        print("probality of fluke: ", 1-prob)
        print("earnings: ", earnings)

        #compare price chart with earnings (scaled)
        plt.figure(1)
        plt.plot(total_earnings/np.max(abs(total_earnings))*np.max(answers),color="blue")
        plt.plot(answers,color="red")
        plt.legend(["Earnings (scaled)", "Price Chart"])
        plt.xlabel("Minutes from start time")
        plt.ylabel("{} per {}".format(self.symbol2,self.symbol1))
        plt.show()

    def evaluate_trader_v2(self, filepath, network_path, *args , **kwargs):

        #get data
        now = int(time.time()//60*60*1000)
        # start = now-2000*60*1000 #time 2000 minutes prior
        start = 0
        data = self.DataManager.load_data(filepath,start,cols=[0,4])

        formatted_input, answers = self._format_data(data)

        #get network
        model = self._load_network(network_path)
        
        #get predictions
        y_pred = model.predict(formatted_input)
        y_pred = y_pred[:,0]

        trans_amt = 1
        earnings = 0
        wins=0
        losses=0
        no_trades=0
        total_earnings = np.zeros((len(answers),))
        commission = self.taker_commission*trans_amt

        for i, ans in enumerate(answers):
            last_close = formatted_input[i,-1]

            if(y_pred[i]>last_close):
                trans_price = y_pred[i]-self.threshold
                if(trans_price>last_close+commission):
                    if(ans>trans_price):
                        wins+=1
                        earnings += (ans/trans_price-1)*trans_amt - commission
                    else:
                        losses += 1
                        earnings += (ans/trans_price-1)*trans_amt - commission
                else:
                    no_trades += 1
            elif(y_pred[i]<last_close):
                trans_price = y_pred[i]+self.threshold
                if(trans_price<last_close-commission):
                    if(ans<trans_price):
                        wins+=1
                        earnings += (1-ans/trans_price)*trans_amt - commission
                    else:
                        losses+=1
                        earnings += (1-ans/trans_price)*trans_amt - commission
                else:
                    no_trades += 1
            else:
                no_trades += 1
            total_earnings[i] = earnings
    

        N = wins+losses+no_trades
        diffs = answers-formatted_input[:,-1] #find changes in value between last close price and next close price
        diffs_mean = np.mean(diffs) #find mean of diffs
        num_times_goes_down = np.count_nonzero(diffs<0)
        num_times_goes_up = len(diffs)-num_times_goes_down
        prob = binom.cdf(losses,N,0.5)

        #print evaluation info
        print("wins: ", wins)
        print("losses: ", losses)
        print('no trades: ', no_trades)
        print("num times goes down: ", num_times_goes_down)
        print("num times goes up: ", num_times_goes_up)
        print("probality of fluke: ", 1-prob)
        print("earnings: ", earnings)

        #compare price chart with earnings (scaled)
        plt.figure(1)
        plt.plot(total_earnings/np.max(abs(total_earnings))*np.max(answers),color="blue")
        plt.plot(answers,color="red")
        plt.legend(["Earnings (scaled)", "Price Chart"])
        plt.xlabel("Minutes from start time")
        plt.ylabel("{} per {}".format(self.symbol2,self.symbol1))
        plt.show()

    def place_market_order(self, order_object, *args, **kwargs):

        #order_object properties
        action = order_object["action"] #BUY or SELL or HOLD
        price = order_object["price"] #price in symbol2 to buy or sell symbol1 at
        amount = order_object["amount"] #amount of symbol1 to buy or sell

        if action != "HOLD":
            #get open orders for given symbol
            open_orders = self.client.get_open_orders(symbol=self.symbol)

            #cancel old orders
            open_orders = self.cancel_old_orders(open_orders,1000*60*self.cancel_after_mins)

            #get current balances in said coins
            symbol1_balance, symbol2_balance = self._get_asset_balance()
            print("{} Balance: {}".format(self.symbol1,symbol1_balance))
            print("{} Balance: {}".format(self.symbol2,symbol2_balance))

            #calculate minimum quantity required to buy
            self.min_amount = self.min_notional/price
            print("MIN_AMOUNT: ", self.min_amount)

            #determine amount to buy
            trans_amount = self._calculate_amount(action,symbol1_balance,symbol2_balance,amount)
            trans_amount = self._round_up(trans_amount,self.trade_precision)
            print("DESIRED_AMOUNT: ", amount)
            print("TRANS_AMOUNT: ", trans_amount)

            #buy/sell
            if(len(open_orders)<self.max_algo_orders and trans_amount>0 and amount>=trans_amount):
                if(action=="BUY"):
                    #buy symbol1 with symbol2
                    print("Buying {} {} for {} {}".format(trans_amount,self.symbol1,trans_amount*price,self.symbol2))
                    # order = client.order_market_buy(symbol=self.symbol,quantity=trans_amount)
                    self.statistics["orders_made"] += 1
                    self.statistics["total_trade_qty"] += trans_amount
                    

                elif(action=="SELL"):
                    #sell symbol1 for symbol2
                    print("Selling {} {} for {} {}".format(trans_amount,self.symbol1,trans_amount*price,self.symbol2))
                    # order = client.order_market_sell(symbol=self.symbol,quantity=trans_amount)
                    self.statistics["orders_made"] += 1
                    self.statistics["total_trade_qty"] += trans_amount

                print(order)
            
            symbol1_balance, symbol2_balance = self._get_asset_balance()
            print("{} Balance: {}".format(self.symbol1,symbol1_balance))
            print("{} Balance: {}".format(self.symbol2,symbol2_balance))


    def place_limit_order(self, order_object, *args, **kwargs):
        
        #order_object properties
        action = order_object["action"] #BUY or SELL or HOLD
        price = order_object["price"] #price in symbol2 to buy or sell symbol1 at
        amount = order_object["amount"] #amount of symbol1 to buy or sell

        if action != "HOLD":
            #get open orders for given symbol
            open_orders = self.client.get_open_orders(symbol=self.symbol)

            #cancel old orders
            open_orders_new = self.cancel_old_orders(open_orders,1000*60*self.cancel_after_mins)

            #get current balances in said coins
            symbol1_balance, symbol2_balance = self._get_asset_balance()
            print("{} Balance: {}".format(self.symbol1,symbol1_balance))
            print("{} Balance: {}".format(self.symbol2,symbol2_balance))

            #calculate minimum quantity required to buy
            self.min_amount = self.min_notional/price
            print("MIN_AMOUNT: ", self.min_amount)

            #determine amount to buy
            trans_amount = self._calculate_amount(action,symbol1_balance,symbol2_balance,amount)
            trans_amount = self._round_up(trans_amount,self.trade_precision)
            print("TRANS_AMOUNT: ", trans_amount)

            #calculate amount to buy
            if(len(open_orders_new)<self.max_algo_orders and trans_amount>0):
                if(action=="BUY"):
                    #buy symbol1 with symbol2
                    print("Buying {} {} for {} {}".format(trans_amount,self.symbol1,trans_amount*price,self.symbol2))
                    # order = client.order_limit_buy(symbol=self.symbol,quantity=trans_amount,price=price)
                    self.trade_history.append({"action":"BUY","amount":trans_amount,"price":price,"time":int(1000*(time.time()//60*60))})
                    self.statistics["orders_made"] += 1
                    self.statistics["total_trade_qty"] += trans_amount

                    self.statistics["conversion_rate"] = (self.statistics["orders_made"]-self.statistics["orders_cancelled"])/ (self.statistics["orders_made"]+self.statistics["orders_cancelled"])
                    self.statistics["trade_weight"] = self.statistics["total_trade_qty"]/(self.statistics["orders_made"]+self.statistics["orders_cancelled"])

                elif(action=="SELL"):
                    #sell symbol1 for symbol2
                    print("Selling {} {} for {} {}".format(trans_amount,self.symbol1,trans_amount*price,self.symbol2))
                    # order = client.order_limit_sell(symbol=self.symbol,quantity=trans_amount,price=price)
                    self.trade_history.append({"action":"SELL","amount":trans_amount,"price":price,"time":int(1000*(time.time()//60*60))})
                    self.statistics["orders_made"] += 1
                    self.statistics["total_trade_qty"] += trans_amount

                    self.statistics["conversion_rate"] = (self.statistics["orders_made"]-self.statistics["orders_cancelled"])/ (self.statistics["orders_made"]+self.statistics["orders_cancelled"])
                    self.statistics["trade_weight"] = self.statistics["total_trade_qty"]/(self.statistics["orders_made"]+self.statistics["orders_cancelled"])

                # print(order)
            
            # symbol1_balance, symbol2_balance = self._get_asset_balance()
            # print("{} Balance: {}".format(self.symbol1,symbol1_balance))
            # print("{} Balance: {}".format(self.symbol2,symbol2_balance))

    def run(self, *args, **kwargs):
        #run trading algorithm

        filepath = "./data/{}_1m_run_data.csv".format(self.symbol)
        network_path = "./trading/networks/simple_network_{}".format(self.symbol)
        commission = self.taker_commission

        #create/update data file
        now = int((time.time()//60)*60*1000)
        start = int(now-(self.timeframe)*60000)
        
        if(os.path.exists(filepath)):
            self.DataManager.update_historical_data(filepath.format(self.symbol))
        else:
            self.DataManager.create_historical_data(filepath.format(self.symbol),"1m",start_str=start,end_str=now,limit=1000)

        #load data
        data = self.DataManager.load_data(filepath,cols=[0,4])

        #format data for network
        input_data = data[-self.timeframe:,1]
        input_data = np.reshape(input_data,(1,len(input_data)))

        #get network
        model = self._load_network(network_path)

        #plug data into network
        prediction = model.predict(input_data)[0,0]

        #make decision based off prediction
        last_close = input_data[0,-1]

        order_object = {
            "action": "HOLD",
            "amount": 0,
            "price": 0
        }

        # print(order_object)
        

        sell_price = prediction+self.threshold
        buy_price = prediction-self.threshold
        small_commission = commission*self.min_notional/sell_price
        large_commission = commission*self.min_notional/buy_price

        print('Prediction: ', prediction)
        print('Last buy price: ', last_close+commission*self.min_notional/(prediction-self.threshold))
        print('Last sell price: ', last_close-commission*self.min_notional/(prediction+self.threshold))
        print('Buy Price: ', buy_price)
        print("Sell Price: ", sell_price)

        # try:
        if(buy_price>last_close+large_commission):
            order_object["action"] = "BUY"
            order_object["amount"] = self.min_notional/(buy_price)
            max_amount = self._get_amount_at_price(order_object["action"],buy_price)
            order_object["price"] = buy_price
            # print(order_object)
            # if(order_object["amount"]<max_amount):
            print("{}: Buying".format(self.name))
            self.place_limit_order(order_object)

        elif(sell_price<last_close-small_commission):
            order_object["action"] = "SELL"
            order_object["amount"] = self.min_notional/(sell_price)
            max_amount = self._get_amount_at_price(order_object["action"],sell_price)
            order_object["price"] = sell_price
            # print(order_object)
            # if(order_object["amount"]<max_amount):
            print("{}: Selling".format(self.name))
            self.place_limit_order(order_object)
        
        else:
            print("{}: No order placed".format(self.name))
        # except Exception as e:
        #     print("Error occured when trying to place order: ",e)

    def _update_stats(self, *args, **kwargs):
        # Update trading stats
        self.statistics["orders_cancelled"] += 0
        self.statistics["orders_made"] += 0
        self.statistics["orders_completed"] += 0
        self.statistics["run_time"] += 0
        self.statistics["total_trade_qty"] += 0

        

    def _format_data(self, data, *args, **kwargs):
        #trim data
        answers = data[1:,1]
        input_data = data[:-1,1]

        #format data
        num_data_vectors = len(input_data)-self.timeframe
        answers = answers[self.timeframe:]
        formatted_input = np.zeros((num_data_vectors,self.timeframe))
        for i in range(num_data_vectors):
            formatted_input[i,:] = input_data[i:i+self.timeframe]

        return formatted_input, answers


    def _calculate_amount(self, action, symbol1_amt, symbol2_amt, trans_amt):
        commission = self.taker_commission*trans_amt

        if(action=="BUY"):
            if(symbol2_amt-trans_amt-commission>=self.min_balance_symbol2):
                return max(trans_amt,self.min_amount)
            elif(symbol2_amt-self.min_amount-commission>=self.min_balance_symbol2):
                return self.min_amount
        elif(action=="SELL"):
            if(symbol1_amt-trans_amt-commission>=self.min_balance_symbol1):
                return max(trans_amt,self.min_amount)
            elif(symbol1_amt-self.min_amout-commission>=self.min_balance_symbol1):
                return self.min_amount
        return 0



if __name__ == "__main__":
    api_key = os.environ["binance_api"]
    api_secret = os.environ["binance_secret"]
    client = Client(api_key,api_secret)
    
    symbol1 = "ETH"
    symbol2 = "BKRW"
    symbol = symbol1+symbol2

    json_file = open('./configurations/simple_net_config1.json','r')
    config = json.load(json_file)

    simple = SimpleNetworkTrader(client, config)
    
    simple.DataManager.update_historical_data("../data/"+symbol+"_1m_saved.csv",limit=1000)
    simple.train_network_v2("../data/"+symbol+"_1m_saved.csv","./networks/simple_network_"+symbol)
    simple.evaluate_trader_v2("../data/"+symbol+"_1m_saved.csv","./networks/simple_network_"+symbol)
    # simple.run()


    


