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
from scipy.stats import binom


sys.path.append(os.path.abspath("../data_management/"))
sys.path.append(os.path.abspath("./"))
sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("./data"))

from Trader import Trader
from local_packages.python_binance_master.binance.client import Client


class SimpleLinearTrader(Trader):


    def evaluate_trader(self, filepath, *args, **kwargs):

        # load all data
        data = self.DataManager.load_data(filepath,cols=[0,4])
        
        # choose a time frame of 10
        time_frame = 10
        length = len(data[:,1])
        floor_length = length//time_frame * time_frame #make sure length of array is divisible by 10

        inputs = np.reshape(data[:floor_length,1],(floor_length//time_frame,time_frame))

        answers = inputs[1:,0] #get answers
        inputs = inputs[:-1,:] #remove last element from inputs as we don't know the answer

        wins = 0
        losses = 0
        earnings = 0
        trans_amount = 1
        total_earnings = np.zeros((len(inputs[:,0]),))
        commission = 0.00075
        commission_amt = commission*trans_amount
        threshold = 1000



        for i,row in enumerate(inputs):
            last_price = row[-1]
            estimate = self._calc_close_price(row)

            if(answers[i]>=last_price):
                #price is going up
                if(estimate>last_price+commission_amt+threshold):
                    wins+=1
                    earnings += (answers[i]/last_price-1)*trans_amount-commission_amt
                elif(estimate<last_price-commission_amt-threshold):
                    losses+=1
                    earnings += (1-answers[i]/last_price)*trans_amount-commission_amt

            else:
                #price is going down
                if(estimate>last_price+commission_amt+threshold):
                    losses+=1
                    earnings += (answers[i]/last_price-1)*trans_amount-commission_amt
                elif(estimate<last_price-commission_amt-threshold):
                    wins+=1
                    earnings += (1-answers[i]/last_price)*trans_amount-commission_amt
            
            total_earnings[i] = earnings


        N = wins+losses
        diffs = answers-inputs[:,-1] #find changes in value between last close price and next close price
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
        print("earnings: ",earnings)

        plt.figure(1)
        plt.plot(total_earnings/np.max(abs(total_earnings))*np.max(answers),color="blue")
        plt.plot(data[:floor_length:time_frame,1],color="red")
        plt.legend(["Earnings (scaled)", "Price Chart"])
        plt.xlabel("Minutes from start time")
        plt.ylabel("{} per {}".format(symbol2,symbol1))
        plt.show()

    def place_order(self, order_object, *args, **kwargs):
        #place order as defined by order_object
        action = order_object["action"] #BUY or SELL or HOLD
        price = order_object["price"] #price in symbol2 to buy or sell symbol1 at
        amount = order_object["amount"] #amount of symbol1 to buy or sell

        if action != "HOLD":
            #get open orders for given symbol
            open_orders = self.client.get_open_orders(symbol=self.symbol)

            #cancel old orders
            open_orders = self.cancel_old_orders(open_orders,1000*60*10)

            #get current balances in said coins
            symbol1_balance, symbol2_balance = self._get_asset_balance()
            print("{} Balance: {}".format(symbol1,symbol1_balance))
            print("{} Balance: {}".format(symbol2,symbol2_balance))

            #calculate minimum quantity required to buy
            self.min_amount = self.min_notional/price
            print("MIN_AMOUNT: ", self.min_amount)

            #determine amount to buy
            trans_amount = self._calculate_amount(action,symbol1_balance,symbol2_balance,amount)
            trans_amount = self._round_up(trans_amount,self.trade_precision)
            print("TRANS_AMOUNT: ", trans_amount)

            #calculate amount to buy
            if(len(open_orders)<self.max_algo_orders and trans_amount>0):
                if(action=="BUY"):
                    #buy symbol1 with symbol2
                    print("Buying {} {} for {} {}".format(trans_amount,symbol1,trans_amount*price,symbol2))
                    order = client.order_limit_buy(symbol=self.symbol,quantity=trans_amount,price=price)

                elif(action=="SELL"):
                    #sell symbol1 for symbol2
                    print("Selling {} {} for {} {}".format(trans_amount,symbol1,trans_amount*price,symbol2))
                    order = client.order_limit_sell(symbol=self.symbol,quantity=trans_amount,price=price)

                print(order)
            
            symbol1_balance, symbol2_balance = self._get_asset_balance()
            print("{} Balance: {}".format(symbol1,symbol1_balance))
            print("{} Balance: {}".format(symbol2,symbol2_balance))


    def run(self, *args, **kwargs):
        # run trading algorithm

        #define constants
        filepath = "./data/{}_1m_run_data.csv".format(self.symbol)
        timeframe = 10
        commission = 0.00075
        threshold = 1000

        #create/update data file
        now = int((time.time()//60)*60*1000)
        start = int(now-(timeframe-1)*60000)

        if(os.path.exists(filepath)):
            self.DataManager.update_historical_data(filepath.format(self.symbol))
        else:
            self.DataManager.create_historical_data(filepath.format(self.symbol),"1m",start_str=start,end_str=now,limit=1000)
        

        #load data
        data = self.DataManager.load_data(filepath,cols=[0,4])

        #format data for network
        input_data = data[-timeframe:,1]

        #make prediction
        prediction = self._calc_close_price(input_data)

        #make decision based off prediction
        last_close = input_data[-1]
        order_object = {
            "action": "HOLD",
            "amount": self.min_notional/last_close, # if zero, it will be auto-populated with minimum amount
            "price": last_close
        }
        commission_amount = commission*order_object["amount"]

        print(order_object)
        if(prediction>last_close+commission_amount+threshold):
            print("{}: Buying".format(self.name))
            order_object["action"] = "BUY"
            self.place_order(order_object)

        elif(prediction<last_close-commission_amount-threshold):
            print("{}: Selling".format(self.name))
            order_object["action"] = "SELL"
            self.place_order(order_object)
        
        else:
            print("{}: No order placed".format(self.name))

    
    def _calc_close_price(self,close_prices):
        # calculate next close price
        x = np.arange(len(close_prices))
        close_price_fit = np.polyfit(x,close_prices,1)
        next_close = np.polyval(close_price_fit,len(close_prices)+1)
        return next_close


        
        



if __name__ == "__main__":
    api_key = os.environ["binance_api"]
    api_secret = os.environ["binance_secret"]
    client = Client(api_key,api_secret)

    symbol1 = "ETH"
    symbol2 = "BKRW"
    symbol = symbol1 + symbol2
    
    simple = SimpleLinearTrader(client,"MySimpleTrader",symbol1,symbol2)

    simple.evaluate_trader("../data/{}_1m_saved.csv".format(symbol))
    # simple.run()




