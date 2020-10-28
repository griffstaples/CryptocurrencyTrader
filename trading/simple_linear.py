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
import json


sys.path.append(os.path.abspath("../data_management/"))
sys.path.append(os.path.abspath("./"))
sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("./data"))

from Trader import Trader
from local_packages.python_binance_master.binance.client import Client


class SimpleLinearTrader(Trader):

    def __init__(self, client, config, *args, **kwargs):
        #Initialize trader
        for key in config["specific_config"].keys():
            setattr(self,key,config["specific_config"][key])

        # Default instantiation
        super().__init__(client, config)

    def evaluate_trader(self, filepath, *args, **kwargs):

        # load all data
        data = self.DataManager.load_data(filepath,cols=[0,4])
        
        length = len(data[:,1])
        floor_length = length//self.timeframe * self.timeframe #make sure length of array is divisible by timeframe

        inputs = np.reshape(data[:floor_length,1],(floor_length//self.timeframe,self.timeframe))

        answers = inputs[1:,0] #get answers
        inputs = inputs[:-1,:] #remove last element from inputs as we don't know the answer

        wins = 0
        losses = 0
        earnings = 0
        trans_amount = 1
        total_earnings = np.zeros((len(inputs[:,0]),))
        commission = self.taker_commission
        commission_amt = commission*trans_amount

        for i,row in enumerate(inputs):
            last_price = row[-1]
            estimate = self._calc_close_price(row)

            if(answers[i]>=last_price):
                #price is going up
                if(estimate>last_price+commission_amt+self.threshold):
                    wins+=1
                    earnings += (answers[i]/last_price-1)*trans_amount-commission_amt
                elif(estimate<last_price-commission_amt-self.threshold):
                    losses+=1
                    earnings += (1-answers[i]/last_price)*trans_amount-commission_amt

            else:
                #price is going down
                if(estimate>last_price+commission_amt+self.threshold):
                    losses+=1
                    earnings += (answers[i]/last_price-1)*trans_amount-commission_amt
                elif(estimate<last_price-commission_amt-self.threshold):
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
        plt.plot(data[:floor_length:self.timeframe,1],color="red")
        plt.legend(["Earnings (scaled)", "Price Chart"])
        plt.xlabel("Minutes from start time")
        plt.ylabel("{} per {}".format(self.symbol2,self.symbol1))
        plt.show()

    def evaluate_trader_v2(self, filepath, *args , **kwargs):

        # load all data
        data = self.DataManager.load_data(filepath,cols=[0,4])
        
        length = len(data[:,1])
        floor_length = length//self.timeframe * self.timeframe #make sure length of array is divisible by 10

        inputs = np.zeros((length-self.timeframe,self.timeframe))
        for i in range(length-self.timeframe):
            inputs[i,:] = data[i:i+self.timeframe,1]
        # inputs = np.reshape(data[:floor_length,1],(floor_length//self.timeframe,self.timeframe))
        print("shape: ", np.shape(inputs))

        answers = inputs[1:,0] #get answers
        formatted_input = inputs[:-1,:] #remove last element from inputs as we don't know the answer

        trans_amt = 1
        earnings = 0
        wins=0
        losses=0
        no_trades=0
        total_earnings = np.zeros((len(answers),))
        commission = self.taker_commission*trans_amt

        for i, row in enumerate(formatted_input):
            last_close = row[-1]
            y_pred = self._calc_close_price(row)

            if(y_pred>last_close):
                trans_price = y_pred-self.threshold
                if(trans_price>last_close+commission):
                    if(answers[i]>trans_price):
                        wins+=1
                        earnings += (answers[i]/trans_price-1)*trans_amt - commission
                    else:
                        losses += 1
                        earnings += (answers[i]/trans_price-1)*trans_amt - commission
                else:
                    no_trades += 1
            elif(y_pred<last_close):
                trans_price = y_pred+self.threshold
                if(trans_price<last_close-commission):
                    if(answers[i]<trans_price):
                        wins+=1
                        earnings += (1-answers[i]/trans_price)*trans_amt - commission
                    else:
                        losses+=1
                        earnings += (1-answers[i]/trans_price)*trans_amt - commission
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
        #place order as defined by order_object
        action = order_object["action"] #BUY or SELL or HOLD
        price = order_object["price"] #price in symbol2 to buy or sell symbol1 at
        # amount = order_object["amount"] #amount of symbol1 to buy or sell

        if action != "HOLD":
            #get open orders for given symbol
            open_orders = self.client.get_open_orders(symbol=self.symbol)

            #cancel old orders
            open_orders = self.cancel_old_orders(open_orders,1000*60*self.cancel_after_mins)

            amount = self._get_amount_at_price(action,price)

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
            if(len(open_orders)<self.max_algo_orders and trans_amount>0 and amount>=trans_amount):
                if(action=="BUY"):
                    #buy symbol1 with symbol2
                    print("Buying {} {} for {} {}".format(trans_amount,self.symbol1,trans_amount*price,self.symbol2))
                    order = client.order_limit_buy(symbol=self.symbol,quantity=trans_amount,price=price)

                elif(action=="SELL"):
                    #sell symbol1 for symbol2
                    print("Selling {} {} for {} {}".format(trans_amount,self.symbol1,trans_amount*price,self.symbol2))
                    order = client.order_limit_sell(symbol=self.symbol,quantity=trans_amount,price=price)

                print(order)
            
            symbol1_balance, symbol2_balance = self._get_asset_balance()
            print("{} Balance: {}".format(self.symbol1,symbol1_balance))
            print("{} Balance: {}".format(self.symbol2,symbol2_balance))


    def place_limit_order(self, order_object, *args, **kwargs):
        #place order as defined by order_object
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
            print("TRANS_AMOUNT: ", trans_amount)

            #calculate amount to buy
            if(len(open_orders)<self.max_algo_orders and trans_amount>0):
                if(action=="BUY"):
                    #buy symbol1 with symbol2
                    print("Buying {} {} for {} {}".format(trans_amount,self.symbol1,trans_amount*price,self.symbol2))
                    order = client.order_limit_buy(symbol=self.symbol,quantity=trans_amount,price=price)

                elif(action=="SELL"):
                    #sell symbol1 for symbol2
                    print("Selling {} {} for {} {}".format(trans_amount,self.symbol1,trans_amount*price,self.symbol2))
                    order = client.order_limit_sell(symbol=self.symbol,quantity=trans_amount,price=price)

            
            symbol1_balance, symbol2_balance = self._get_asset_balance()
            print("{} Balance: {}".format(self.symbol1,symbol1_balance))
            print("{} Balance: {}".format(self.symbol2,symbol2_balance))


    def run(self, *args, **kwargs):
        # run trading algorithm

        #define constants
        filepath = "../data/{}_1m_run_data.csv".format(self.symbol)
        filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)),filepath)
        commission = self.taker_commission

        #create/update data file
        now = int((time.time()//60)*60*1000)
        start = int(now-(self.timeframe)*60000)

        if(os.path.exists(filepath)):
            self.DataManager.update_historical_data(filepath)
        else:
            self.DataManager.create_historical_data(filepath,"1m",start_str=start,end_str=now,limit=1000)
        

        #load data
        data = self.DataManager.load_data(filepath,cols=[0,4])

        #format data for network
        input_data = data[-self.timeframe:,1]

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
        # try:
        if(prediction>last_close+commission_amount+self.threshold):
            print("{}: Buying".format(self.name))
            order_object["action"] = "BUY"
            self.place_limit_order(order_object)

        elif(prediction<last_close-commission_amount-self.threshold):
            print("{}: Selling".format(self.name))
            order_object["action"] = "SELL"
            self.place_limit_order(order_object)
        
        else:
            print("{}: No order placed".format(self.name))
        # except Exception as e:
        #     print("Error occurred when placing an order: ", e)

    
    def _calc_close_price(self,close_prices):
        # calculate next close price
        x = np.arange(len(close_prices))
        close_price_fit = np.polyfit(x,close_prices,1)
        next_close = np.polyval(close_price_fit,len(close_prices)+1)
        return next_close

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
    symbol = symbol1 + symbol2

    json_file = open('./configurations/simple_linear_config1.json','r')
    config = json.load(json_file)

    simple = SimpleLinearTrader(client, config)
    
    # simple.DataManager.update_historical_data("../data/"+symbol+"_1m_saved.csv",limit=1000)
    # simple.evaluate_trader_v2("../data/"+symbol+"_1m_saved.csv")

    simple.run()


