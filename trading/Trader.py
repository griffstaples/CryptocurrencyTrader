#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Griffin Staples
Date Created: Fri Oct 02 2020
License:
The MIT License (MIT)
Copyright (c) 2020 Griffin Staples

"""

import sys, os
import numpy as np
import time
import math
from keras.models import model_from_json

sys.path.append(os.path.abspath("../data_management/"))
sys.path.append(os.path.abspath("./data_management/"))

from DataManager import DataManager



class Trader(object):

    def __init__(self, client, name, symbol1, symbol2 , min_symbol1, min_symbol2, save_data=False, auto_train=False, *args, **kwargs):
        # Class Initialization
        
        self.client = client
        self.name = name
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.symbol = symbol1+symbol2
        self.save_data = save_data  # Save data methods only
        self.auto_train = auto_train # Machine learning methods only
        self.options = {}
        self.DataManager = DataManager(self.client,self.symbol)
        self.maker_commission = 0.00075
        self.taker_commission = 0.00075
        self.min_balance_symbol1 = min_symbol1
        self.min_balance_symbol2 = min_symbol2

        # Get important exchange info for trading pair
        symbol_info = self.client.get_symbol_info(self.symbol)
        self.max_algo_orders = float(self._findAtKey(symbol_info["filters"],"filterType","MAX_NUM_ALGO_ORDERS","maxNumAlgoOrders"))
        self.min_notional = float(self._findAtKey(symbol_info["filters"],"filterType","MIN_NOTIONAL","minNotional"))
        trade_precision = float(self._findAtKey(symbol_info["filters"],"filterType","LOT_SIZE","minQty"))
        self.trade_precision = int(np.ceil(abs(np.log10(trade_precision))))
        self.current_price = self.client.get_avg_price(symbol=self.symbol)
        self.current_price = float(self.current_price.get('price',0))
        self.min_amount = self.min_notional/self.current_price
        symbol1_in_usd = self.client.get_avg_price(symbol=symbol1+"USDT")
        symbol1_in_usd = float(symbol1_in_usd.get('price',0))
        min_in_usd = self.min_amount*symbol1_in_usd

        # Print out important exchange info
        print("{} symbol exchange info:".format(self.symbol))
        print("Cost of {} in {}: ".format(symbol1,symbol2),self.current_price)
        print("Min trans. cost in $BKRW", self.min_notional)
        print("Min trans. cost in $ETH: ", self.min_amount)
        print("Min trans. cost in $USD: ", min_in_usd)

        # Initialize stats and score
        self.statistics = {}

        # Assign additional options
        for key, value in kwargs.items():
            self.options[key] = value

    def __del__(self, *args, **kwargs):
        #class destructor

        pass

    def evaluate_trader(self, *args, **kwargs):

        pass
    
    def cancel_old_orders(self, open_orders, max_age_ms, *args, **kwargs):
        remove_i = []
        for i,order in enumerate(open_orders):
            if(order.get("time")+max_age_ms < int(time.time()*1000)):
                self.client.cancel_order(symbol=self.symbol,orderId=order.get("orderId"))
                remove_i[i] = i
        open_orders = np.delete(open_orders,remove_i)
        return open_orders

    def run(self, *args, **kwargs):
        # run network

        pass
        

    def _save_network(self, model, network_path, *args, **kwargs):
        #save model
        model_json = model.to_json()
        with open(network_path+".json",'w') as json_file:
            json_file.write(json.dumps(json.loads(model_json),indent=4))
        
        model.save_weights(network_path+".h5")



    def _load_network(self, network_path, *args, **kwargs):

        # load network structure
        json_file = open(network_path+".json", "r")
        loaded_model_json= json_file.read()
        json_file.close()

        #create model from json file
        loaded_model = model_from_json(loaded_model_json)

        #load weights into model
        loaded_model.load_weights(network_path+'.h5')

        return loaded_model

    def _get_amount_at_price(self, action, price):
        res = self.client.get_order_book(symbol=self.symbol,limit=1000)
        bids = res["bids"] #what people are buying for
        asks = res["asks"] #what people are selling for

        amount = 0
        total_amount = 0
        
        if(action=="SELL"):
            for bid in bids:
                total_amount+=float(bid[1])
                if(price<float(bid[0])):
                    amount+=float(bid[1])

            print("total sell amount: ", total_amount)
            print("amount can sell: ", amount)

        elif(action=="BUY"):
            for ask in asks:
                total_amount+=float(ask[1])
                if(price>float(ask[0])):
                    amount+=float(ask[1])
            
            print("total buy amount: ", total_amount)
            print("amount can buy: ", amount)

        return amount

    def _get_asset_balance(self, *args, **kwargs):

        symbol1_balance = self.client.get_asset_balance(asset=self.symbol1)
        symbol2_balance = self.client.get_asset_balance(asset=self.symbol2)
        symbol1_balance = symbol1_balance.get("free",0)
        symbol2_balance = symbol2_balance.get("free",0)

        return float(symbol1_balance), float(symbol2_balance)

    def _findAtKey(self, array,at_key,at_value,get_key):
        #searches array for object with flag key-value pair and gets desired value
        for obj in array:
            if(obj.get(at_key,"")==at_value):
                return obj.get(get_key);
        return 0
    
    def _round_up(self, num, decimals=0):
        multiplier = 10 ** decimals
        return math.ceil(num * multiplier) / multiplier
    
    