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

from local_packages.python_binance_master.binance.client import Client
from Trader import Trader
from simple_network import SimpleNetworkTrader
from simple_linear import SimpleLinearTrader


class TradeManager():

    def __init__(self, def_options, *args, **kwargs):
        # Initialize Trade Manager

        api_key = os.environ["binance_api"]
        api_secret = os.environ["binance_secret"]

        # Initialize List of Traders
        self.traders = []
        self.client = Client(api_key,api_secret)
        
        self.statistics = {}
    
    def add_trader(self, config, *args, **kwargs):
        # Add trader

        name = config["universal_config"]["name"]
        method = config["universal_config"]["method"]

        # Select trading method
        if(self._trader_exists(name) == None):
            if(method=="Simple Network"):
                trader = SimpleNetworkTrader(self.client, config)
            elif(method=="Simple Linear"):
                trader = SimpleLinearTrader(self.client, config)
            else:
                raise Exception("No method of type {} found for trader".format(method))
            self.traders.append(trader)
            self.statistics[name] = {}
        else:
            print("Trader already exists")


    def remove_trader(self, name, *args, **kwargs):
        # Remove trader
        trader_index = self._trader_exists(name)

        if(trader_index != None):
            trader = self.traders.pop(trader_index)
            del trader
            self.statistics.pop(name,{})
        else:
            print("Trader not found")

    def run_traders(self, *args, **kwargs):
        for trader in self.traders:
            trader.run()

    def _trader_exists(self, name, *args, **kwargs):
        # Returns index of trader if found otherwise returns None

        for i, trader in enumerate(self.traders):
            if(name == trader.name):
                return i
        
        # No trader found
        return None
    
