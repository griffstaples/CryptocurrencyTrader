import os, sys
sys.path.append(os.path.abspath("../config/"))

from binance.client import Client

from Trader import Trader


class TradeManager():

    def __init__(self, def_options, *args, **kwargs):
        # Initialize Trade Manager

        api_key = os.environ["binance_api"]
        api_secret = os.environ["binance_secret"]

        # Initialize List of Traders
        self.traders = []
        self.client = Client(api_key,api_secret)
        
        self.statistics = {}
        self.orders_per_sec = 0
        self.reqs_per_min = 0
        self.reqs_per_day = 0

    
    def add_trader(self, name, method, save_data, auto_train, *args, **kwargs):
        # Add trader
        if(self._trader_exists(name) == None):
            trader = Trader(name, method, save_data, auto_train, args, kwargs)
            self.traders.append(trader)
            self.statistics[name] = {}
        else:
            print("Trader already exists")


    def remove_trader(self, name, *args, **kwargs):
        # Remove trader
        trader_index = self._trader_exists(name)

        if(trader_index != None):
            self.traders.pop(trader_index)
            self.statistics.pop(name,{})
        else:
            print("Trader not found")

        

    def modify_trader(self, name, options, *args, **kwargs):
        # Modify trader
        trader_index = self._trader_exists(name)
        if(trader_index != None):
            trader = self.traders[trader_index]
            trader.modify_trader(options)
        else:
            print("Trader not found")



    def _trader_exists(self, name, *args, **kwargs):
        # Returns index of trader if found otherwise returns None

        for i, trader in enumerate(self.traders):
            if(name == trader.name):
                return i
        
        # No trader found
        return None
    
    def get_all_stats(self, *args, **kwargs):
        # Get all trader statistics
        for trader in self.traders:
            self.statistics[trader.name] = trader.get_stats()


    def get_total_score(self, *args, **kwargs):
        # Get API Score
        for trader in self.traders:
            self.orders_per_sec += trader.get_orders_per_sec()
            self.reqs_per_min += trader.get_reqs_per_min()
            self.reqs_per_day += trader.get_reqs_per_day()
    
    
