import sys, os

sys.path.append(os.path.abspath("../data_management/"))

from DataManager import DataManager



class Trader(object):

    def __init__(self, client, name, symbol, method, save_data=False, auto_train=False, *args, **kwargs):
        # Class Initialization
        
        self.name = name
        self.symbol = symbol
        self.method = method
        self.save_data = save_data  # Save data methods only
        self.auto_train = auto_train # Machine learning methods only
        self.options = {}
        self.DataManager = DataManager(client,symbol)

        # Initialize stats and score
        self.statistics = {}

        # Assign additional options
        for key, value in kwargs.items():
            self.options[key] = value


    # def modify_trader(self, options, *args, **kwargs):
    #     # Modify trader
    #     print("mo")

    # def get_orders_per_sec(self, *args, **kwargs):
    #     return self.orders_per_sec

    # def get_reqs_per_min(self, *args, **kwargs):
    #     return self.reqs_per_min

    # def get_reqs_per_day(self, *args, **kwargs):
    #     return self.reqs_per_day

    def get_stats(self, *args, **kwargs):
        return self.statistics
    
    