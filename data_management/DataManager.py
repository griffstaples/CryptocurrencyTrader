
import os, sys
sys.path.append(os.path.abspath("../config/"))
sys.path.append(os.path.abspath("./"))
sys.path.append(os.path.abspath("../"))

import numpy as np
from local_packages.python_binance_master.binance.client import Client
import time


class DataManager():

    def __init__(self, client, symbol, *args, **kwargs):
        # Initialize data management
        self.client = client # API client
        self.symbol = symbol

    def create_historical_data(self, interval, start_str, end_str, limit, *args, **kwargs):
        # Get historical data
        klines = self.client.get_historical_klines(symbol=self.symbol, interval=interval, start_str=start_str, end_str=end_str, limit = limit)

        return klines



if __name__ == "__main__":

    api_key = os.environ["binance_api"]
    api_secret = os.environ["binance_secret"]
    client = Client(api_key,api_secret)
    
    dataManager = DataManager(client, "ETHBTC")
    client.ping()

    
    # def update_historical_data(self, *args, **kwargs):
    #     # Update existing data set

    # def load_data(self, *args, **kwargs):
    #     # Load data

    # def create_reduced_set(self, *args, **kwargs):
    #     # Create reduced data set

    # def create_binary_answers(self, *args, **kwargs):
    #     # Create binary answers