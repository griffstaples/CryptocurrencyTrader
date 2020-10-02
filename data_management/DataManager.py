#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Griffin Staples
Date Created: Fri Oct 02 2020
License:
The MIT License (MIT)
Copyright (c) 2020 Griffin Staples

"""

import os, sys
sys.path.append(os.path.abspath("./"))
sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("./data"))

import numpy as np
from local_packages.python_binance_master.binance.client import Client
import time
import csv


class DataManager():

    def __init__(self, client, symbol, *args, **kwargs):
        # Initialize data management
        self.client = client # API client
        self.symbol = symbol # trading symbol


    def create_historical_data(self, filepath, interval, start_str, end_str, limit, *args, **kwargs):
        # Create historical data if doesn't exist

        if(os.path.exists(filepath)):
            raise Exception("File already exists. Delete existing file, or rename new file.")
        else:
            with open(filepath,'a') as file:
                klines = self.client.get_historical_klines(symbol=self.symbol, interval=interval, start_str=start_str, end_str=end_str, limit = limit)
                writer = csv.writer(file)
                writer.writerows(klines)
    
    def update_historical_data(self, filepath, limit=1000, *args, **kwargs):
        # Update existing data set
        if not os.path.isfile(filepath):
            raise Exception("The following file path does not exist yet: {}".format(filepath))

        # Determine next time to get data for
        with open(filepath,'r') as file:
            last_line = file.readlines()[-1]
            parsed_line = last_line.split(",")
            start_time = int(parsed_line[6])+1 # get start time for next interval
            last_start_time = int(parsed_line[0])
            interval = start_time-last_start_time
            binance_interval = self._millis_to_binance_interval(interval)
        
        # If next interval has passed, get new interval data
        if(int(time.time()*1000)>start_time+interval):
            with open(filepath,'a') as file:
                end_time = int(time.time()*1000) # get current time and convert to millis for binance
                new_klines = self.client.get_historical_klines(symbol=self.symbol, interval=binance_interval, start_str=start_time, end_str=end_time, limit = limit)
                writer = csv.writer(file)
                writer.writerows(new_klines)


    def load_data(self, filepath, start_time=None, end_time=None ,cols=None, *args, **kwargs):
        # Load data 
        
        #always need time stamp
        if cols:
            if not cols.count(0):
                cols.insert(0,0)

        data = np.loadtxt(filepath,dtype=np.float,delimiter=",",usecols=cols)
        # Turn start time (unix in milliseconds) to start_index
        if start_time:
            start_index = np.where(data[:,0]==start_time)[0]
            if(len(start_index)):
                start_time = start_index[0]
        if end_time:
            end_index = np.where(data[:,0]==end_time-1)[0]
            if(len(end_index)):
                end_time = end_index[0]
                
        return data[start_time:end_time,:]


    def _millis_to_binance_interval(self, millis):
        intervals_unix = {
            "60000":'1m',
            "180000":'3m',
            "300000":'5m',
            "900000":'15m',
            "1800000":'30m',
            "3600000":'1h',
            "7200000":'2h',
            "14400000":'4h',
            "21600000":'6h',
            "28800000":'8h',
            "43200000":'12h',
            "86400000":'1d',
            "259200000":'3d',
            "604800000":'1w', #worth about 12 points for get_historical_klines at 1M frequency and limit of 1000
        }
        binance_interval = intervals_unix[str(int(millis))]
        return binance_interval



if __name__ == "__main__":

    api_key = os.environ["binance_api"]
    api_secret = os.environ["binance_secret"]
    client = Client(api_key,api_secret)
    
    symbol = "ETHBKRW"
    interval = "1m"
    dataManager = DataManager(client, symbol)
    dataManager.create_historical_data("../data/"+symbol+"_"+interval+".csv","1m",0,int(time.time()*1000),1000)
    # dataManager.update_historical_data("../data/"+symbol+"_"+interval+"test.csv",1000)
    
    


