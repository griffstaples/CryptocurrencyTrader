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
import time
import json

sys.path.append(os.path.abspath("./trading/"))

from TradeManager import TradeManager

def run_every(func,update_time):
    #define loop function
    start_time = time.time()
    
    func() #call function
    
    curr_time = time.time()
    if(curr_time-start_time<update_time):
                time.sleep(update_time-(curr_time-start_time))

def synchronize_api(update_time):
    # Synchronize API fetch time
    print("Wait until start of new minute...")
    wait_time = update_time-(time.time()-(time.time()//update_time)*update_time)
    print('waiting {}s'.format(wait_time))
    time.sleep(wait_time)
    print("Starting Trading Algorithm")


if __name__ == "__main__":
    update_time = 60 #seconds
    symbol1 = "ETH"
    symbol2 = "BKRW"

    # Synchronize API fetch time
    synchronize_api(update_time)

    # Instantiate Trade Manager
    Manager = TradeManager(def_options={})
    
    # Load trader configurations
    file_linear = open('./trading/configurations/simple_linear_config1.json','r')
    file_network = open('./trading/configurations/simple_net_config1.json','r')
    simple_linear_config1 = json.load(file_linear)
    simple_net_config1 = json.load(file_network)
    
    # Add traders into manager object
    Manager.add_trader(simple_linear_config1)
    Manager.add_trader(simple_net_config1)


    # Run traders
    while True:
        try:
            run_every(Manager.run_traders, update_time)

        except Exception as e:
            print(e, "occurred!")
            break

    
