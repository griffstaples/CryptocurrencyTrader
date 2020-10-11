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



if __name__ == "__main__":
    update_time = 60 #seconds
    symbol1 = "ETH"
    symbol2 = "BKRW"
    min_symbol1 = 0.0014
    min_symbol2 = 10000

    # Synchronize API fetch time
    print("Wait until start of new minute...")
    wait_time = update_time-(time.time()-(time.time()//update_time)*update_time)
    print('waiting {}s'.format(wait_time))
    time.sleep(wait_time)
    print("Starting Trading Algorithm")

    # Instantiate Trade Manager
    Manager = TradeManager(def_options={})
    
    # Add traders
    


    json_file = open('./trading/configurations/simple_linear_config1.json','r')
    simple_linear_config1 = json.load(json_file)

    json_file = open('./trading/configurations/simple_net_config1.json','r')
    simple_net_config1 = json.load(json_file)
    
    # Manager.add_trader(simple_linear_config1)
    Manager.add_trader(simple_net_config1)

    while True:
        try:
            run_every(Manager.run_traders, update_time)

        except Exception as e:
            print(e, "occurred!")
            break

    # print('running shut down routine')
    
