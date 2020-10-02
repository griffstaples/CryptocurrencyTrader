
import sys
import os

sys.path.append(os.path.abspath("./trading/"))

import time

from TradeManager import TradeManager

update_time = 60 #seconds

if __name__ == "__main__":
    print("Starting Trading Algorithm")

    start_time = time.time()
    
    # Instantiate Trade Manager
    Manager = TradeManager(def_options={})
    
    Manager.add_trader("Sample trader", "ETH", "BKRW","Simple Network")

    Manager.run_traders()

    curr_time = time.time()
    if(curr_time-start_time<update_time):
        time.sleep(update_time-(curr_time-start_time))
        