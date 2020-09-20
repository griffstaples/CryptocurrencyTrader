
import os, sys
sys.path.append(os.path.abspath("./trading/"))

from TradeManager import TradeManager


if __name__ == "__main__":
    print("Starting Trading Algorithm")

    # Instantiate Trade Manager
    trader = TradeManager(def_options={})