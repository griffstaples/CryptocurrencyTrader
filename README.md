# Crypto Trader

The purpose of this project was to build a configurable cryptocurrency trader off to experiment with trading different cryptocurrencies using different trading algorithms. This project uses sammchardy's public python-binance library to simplify calls made to the binance API (see https://github.com/sammchardy/python-binance). I modified sammchardy's library for my purposes to ensure that request weight and order limits are more strictly followed (reducing risk of an IP ban). The modified version of sammchardy's code is included in the local_packages directory and contains the original MIT License of the library. This project is a work in progress and I will likely be updating it in the coming weeks and months. 


### Getting started
##### Clone the repo:

```
git clone <repo_url>
```

##### Set up your virtual environment with installed libraries:

```
python3 -m venv ./
source venv/bin/activate
pip install -r requirements.txt
```

##### Make a binance account if you don't already have one:

https://accounts.binance.com/en/register


##### Set up your Binance API:

https://www.binance.com/en/support/articles/360002502072


##### Set your binance api and secret as shell environment variables by modifying your .bashrc or .zshrc to include the following lines:

```
export binance_api="my_api_key"
export binance_secret="my_api_secret"
```

Navigate to the top level folder and open main.py. The code as is will buy and sell ETH with BKRW using two different types of trading algorithms - a simple linear fit (Simple Linear) and a simple neural network (Simple Network). I do not recommend using these as their success is questionable. However, feel free to add your own trading algorithms.

##### Using an existing Trader
There are currently two traders included in the trading/ folder. Each trader implements a configuration json file with a universal and specific configuration section. I have found the use of configuration files to be a simple way of experimenting with different parameters without crowding the trader declarations themselves. The project is by default configured to run both a linear regression trader and a simple multi-perception trader. All you need to do is run main.py

##### Make a custom Trader

To make a custom trader, simply implement a trader as a child of the Trader class. This will cover the initialization which has a lot of useful components. The trader should implement a run() method which places an order to the API based on your trading algorithm. The run() method is called from the TradeManager and therefore you should import your new trader here as well and add it to the add_trader function with a name to reference it (See current implement of traders in TradeManager). If you wish to implement some of the functionality I have implemented in my traders, such as keeping track of trading statistics, look at how I have implemented it in simple_network.py.
