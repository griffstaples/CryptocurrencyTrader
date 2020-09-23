import sys,os
from Trader import Trader
sys.path.append(os.path.abspath("../data_management/"))
import numpy as np
import time
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath("./"))
sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("./data"))
from local_packages.python_binance_master.binance.client import Client


class SimpleLinearTrader(Trader):

    def __call__(self, filepath, *args, **kwargs):
        # return object with "BUY" or "SELL" or "HOLD", flag, and amount

        self.DataManager.update_historical_data(filepath)

        now = int(time.time()//60*60*1000) #time rounded down to nearest minute
        start = now-10*60*1000 #time 10 minutes prior
        data = self.DataManager.load_data(filepath,start,now,cols=[0,4]) # get data
        
        # calculate next close price
        next_close = self._calc_close_price(data[:,1])
        

        trade_object = {}
        last_close = data[-1,1]

        if next_close >= last_close:
            trade_object["action"] = "BUY" #buy first symbol, sell second symbol
        else:
            trade_object["action"] = "SELL" #sell first symbol, buy second symbol




    
    def _calc_close_price(self,close_prices):
        # calculate next close price
        x = np.arange(len(close_prices))
        close_price_fit = np.polyfit(x,close_prices,1)
        next_close = np.polyval(close_price_fit,len(close_prices)+1)
        return next_close


        

    def evaluate(self, filepath, *args, **kwargs):

        # load all data
        data = self.DataManager.load_data(filepath,cols=[0,4])
        
        # choose a time frame of 10
        time_frame = 10
        length = len(data[:,1])
        floor_length = length//time_frame * time_frame #make sure length of array is divisible by 10

        inputs = np.reshape(data[:floor_length,1],(floor_length//time_frame,time_frame))

        answers = inputs[1:,0] #get answers
        inputs = inputs[:-1,:] #remove last element from inputs as we don't know the answer

        wins = 0
        losses = 0
        earnings = 0
        trans_amount = 1
        earnings_history = []


        for i,row in enumerate(inputs):
            last_price = row[-1]
            estimate = self._calc_close_price(row)
            if(estimate>=last_price and answers[i]>=last_price):
                wins += 1
                earnings += (answers[i]/last_price-1)*trans_amount

            elif(estimate>=last_price and answers[i]<last_price):
                losses +=1
                earnings += (answers[i]/last_price-1)*trans_amount

            elif(estimate<last_price and answers[i]>=last_price):
                losses +=1
                earnings -= (answers[i]/last_price-1)*trans_amount

            elif(estimate<last_price and answers[i]<last_price):
                wins +=1
                earnings -= (answers[i]/last_price-1)*trans_amount
            earnings_history.append(earnings)

        

        print("wins: ",wins)
        print("losses: ",losses)
        print("earnings: ",earnings)

        plt.plot(earnings_history,color="red")
        plt.plot(data[:floor_length:time_frame,1],"blue")
        plt.show()




    def make_input(self, filepath, *args, **kwargs):
        # load and format data

        self.DataManager.update_historical_data(filepath)
        data = self.DataManager.load_data(filepath)
        
        return data
        
    # def calculate_answer(self, input_data, *args, **kwargs):
    #     # calculate answers



        



if __name__ == "__main__":
    api_key = os.environ["binance_api"]
    api_secret = os.environ["binance_secret"]
    client = Client(api_key,api_secret)
    
    simple = SimpleLinearTrader(client,"name","ETHUSDT","sample method")
    simple.evaluate("../data/ETHUSDT_1m_saved.csv")


