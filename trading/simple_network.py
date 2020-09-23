import sys,os
from Trader import Trader
sys.path.append(os.path.abspath("../data_management/"))
import numpy as np
import time
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras import Input, Model, optimizers
from keras.layers import Dense
import json


sys.path.append(os.path.abspath("./"))
sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("./data"))
from local_packages.python_binance_master.binance.client import Client

class SimpleNetworkTrader(Trader):
    

    # def __call__(self, filepath, *args, **kwargs):
    #     # return object with "BUY" or "SELL" or "HOLD", flag, and amount
    



    def train_network(self, filepath, network_path, *args, **kwargs):
        
        #get data
        now = int(time.time()//60*60*1000)
        start = now-2000*60*1000 #time 10000 minutes prior

        data = self.DataManager.load_data(filepath,start,cols=[0,4])
        answers = data[1:,1]
        input_data = data[:-1,1]

        #format data
        timeframe = 10 #10 minutes time frame
        num_data_vectors = len(input_data)-timeframe+1
        answers = answers[timeframe:]
        formatted_input = np.zeros((num_data_vectors,timeframe))

        for i in range(num_data_vectors):
            formatted_input[i,:] = input_data[i:i+timeframe]

        # randomize inputs and answers in unison
        p = np.random.permutation(len(answers))
        answers = answers[p]
        formatted_input = formatted_input[p]

        # build network structure
        net_input = Input(shape=(timeframe,))
        hidden = Dense(timeframe,activation="relu",kernel_initializer="random_uniform")(net_input)
        output = Dense(1,activation='linear',kernel_initializer="random_uniform")(hidden)
        model = Model(net_input,output)

        #define custom loss function
        def custom_loss_wrapper(net_input):
            def custom_loss(y_true, y_pred):
                if(y_true>=net_input[-1]):
                    #stock is going up
                    if(y_pred>=y_true):
                        #rewards as I would have bought
                        return 0.0
                    else:
                        #punish I was wouldn't have made as much
                        return y_pred-y_true
                else:
                    #stock is going down
                    if(y_pred>=y_true):
                        #punish as I would have lost more by holding on too much
                        return y_pred-y_true
                    else:
                        #reward as I would have sold more stock
                        return 0.0
            return custom_loss

        # add training method and metrics
        sgd = optimizers.SGD()
        model.compile(optimizer=sgd, loss=custom_loss_wrapper(net_input),metrics=["accuracy"])

        # fit model
        fit = model.fit(x=formatted_input,y=answers, validation_split=0.2, epochs=200, batch_size=32)
        
        # save network stats
        self.network_stats = {}
        self.network_stats["history"] = fit.history
        print("MAE: ", fit.history["mean_absolute_error"])
        print("MAE value: ", fit.history["val_mean_absolute_error"])

        self._save_network(model,network_path)


    
    def evaluate_network(self, filepath, network_path, *args , **kwargs):

        #get data
        now = int(time.time()//60*60*1000)
        start = now-2000*60*1000 #time 2000 minutes prior

        data = self.DataManager.load_data(filepath,start,cols=[0,4])
        answers = data[1:,1]
        input_data = data[:-1,1]

        #format data
        timeframe = 10 #10 minutes time frame
        num_data_vectors = len(input_data[:,0])-timeframe+1
        formatted_input = np.zeros((num_data_vectors,timeframe))
        for i in range(num_data_vectors):
            formatted_input[i,:] = input_data[i:i+timeframe]

        #get network
        model = self._load_network(network_path)
        
        #get predictions
        predictions = model.predict(formatted_input)

        trans_amt = 1
        earnings = 0
        wins=0
        losses=0


        for i, ans in enumerate(answers):
            last_close = formatted_input[i,-1]
            if(ans>=last_close):
                #stock is going up
                if(y_pred>=ans):
                    #rewards as I would have bought
                    wins+=1
                    earnings += (y_pred/last_close-1)*trans_amt 
                else:
                    #punish I was wouldn't have made as much
                    losses+=1
                    earnings += (y_pred/last_close-1)*trans_amt
            else:
                #stock is going down
                if(y_pred>=ans):
                    #punish as I would have lost more by holding on too much
                    losses+=1
                    earnings += (y_pred/last_close-1)*trans_amt
                else:
                    #reward as I would have sold more stock
                    wins+=1
                    earnings += (y_pred/last_close-1)*trans_amt
    
        print("wins: ", wins)
        print("losses: ", losses)
        print("earnings: ", earnings)



    
    def _save_network(self, model, network_path, *args, **kwargs):
        #save model
        model_json = model.to_json()
        with open(network_path+".json",'w') as json_file:
            json_file.write(json.dumps(json.loads(model_json),indent=4))
        
        model.save_weights(network_path+".h5")



    def _load_network(self, network_path, *args, **kwargs):

        # load network structure
        json_file = open(network_path+".json", "r")
        loaded_model_json= json_file.read()
        json_file.close()

        #create model from json file
        loaded_model = model_from_json(loaded_model_json)

        #load weights into model
        loaded_model.load_weights(network_path+'.h5')

        return loaded_model

    
    # def calculate_answer(self, input_data, *args, **kwargs):
    #     # calculate answers



        



if __name__ == "__main__":
    api_key = os.environ["binance_api"]
    api_secret = os.environ["binance_secret"]
    client = Client(api_key,api_secret)
    
    simple = SimpleNetworkTrader(client,"name","ETHUSDT","sample method")

    # simple.DataManager.update_historical_data("../data/ETHUSDT_1mtest_saved.csv",limit=1000)

    simple.train_network("../data/ETHUSDT_1mtest_saved.csv","./networks/simple_network")


