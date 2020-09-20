



class Trader():

    def __init__(self, name, method, save_data, auto_train, *args, **kwargs):
        # Class Initialization
        
        self.name = name
        self.method = method
        self.save_data = save_data  # Save data methods only
        self.auto_train = auto_train # Machine learning methods only
        self.options = {}

        # Initialize stats and score
        self.statistics = {}
        self.orders_per_sec = 0
        self.reqs_per_min = 0
        self.reqs_per_day = 0

        # Assign additional options
        for key, value in kwargs.items():
            self.options[key] = value


    def modify_trader(self, options, *args, **kwargs):
        # Modify trader
        print("mo")

    def get_orders_per_sec(self, *args, **kwargs):
        return self.orders_per_sec

    def get_reqs_per_min(self, *args, **kwargs):
        return self.reqs_per_min

    def get_reqs_per_day(self, *args, **kwargs):
        return self.reqs_per_day

    def get_stats(self, *args, **kwargs):
        return self.statistics

    def __getattribute__(self, attr):
        return attr

    
    