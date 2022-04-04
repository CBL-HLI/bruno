
class Hyperparameters():
    def __init__(self):
        self.num_node_features = None
        self.num_classes = None
        self.lr = 0.05
        self.w_decay = 5e-2   
        self.dropout = 0.3
        self.epochs = 1000                
        self.cuda = True                
        self.device  =  None