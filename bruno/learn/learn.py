import torch
import torch.nn as nn
import numpy as np
import re

class TrainModel(): 
    
    def __init__(self, graph, model, args, criterion=None):
        self.args = args
        self.graph = graph.to(self.args.device)
        self.model = model.to(self.args.device)
        if type(model.map) is dict:
            self.map = None
        else:
            self.map = self.convert_map(model.method, model.map)
        
        if not criterion: 
            criterion = nn.CrossEntropyLoss()
        self.criterion = criterion
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.w_decay)
        
        self.train_loss = []
        self.val_loss = []
        self.train_complete = False
        self.best_loss = 1000
        
    def learn(self) -> None:
        # tracks training and validation loss over epochs
        # can add early stopping mechanism by comparing losses
        for epoch in range(self.args.epochs): 
            if self.train_complete: return
            
            tl = self.train_epoch()
            self.train_loss.append(tl)
            
            vl = self.val()
            self.val_loss.append(vl)

            if vl > self.best_loss:
                self.best_loss = vl
                torch.save(self.model, "best_model.pt")
            
            if abs(self.best_loss - vl) < 0.000001:
                self.train_complete = True
        # self.train_complete = True
        
    def train_epoch(self) -> float:
        # trains a single epoch (ie. one pass over the full graph) and updates the models parameters
        # returns the loss
        self.model.train()
        labels = self.graph.y[self.graph.train_mask]
        self.optim.zero_grad()
        output = self.model.forward(self.graph) 
        loss = self.criterion(output[self.graph.train_mask], labels)
        loss.backward()
        self.optim.step()
        if self.map is not None:
            for name, param in self.model.named_parameters():
                if re.search("weight", name):
                    weights = param.cpu() * self.map[name]
                    weights = weights/np.linalg.norm(weights, axis=0, keepdims=1)
                    self.model.state_dict()[name].data.copy_(weights)
        return loss.data.item()
    
    def val(self) -> float:
        # returns the validation loss 
        self.model.eval()
        labels = self.graph.y[self.graph.val_mask]
        output = self.model.forward(self.graph) 
        loss = self.criterion(output[self.graph.val_mask], labels)
        return loss.data.item()

    def test(self) -> float: 
        # returns the test accuracy 
        if not self.train_complete: 
            self.learn()
        self.model.eval()
        labels = self.graph.y[self.graph.test_mask]
        pred = self.model.forward(self.graph)
        _, truth = self.model.forward(self.graph).max(dim=1)
        correct = float ( truth[self.graph.test_mask].eq(labels).sum().item() )
        acc = correct / self.graph.test_mask.sum().item()
        return pred[self.graph.test_mask], truth[self.graph.test_mask], labels, acc

    def weights(self, map, index):
        w = map[[str(index), str(index+1)]].drop_duplicates()
        w = w.rename(columns = {str(index): '0', str(index+1):'1'})
        w.insert(2, "values", 1)
        return w.pivot(index=['0'], columns=['1']).fillna(0)
    
    def convert_map(self, model_name, map):
        if model_name == "GCN":
            return dict([(''.join(["layers.", str(i), ".0.lin.weight"]), torch.tensor(np.transpose(self.weights(map, i).to_numpy()))) for i in range(len(map.columns)-1)])
        else:
            return dict([(''.join(["layers.", str(i), ".0.weight"]), torch.tensor(np.transpose(self.weights(map, i).to_numpy()))) for i in range(len(map.columns)-1)])