import torch
import torch.nn as nn
import numpy as np
import re
from torch import linalg as LA

class TrainModel(): 
    
    def __init__(self, graph, model, args, criterion=None):
        self.args = args
        self.epoch = 0
        self.graph = graph.to(self.args.device)
        self.model = model.to(self.args.device)
        if model.map.shape[0] == 2:
            self.mask = self.keepX(self.model.method, self.model.map.iloc[1])
        else:
            self.mask = self.convert_map(self.model.method, self.model.map)
            # map[next(iter(map))] = torch.cat((map[next(iter(map))], torch.tensor(np.zeros((self.model.units[1], self.model.units[0]-list(self.model.map.nunique())[0])))), 1)
            # self.mask = map
        
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
                self.epoch = epoch
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
        if self.model.map.shape[0] == 2:
            for name, param in self.model.named_parameters():
                if re.search("lin.weight", name):
                    weights = param.cpu().clone()
                    weights = self.soft_thresholding(weights, self.mask[name].item())
                    # weights = weights.detach().numpy()
                    #weights = self.l2(weights)
                    #weights = weights/LA.matrix_norm(weights)
                    self.model.state_dict()[name].data.copy_(weights)
        else:
            for name, param in self.model.named_parameters():
                if re.search("lin.weight", name):
                    weights = param.cpu().clone() * self.mask[name]
                    # weights = weights.detach().numpy()
                    #weights = self.l2(weights)
                    weights = weights/LA.matrix_norm(weights)
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

    def keepX(self, model_name, map):
        if model_name == "GCN":
            return dict([(''.join(["layers.", str(i), ".0.lin.weight"]), map[i]) for i in range(len(map)-1)])
        else:
            return dict([(''.join(["layers.", str(i), ".0.weight"]), map[i]) for i in range(len(map)-1)])

    def func(self, x, keep=2):
        x1 = torch.zeros(len(x))
        x1[torch.argsort(abs(x), descending = False)[0:keep]]=1
        return x1*x

    def soft_thresholding(self, weights, keepX = 2):
        for i in list(range(weights.shape[0])):
          weights[i,:] = self.func(weights[i,:], keepX)
          #weights[i,:] = weights[i,:]/torch.linalg.norm(weights[i,:])
        return weights
    
    def l2(self, w):
        for i in list(range(w.shape[0])):
          w[i,:] = w[i,:]/torch.linalg.norm(w[i,:])
        return w