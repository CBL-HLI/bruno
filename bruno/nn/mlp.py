import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

class MLP(nn.Module):
    def __init__(
        self,
        map,
        args,
        bias: bool = True
    ):
        super().__init__()
        self.method = "MLP"
        self.map = map
        self.map_f = self.map.apply(lambda x: pd.factorize(x)[0])
        self.args = args
        if self.map_f.shape[0] == 2:
            units = list(self.map_f.to_numpy()[0])
            units[0] = self.args.num_node_features
            self.units = units
        else:
            units = list(self.map_f.nunique())
            units[0] = self.args.num_node_features
            self.units = units
        self.bias = bias
        self.modules = [
                        nn.Sequential(
                            nn.Linear(
                                self.units[i],
                                self.units[i+1],
                                bias= self.bias,
                            ),
                            nn.BatchNorm1d(self.units[i+1], momentum=0.01, eps=0.001)
                        )
                    for name, i in zip(self.map_f.keys(), range(len(self.units)-1))
                ]
        self.modules.append(nn.Sequential(nn.Linear(self.units[-1], self.args.num_classes, bias = self.bias)))
        self.layers = nn.Sequential(*self.modules)
    
    def forward(self, data):
        x = data.x
        outputs = []
        for i, layers in enumerate(self.layers):
            for layer in layers:
                if i == (len(layers)-1):
                    x = layer(x)
                else:
                    x = F.relu(layer(x))
            ## save embeddings
            outputs.append(x.cpu().detach().numpy())
        return x, outputs