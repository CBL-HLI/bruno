import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, Linear
import torch.nn.functional as F
import pandas as pd

class Encoder(nn.Module):
    def __init__(
        self,
        map,
        args,
        bias: bool = True
    ):
        super().__init__()
        self.map = map
        self.map_f = self.map.apply(lambda x: pd.factorize(x)[0])
        self.args = args
        if self.map.shape[0] == 2:
            units = list(self.map_f.to_numpy()[0])
            units[0] = self.args.num_node_features
            self.units = units
        else:
            units = list(self.map_f.nunique())
            units[0] = self.args.num_node_features
            self.units = units
        self.bias = bias
        self.transition = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=self.args.dropout)
        ) 

        self.modules = []
        for i in range(len(self.units)-1):
            if self.args.method == "GCNConv":
                self.modules.append(nn.Sequential(
                                GCNConv(
                                    self.units[i],
                                    self.units[i+1],
                                    bias= self.bias,
                                )
                            , nn.BatchNorm1d(self.units[i+1])
                        ))
            elif self.args.method == "GATConv":
                self.modules.append(nn.Sequential(
                                GATConv(
                                    self.units[i],
                                    self.units[i+1],
                                    bias= self.bias,
                                    heads=1
                                )
                            , nn.BatchNorm1d(self.units[i+1])
                        ))
            elif self.args.method == "MLP":
                self.modules.append(nn.Sequential(
                                Linear(
                                    self.units[i],
                                    self.units[i+1],
                                    bias= self.bias,
                                )
                            , nn.BatchNorm1d(self.units[i+1])
                        ))
            else:
                raise ValueError('args.type shoud be one of "GCNConv", "GATConv", or "MLP"')
        if self.args.num_classes is not None:
            self.modules.append(nn.Sequential(nn.Linear(self.units[-1], self.args.num_classes, bias = self.bias)))
        self.layers = nn.Sequential(*self.modules)
    
    def forward(self, data, edge_index = None):
        x = data
        if self.args.method == "MLP":
            outputs = []
            for i, layers in enumerate(self.layers):
                for layer in layers:
                    if i == (len(layers)-1):
                        x = layer(x)
                    else:
                        x = F.relu(layer(x))
                        # x = self.transition(x)
                ## save embeddings
                outputs.append(x.cpu().detach().numpy())
        else:
            outputs = []
            for i, layers in enumerate(self.layers):
                for layer in layers:
                    if isinstance(layer, nn.Linear) or isinstance(layer, nn.BatchNorm1d):
                        if i == (len(layers)-1):
                            x = layer(x)
                        else:
                            x = F.relu(layer(x))
                            # x = self.transition(x)
                    else:
                        x = F.relu(layer(x, edge_index))
                        # x = self.transition(x)
                ## save embeddings
                outputs.append(x.cpu().detach().numpy())
        return x, outputs

class Decoder(nn.Module):
    def __init__(
        self,
        map,
        args,
        bias: bool = True
    ):
        super().__init__()
        self.map = map
        self.map_f = self.map.apply(lambda x: pd.factorize(x)[0])
        self.args = args
        if self.map.shape[0] == 2:
            units = list(self.map_f.to_numpy()[0])
            units[0] = self.args.num_node_features
            self.units = units
        else:
            units = list(self.map_f.nunique())
            units[0] = self.args.num_node_features
            self.units = units
        self.bias = bias 

        self.modules = []
        for i in reversed(range(len(self.units)-1)):
            if self.args.method == "GCNConv":
                self.modules.append(nn.Sequential(
                                GCNConv(
                                    self.units[i+1],
                                    self.units[i],
                                    bias= self.bias,
                                )
                            , nn.BatchNorm1d(self.units[i], momentum=0.01, eps=0.001)
                        ))
            elif self.args.method == "GATConv":
                self.modules.append(nn.Sequential(
                                GATConv(
                                    self.units[i+1],
                                    self.units[i],
                                    bias= self.bias,
                                    heads=1
                                )
                            , nn.BatchNorm1d(self.units[i], momentum=0.01, eps=0.001)
                        ))
            elif self.args.method == "MLP":
                self.modules.append(nn.Sequential(
                                Linear(
                                    self.units[i+1],
                                    self.units[i],
                                    bias= self.bias,
                                )
                            , nn.BatchNorm1d(self.units[i], momentum=0.01, eps=0.001)
                        ))
            else:
                raise ValueError('args.type shoud be one of "GCNConv", "GATConv", or "MLP"')
        # if self.args.num_classes is not None:
        #     self.modules.append(nn.Sequential(nn.Linear(self.units[-1], self.args.num_classes, bias = self.bias)))
        self.layers = nn.Sequential(*self.modules)
    
    def forward(self, data, edge_index = None):
        if self.args.method == "MLP":
            x = data
            outputs = []
            for i, layers in enumerate(self.layers):
                for layer in layers:
                    if i == (len(layers)-1):
                        x = layer(x)
                    else:
                        x = F.relu(layer(x))
                ## save embeddings
                outputs.append(x.cpu().detach().numpy())
        else:
            x = data
            outputs = []
            for i, layers in enumerate(self.layers):
                for layer in layers:
                    if isinstance(layer, nn.Linear) or isinstance(layer, nn.BatchNorm1d):
                        if i == (len(layers)-1):
                            x = layer(x)
                        else:
                            x = F.relu(layer(x))
                    else:
                        x = F.relu(layer(x, edge_index))
                ## save embeddings
                outputs.append(x.cpu().detach().numpy())
        return x, outputs

class AE(nn.Module):
    def __init__(self, map, args):
        super().__init__()
        self.map = map
        self.args = args
        self.args.num_classes = None
        self.encoder = Encoder(self.map, self.args)
        self.decoder = Decoder(self.map, self.args)

    def forward(self, data, edge_index = None):
        x = data
        if self.args.method == "MLP":
            x, encoded_embeddings = self.encoder(x)
            x, decoded_embeddings = self.decoder(x)
        else:
            x, encoded_embeddings = self.encoder(x, edge_index)
            x, decoded_embeddings = self.decoder(x, edge_index)
        return x, encoded_embeddings