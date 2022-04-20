import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

## modified from https://github.com/scverse/scvi-tools/blob/master/scvi/nn/_base_components.py

class BRUNO(nn.Module):
    def __init__(
        self,
        map,
        args,
        bias: bool = True
    ):
        super().__init__()
        self.method = "GCN"
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
        self.modules = [
                        nn.Sequential(
                            GCNConv(
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
        x, edge_index = data.x, data.edge_index
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