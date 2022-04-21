from typing import Callable, Optional
from anndata import AnnData
import numpy as np
import pandas as pd
import matplotlib as plt
import torch

from torch_geometric.data import Data, InMemoryDataset
from sklearn import preprocessing
import anndata as ad
from scipy.sparse import coo_matrix, csr_matrix

import graphtools as gt
from sklearn import datasets, preprocessing
import scipy as sp
import scprep as scprep
import graphtools as gt

class AnnDataToGraphData(InMemoryDataset):
    r"""Convert AnnData Object to PyTorch Geometric Data

    Args:
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)

    """
    def __init__(self, 
                root: str,
                transform: Optional[Callable] = None, 
                group = None, 
                adjMat = None, 
                adata: AnnData = None,
                knn = 3):
        super().__init__(root, transform, adata)

        # Node features
        coo = coo_matrix(adata.X, shape=adata.X.shape)
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        x = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

        # Adjacency matrix
        if adjMat is not None:
            adj = csr_matrix(adjMat).tocoo()
        else:
            #adj = csr_matrix(np.transpose(pd.get_dummies(adata.obs)).corr()).tocoo()
            Y_true = adata.obs[list(set(adata.obs_keys()).difference(set([group])))]
            G = gt.Graph(Y_true, knn=knn)
            adj = csr_matrix(G.kernel.toarray()).tocoo()

        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        # y
        le = preprocessing.LabelEncoder()
        y = torch.as_tensor(le.fit_transform(adata.obs[group].values))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.transform is None else self.transform(data)

        self.data, self.slices = self.collate([data])

class PrepareAnnData():
    def __init__(self, 
                 data,
                 obs_vars,
                 map):
        self.data = data
        self.obs_vars = obs_vars
        self.map = map
    
    def anndata(self) -> float:
        com_genes = set(self.data.columns).intersection(self.map['layer0'])
        eset_subset = self.data[com_genes]
        map = self.map.loc[self.map['layer0'].isin(com_genes)]
        map.columns = [i for i in range(len(map.columns))]
        map.index = map.index.map(str)
        map.columns = map.columns.map(str)

        obs_meta = pd.DataFrame(self.data[self.obs_vars].values, columns=self.obs_vars)
        adata = ad.AnnData(eset_subset.to_numpy(), obs=obs_meta)

        return adata, map