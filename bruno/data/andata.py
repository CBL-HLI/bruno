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
                adjMat = False, 
                adata: AnnData = None):
        super().__init__(root, transform)

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
            adj = csr_matrix(np.transpose(pd.get_dummies(adata.obs)).corr()).tocoo()
        
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        # y
        le = preprocessing.LabelEncoder()
        y = torch.as_tensor(le.fit_transform(adata.obs[group].values))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.transform is None else self.transform(data)

        self.data, self.slices = self.collate([data])


def weight_matrix(p1, p2):
  W1 = np.zeros((p1, p2))
  divisor = int(p1/p2)
  indx = list(range(W1.shape[1]))
  for col in indx:
    row = divisor*col
    W1[row:(row+divisor), col] = np.random.rand(divisor)
  return W1

def simulate_data(map, n, means, stds, nn):
    r"""Generate graph data

    Y = AAXW0W1W2
    Y = AH1W1W2; H1=AXW0
    Y = H2W2; H2=AH1W1

    Args:
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)

    """
    # compute sample representations
    Y_true, y = datasets.make_blobs(n_samples=2*n,
                                      centers=2,
                                      cluster_std = stds, 
                                      center_box = means,
                                      n_features=2,
                                      random_state=170)
    Y_true = preprocessing.scale(Y_true)

    # plot graph
    G = gt.Graph(Y_true, knn=nn)

    # This code draws the nodes of the graph
    title = 'mu1='+str(means[0])+' mu2='+str(means[1])
    ax = scprep.plot.scatter2d(Y_true, c=y, 
                              xlabel='Component 1', ylabel='Component 2',
                              title=title, figsize=(8,6),
                              legend=False)

    # This block of code draws lines between each point and its neighbors
    for i in range(Y_true.shape[0]-1):
        for j in range(i+1, Y_true.shape[0]):
            affinity = G.kernel[i,j]
            if affinity > 0:
                ax.plot(Y_true[[i,j],0], Y_true[[i,j],1],
                        color='black', zorder=0)
        
    ax.set_aspect(1)
    #plt.tight_layout()

    # Compute adjacency matrix
    A = G.kernel.toarray()
    #A[A > 0] = 1
    Ahat = A + np.eye(A.shape[0])
    D = sp.sparse.diags(np.power(np.array(Ahat.sum(1)), -0.5).flatten(), 0)
    A_norm = np.matmul(np.matmul(D.toarray(), Ahat), D.toarray())

    # compute weight matrices
    map = map.apply(lambda x: pd.factorize(x)[0])
    p = map.nunique().tolist()
    Wlist=list()
    for i in range(len(p)-1):
      W0 = weight_matrix(p[i], p[i+1])
      W0 = W0/np.linalg.norm(W0, axis=0, keepdims=1)
      Wlist.append(W0)
    W = np.linalg.multi_dot(Wlist)
    W = W/np.linalg.norm(W, axis=0, keepdims=1)

    # Compute YW.T=AAX
    AAX = np.matmul(preprocessing.scale(Y_true), W.T)+ np.random.normal(0, 0.5, size=(A_norm.shape[0], W.shape[0]))

    # Compute X
    AA_inv = np.linalg.inv(np.matmul(A_norm, A_norm))
    X = np.matmul(AA_inv, AAX)
    #X = cX)

    return A, A_norm, AAX, Wlist, W, X, y