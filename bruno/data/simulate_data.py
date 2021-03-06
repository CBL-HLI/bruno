import numpy as np
from sklearn import datasets, preprocessing
import graphtools as gt
import scipy as sp
import scprep as scprep
import pandas as pd

def weight_matrix(p1, p2):
  W1 = np.zeros((p1, p2))
  divisor = int(p1/p2)
  indx = list(range(W1.shape[1]))
  for col in indx:
    row = divisor*col
    W1[row:(row+divisor), col] = np.random.rand(divisor)
  return W1

def simulate_data(map, n, means, stds, nn, noise, plot=False):
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
                                      n_features=2)
    #Y_true = Y_true
    Y_true = preprocessing.scale(Y_true)

    # plot graph
    G = gt.Graph(Y_true, knn=nn)

    # This code draws the nodes of the graph
    if plot:
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
    Ahat = A + np.eye(A.shape[0])
    D = sp.sparse.diags(np.power(np.array(Ahat.sum(1)), -0.5).flatten(), 0)
    A_norm = np.matmul(np.matmul(D.toarray(), Ahat), D.toarray())

    # compute weight matrices
    map = map.apply(lambda x: pd.factorize(x)[0])
    p = map.nunique().tolist()
    p.append(2)
    Wlist=list()
    for i in range(len(p)-1):
      W0 = weight_matrix(p[i], p[i+1])
      W0 = W0/np.linalg.norm(W0, axis=0, keepdims=1)
      Wlist.append(W0)
    W = np.linalg.multi_dot(Wlist)
    W = W/np.linalg.norm(W, axis=0, keepdims=1)
    W = np.linalg.pinv(W)

    # Compute YW.T=AAX
    AAX = preprocessing.scale(np.matmul(Y_true, W) + np.random.normal(0, noise, size=(A_norm.shape[0], W.shape[1])))

    # Compute X
    AA_inv = np.linalg.inv(np.matmul(A_norm, A_norm))
    X = np.matmul(AA_inv, AAX)
    #X = cX)

    return A, G, A_norm, AAX, Wlist, W, X, y