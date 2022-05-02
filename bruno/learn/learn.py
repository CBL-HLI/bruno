import torch
import torch.nn as nn
import numpy as np
import re
from torch import linalg as LA
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scprep as scprep
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
import plotly.graph_objects as go
import pandas as pd
import torch.nn.functional as F

import warnings
warnings.simplefilter("ignore")

# https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            # print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                # print('INFO: Early stopping')
                self.early_stop = True

class TrainModel(): 
    
    def __init__(self, graph, model, args):
        self.args = args
        self.graph = graph.to(self.args.device)
        self.model = model.to(self.args.device)
        self.early_stopping = EarlyStopping(args.patience, args.min_delta)
        self.mask = self.convert_map(self.model, self.args)
        if self.args.num_classes == None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.w_decay)
        
        self.train_loss = []
        self.val_loss = []
        self.train_complete = False
        self.cim = None
        self.weights = []
        
    def learn(self) -> None:
        for epoch in range(self.args.epochs): 
            if self.train_complete: return
            
            tl = self.train_epoch()
            self.train_loss.append(tl)
            
            vl = self.val()
            self.val_loss.append(vl)
            self.early_stopping(vl)
            if self.early_stopping.early_stop:
                self.train_complete = True
                print('Epoch:' + str(epoch), 'Training loss: ' + str(tl))
                print('Epoch:' + str(epoch), 'Validation loss: ' + str(vl))

        self.train_complete = True
        print('Epoch:' + str(epoch+1), 'Training loss: ' + str(round(tl)))
        print('Epoch:' + str(epoch+1), 'Validation loss: ' + str(round(vl)))
        
    def train_epoch(self) -> float:
        self.model.train()
        y_true = self.graph.y[self.graph.train_mask]
        self.optim.zero_grad()
        y_pred, outputs = self.model(self.graph.x, self.graph.edge_index) 
        loss = self.criterion(y_pred[self.graph.train_mask], y_true)
        loss.backward()
        self.optim.step()
        w = {}
        for name, param in self.model.named_parameters():
            if self.args.method == "GCNConv":
                if re.search(".0.lin.weight", name):
                    weights = param.cpu().clone()
                    weights = weights * self.mask[name]
                    weights = F.normalize(weights, p=2, dim=1)
                    w[name] = weights
                    self.model.state_dict()[name].data.copy_(weights)
            elif self.args.method == "GATConv":
                if re.search(".0.lin_src.weight", name):
                    weights = param.cpu().clone()
                    weights = weights * self.mask[name]
                    weights = F.normalize(weights, p=2, dim=1)
                    w[name] = weights
                    self.model.state_dict()[name].data.copy_(weights)
            else:
                if re.search("0.weight", name):
                    weights = param.cpu().clone()
                    weights = weights * self.mask[name]
                    weights = F.normalize(weights, p=2, dim=1)
                    w[name] = weights
                    self.model.state_dict()[name].data.copy_(weights)
        self.weights.append(w)
        return loss.data.item()
    
    def val(self) -> float:
        # returns the validation loss 
        self.model.eval()
        labels = self.graph.y[self.graph.val_mask]
        output, outputs = self.model(self.graph.x, self.graph.edge_index) 
        loss = self.criterion(output[self.graph.val_mask], labels)
        return loss.data.item()

    def test(self) -> float: 
        # returns the test accuracy 
        if not self.train_complete: 
            self.learn()
        self.model.eval()
        y_true = self.graph.y[self.graph.test_mask]
        output, outputs = self.model(self.graph.x, self.graph.edge_index)
        _, y_pred = output.max(dim=1)
        output = output[self.graph.test_mask].cpu().detach().numpy()
        y_pred = y_pred[self.graph.test_mask].cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        return output, y_true, y_pred
    
    def plot_loss(self) -> None:
        if not self.train_complete: 
            self.learn()
        if self.args.num_classes == None:
            self.loss("Mean Square Error")
        else:
            self.loss("Cross Entrophy Loss")
          
    def loss(self, loss_name) -> None:
        plt.plot(self.train_loss, color='r')
        plt.plot(self.val_loss, color='b')
        plt.yscale('log',basey=10)
        plt.xscale('log')
        plt.xlabel("epoch")
        plt.ylabel(loss_name)
        plt.legend(['Training loss', 'Validation loss'])

    def weights(self, map, index):
        w = map[[str(index), str(index+1)]].drop_duplicates()
        w = w.rename(columns = {str(index): '0', str(index+1):'1'})
        w.insert(2, "values", 1)
        return w.pivot(index=['0'], columns=['1']).fillna(0)
    
    def convert_map(self, model, args):
        map = model.map_f
        method = args.method
        n_classes = args.num_classes
        p = map.nunique().tolist()
        if method == "GCNConv":
            mask = dict([(''.join(["layers.", str(i), ".0.lin.weight"]), torch.tensor(np.transpose(self.weights(map, i).to_numpy()))) for i in range(len(map.columns)-1)])
        elif method == "GATConv":
            mask = dict([(''.join(["layers.", str(i), ".0.lin_src.weight"]), torch.tensor(np.transpose(self.weights(map, i).to_numpy()))) for i in range(len(map.columns)-1)])
        else:
            mask = dict([(''.join(["layers.", str(i), ".0.weight"]), torch.tensor(np.transpose(self.weights(map, i).to_numpy()))) for i in range(len(map.columns)-1)])
        if n_classes is not None:
            mask['layers.'+str(len(mask))+'.0.weight'] = torch.ones(n_classes, p[len(p)-1])
        else:
            ## get weight names
            weight_names = []
            for name, param in model.named_parameters():
                if method == "GCNConv":
                    if re.search(".0.lin.weight", name):
                        weight_names.append(name)
                elif method == "GATConv":
                    if re.search(".0.lin_src.weight", name):
                        weight_names.append(name)
                else:
                    if re.search("0.weight", name):
                        weight_names.append(name)
            mask0 = {}
            for i, key in enumerate(mask.keys()):
                mask0[weight_names[i]] = mask[key]
            mask = mask0
        return mask

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
        return weights
    
    def l2(self, w):
        for i in list(range(w.shape[0])):
          w[i,:] = w[i,:]/torch.linalg.norm(w[i,:])
        return w
    
    def metrics(self) -> float:
        if not self.train_complete: 
            self.learn()
        metrics = self.compute_metrics()
        return metrics

    def compute_metrics(self) -> None:
        output, y_true, y_pred = self.test()
        if self.args.num_classes == None:
            met = pd.DataFrame({'method': self.args.method,
                                'mse': [metrics.mean_squared_error(y_true, output)]})
        else:
            try:
                if self.args.num_classes == 2:
                    auc = metrics.roc_auc_score(y_true, output[:,1])
                    classes = np.array(list(set(y_true)))
                    cm = metrics.confusion_matrix(y_true, y_pred, labels=classes)
                    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                                  display_labels=classes)
                    disp.plot()
                else:
                    auc = 'Cannot be computed'
            except ValueError:
                auc = 'Cannot be computed'
            met = pd.DataFrame({'method': self.args.method,
                                'auc': [auc],
                  'bacc': [metrics.balanced_accuracy_score(y_true, y_pred)]})
        return met
    
    def plot_pca(self) -> None:
        if not self.train_complete: 
            self.learn()
        self.pca()
    
    def plot_tsne(self) -> None:
        if not self.train_complete: 
            self.learn()
        self.tsne()
    
    def pca(self) -> None:
        pred, outputs = self.model(self.graph.x, self.graph.edge_index)
        ## compute PCA of embeddings
        embeddings = {}
        pca = PCA(n_components=2)
        embeddings['inputs'] = pca.fit_transform(self.graph.x.cpu().detach().numpy()[self.graph.test_mask.cpu().detach().numpy()])
        for i, output in enumerate(outputs):
            pca = PCA(n_components=2)
            h = output[self.graph.test_mask.cpu().detach().numpy()]
            if i == (len(outputs)-1):
                embeddings['output'] = pca.fit_transform(h)
            else:
                embeddings['layer'+str(i+1)] = pca.fit_transform(h)
        ## plot embeddings
        fig, axes = plt.subplots(1,len(embeddings),figsize=(12,2))
        col = self.graph.y[self.graph.test_mask].cpu().detach().numpy()
        for i, dataset_name in enumerate(embeddings):
            ax = axes[i]
            pcs = embeddings[dataset_name]
            scprep.plot.scatter2d(pcs, c=col,
                                  ticks=None, ax=ax, 
                                  xlabel='PC1', ylabel='PC2',
                                  title=dataset_name,
                                legend=False)
            
        fig.tight_layout()

    def tsne(self) -> None:
        # make predictions using trained model
        pred, outputs = self.model(self.graph.x, self.graph.edge_index)
        ## compute TSNE of embeddings
        embeddings = {}
        tsne = TSNE(n_components=2)
        embeddings['inputs'] = tsne.fit_transform(self.graph.x.cpu().detach().numpy()[self.graph.test_mask.cpu().detach().numpy()])
        for i, output in enumerate(outputs):
            tsne = TSNE(n_components=2)
            h = output[self.graph.test_mask.cpu().detach().numpy()]
            if i == (len(outputs)-1):
                embeddings['output'] = tsne.fit_transform(h)
            else:
                embeddings['layer'+str(i+1)] = tsne.fit_transform(h)
        ## plot embeddings
        fig, axes = plt.subplots(1,len(embeddings),figsize=(12,2))
        col = self.graph.y[self.graph.test_mask].cpu().detach().numpy()
        for i, dataset_name in enumerate(embeddings):
            ax = axes[i]
            pcs = embeddings[dataset_name]
            scprep.plot.scatter2d(pcs, c=col,
                                  ticks=None, ax=ax, 
                                  xlabel='TSNE1', ylabel='TSNE2',
                                  title=dataset_name,
                                legend=False)    
        fig.tight_layout()
    
    
    def get_weights(self) -> float:
        if not self.train_complete: 
              self.learn()
        w={}
        for name, param in self.model.named_parameters():
                #if re.search("lin.weight", name):
                w[name] = param
        return w
      
    def plot_weights_of_last_layer(self, figsize=(10, 5)) -> None:
        if not self.train_complete: 
            self.learn()
        w = self.get_weights()
        cim=w[list(w.keys())[-1]].cpu().detach().numpy()
        self.cim = pd.DataFrame(cim, columns=list(self.model.map.iloc[:, -1].unique()))
        f, ax = plt.subplots(figsize=figsize)
        ax = sns.heatmap(self.cim, linewidths=.5, cmap="YlGnBu", cbar_kws={'label': "Variable importance"})
        ax.set(xlabel='top-level layer', ylabel='Class')

    def plot_subnetwork(self, names_df=None, pathway=None, figheight=500, figwidth=700):
        layerlist = list(self.model.map.columns)[1:]
        top_layer = layerlist[-1]
        cat_cols = layerlist.copy()
        layerlist.reverse()
        df = pd.DataFrame(self.model.map[self.model.map[top_layer] == pathway]).groupby(layerlist).size()
        df = df.reset_index()
        if names_df is not None:
            df = df.apply(lambda x: [names_df[names_df['reactome_id'] == x]['pathway_name'].to_numpy()[0] if type(x) is not int else x for x in x], axis=0)

        df_sankey = genSankey(df, cat_cols=cat_cols, value_cols=0)
        fig = go.Figure(data=[go.Sankey(
            node = df_sankey['data'][0]['node'],
            link = df_sankey['data'][0]['link'])])
        fig.update_layout(height=figheight, width=figwidth, title_text=pathway+' ('+pathway+')')
        fig.show()


# https://medium.com/kenlok/how-to-create-sankey-diagrams-from-dataframes-in-python-e221c1b4d6b0
def genSankey(df,cat_cols=[],value_cols='',title='Sankey Diagram'):
    # maximum of 6 value cols -> 6 colors
    colorPalette = ['#4B8BBE','#306998','#FFE873','#FFD43B','#646464']
    labelList = []
    colorNumList = []
    for catCol in cat_cols:
        labelListTemp =  list(set(df[catCol].values))
        colorNumList.append(len(labelListTemp))
        labelList = labelList + labelListTemp
        
    # remove duplicates from labelList
    labelList = list(dict.fromkeys(labelList))
    
    # define colors based on number of levels
    colorList = []
    for idx, colorNum in enumerate(colorNumList):
        colorList = colorList + [colorPalette[idx]]*colorNum
        
    # transform df into a source-target pair
    for i in range(len(cat_cols)-1):
        if i==0:
            sourceTargetDf = df[[cat_cols[i],cat_cols[i+1],value_cols]]
            sourceTargetDf.columns = ['source','target','count']
        else:
            tempDf = df[[cat_cols[i],cat_cols[i+1],value_cols]]
            tempDf.columns = ['source','target','count']
            sourceTargetDf = pd.concat([sourceTargetDf,tempDf])
        sourceTargetDf = sourceTargetDf.groupby(['source','target']).agg({'count':'sum'}).reset_index()
        
    # add index for source-target pair
    sourceTargetDf['sourceID'] = sourceTargetDf['source'].apply(lambda x: labelList.index(x))
    sourceTargetDf['targetID'] = sourceTargetDf['target'].apply(lambda x: labelList.index(x))
    
    # creating the sankey diagram
    data = dict(
        type='sankey',
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(
            color = "black",
            width = 0.5
          ),
          label = labelList,
          color = colorList
        ),
        link = dict(
          source = sourceTargetDf['sourceID'],
          target = sourceTargetDf['targetID'],
          value = sourceTargetDf['count']
        )
      )
    
    layout =  dict(
        title = title,
        font = dict(
          size = 10
        )
    )
       
    fig = dict(data=[data], layout=layout)
    return fig

