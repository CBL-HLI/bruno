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

import warnings
warnings.simplefilter("ignore")


class TrainModel(): 
    
    def __init__(self, graph, model, args, criterion=None):
        self.args = args
        self.last_loss = self.args.last_loss
        self.trigger_times = self.args.trigger_times
        self.graph = graph.to(self.args.device)
        self.model = model.to(self.args.device)
        if model.map.shape[0] == 2:
            self.mask = self.keepX(self.model.method, self.model.map.iloc[1])
        else:
            self.mask = self.convert_map(self.model.method, self.model.map_f)
        
        if not criterion: 
            criterion = nn.CrossEntropyLoss()
        self.criterion = criterion
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.w_decay)
        
        self.train_loss = []
        self.val_loss = []
        self.train_complete = False
        self.cim = None
        
    def learn(self) -> None:
        # tracks training and validation loss over epochs
        # can add early stopping mechanism by comparing losses
        for epoch in range(self.args.epochs): 
            if self.train_complete: return
            
            tl = self.train_epoch()
            self.train_loss.append(tl)
            
            vl = self.val()
            self.val_loss.append(vl)

            # Early stopping
            print('Epoch:' + str(epoch), 'Training loss: ' + str(tl))
            print('Epoch:' + str(epoch), 'Validation loss: ' + str(vl))

            if vl > self.last_loss:
              self.trigger_times += 1
              if self.trigger_times >= self.args.patience:
                print('Early stopping')
                self.train_complete = True
            else:
              self.trigger_times = 0
            self.last_loss = vl
        self.train_complete = True
        
    def train_epoch(self) -> float:
        # trains a single epoch (ie. one pass over the full graph) and updates the models parameters
        # returns the loss
        self.model.train()
        labels = self.graph.y[self.graph.train_mask]
        self.optim.zero_grad()
        output, outputs = self.model(self.graph) 
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
                    #weights = weights/LA.matrix_norm(weights)
                    self.model.state_dict()[name].data.copy_(weights)

        return loss.data.item()
    
    def val(self) -> float:
        # returns the validation loss 
        self.model.eval()
        labels = self.graph.y[self.graph.val_mask]
        output, outputs = self.model(self.graph) 
        loss = self.criterion(output[self.graph.val_mask], labels)
        return loss.data.item()

    def test(self) -> float: 
        # returns the test accuracy 
        if not self.train_complete: 
            self.learn()
        self.model.eval()
        labels = self.graph.y[self.graph.test_mask]
        pred, outputs = self.model(self.graph)
        _, truth = pred.max(dim=1)
        correct = float ( truth[self.graph.test_mask].eq(labels).sum().item() )
        acc = correct / self.graph.test_mask.sum().item()
        pred = pred[self.graph.test_mask].cpu().detach().numpy()
        truth = truth[self.graph.test_mask].cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        return pred, truth, labels, acc

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
    
    def metrics(self) -> float:
        if self.train_complete:
            labels = self.graph.y[self.graph.test_mask]
            pred, outputs = self.model(self.graph)
            _, truth = pred.max(dim=1)
            met = pd.DataFrame({'precision': [metrics.precision_score(truth, labels)],
                  'recall': [metrics.recall_score(truth, labels)],
                  'auc': [metrics.roc_auc_score(truth, labels)],
                  'bacc': [metrics.balanced_accuracy_score(truth, labels)]})
        else:
            pred, truth, labels, acc = self.test()
            met = pd.DataFrame({'precision': [metrics.precision_score(truth, labels)],
                  'recall': [metrics.recall_score(truth, labels)],
                  'auc': [metrics.roc_auc_score(truth, labels)],
                  'bacc': [metrics.balanced_accuracy_score(truth, labels)]})
        return met
    
    def plot_pca(self) -> None:
        if self.train_complete:
            # make predictions using trained model
            pred, outputs = self.model(self.graph)
            ## compute PCA of embeddings
            embeddings = {}
            pca = PCA(n_components=2)
            embeddings['inputs'] = pca.fit_transform(self.graph.x.cpu().detach().numpy()[self.graph.test_mask.cpu().detach().numpy()])
            for i, output in enumerate(outputs):
                pca = PCA(n_components=2)
                h = output[data.test_mask.cpu().detach().numpy()]
                if i == (len(outputs)-1):
                    embeddings['output'] = pca.fit_transform(h)
                else:
                    embeddings['layer'+str(i+1)] = pca.fit_transform(h)
            ## plot embeddings
            fig, axes = plt.subplots(1,len(embeddings),figsize=(12,2))
            col = data.y[data.test_mask].cpu().detach().numpy()
            for i, dataset_name in enumerate(embeddings):
                ax = axes[i]
                pcs = embeddings[dataset_name]
                scprep.plot.scatter2d(pcs, c=col,
                                      ticks=None, ax=ax, 
                                      xlabel='PC1', ylabel='PC2',
                                      title=dataset_name,
                                    legend=False)
        else:     
            # train model
            self.test()
            # make predictions using trained model
            pred, outputs = self.model(self.graph)
            ## compute PCA of embeddings
            embeddings = {}
            pca = PCA(n_components=2)
            embeddings['inputs'] = pca.fit_transform(self.graph.x.cpu().detach().numpy()[self.graph.test_mask.cpu().detach().numpy()])
            for i, output in enumerate(outputs):
                pca = PCA(n_components=2)
                h = output[data.test_mask.cpu().detach().numpy()]
                if i == (len(outputs)-1):
                    embeddings['output'] = pca.fit_transform(h)
                else:
                    embeddings['layer'+str(i+1)] = pca.fit_transform(h)
            ## plot embeddings
            fig, axes = plt.subplots(1,len(embeddings),figsize=(12,2))
            col = data.y[data.test_mask].cpu().detach().numpy()
            for i, dataset_name in enumerate(embeddings):
                ax = axes[i]
                pcs = embeddings[dataset_name]
                scprep.plot.scatter2d(pcs, c=col,
                                      ticks=None, ax=ax, 
                                      xlabel='PC1', ylabel='PC2',
                                      title=dataset_name,
                                    legend=False)
                
            fig.tight_layout()

    def plot_tsne(self) -> None:
        if self.train_complete:
            # make predictions using trained model
            pred, outputs = self.model(self.graph)
            ## compute TSNE of embeddings
            embeddings = {}
            tsne = TSNE(n_components=2)
            embeddings['inputs'] = tsne.fit_transform(self.graph.x.cpu().detach().numpy()[self.graph.test_mask.cpu().detach().numpy()])
            for i, output in enumerate(outputs):
                tsne = TSNE(n_components=2)
                h = output[data.test_mask.cpu().detach().numpy()]
                if i == (len(outputs)-1):
                    embeddings['output'] = tsne.fit_transform(h)
                else:
                    embeddings['layer'+str(i+1)] = tsne.fit_transform(h)
            ## plot embeddings
            fig, axes = plt.subplots(1,len(embeddings),figsize=(12,2))
            col = data.y[data.test_mask].cpu().detach().numpy()
            for i, dataset_name in enumerate(embeddings):
                ax = axes[i]
                pcs = embeddings[dataset_name]
                scprep.plot.scatter2d(pcs, c=col,
                                      ticks=None, ax=ax, 
                                      xlabel='TSNE1', ylabel='TSNE2',
                                      title=dataset_name,
                                    legend=False)              
            fig.tight_layout()
        else:
            # train model
            self.test()
            # make predictions using trained model
            pred, outputs = self.model(self.graph)
            ## compute TSNE of embeddings
            embeddings = {}
            tsne = TSNE(n_components=2)
            embeddings['inputs'] = tsne.fit_transform(self.graph.x.cpu().detach().numpy()[self.graph.test_mask.cpu().detach().numpy()])
            for i, output in enumerate(outputs):
                tsne = TSNE(n_components=2)
                h = output[data.test_mask.cpu().detach().numpy()]
                if i == (len(outputs)-1):
                    embeddings['output'] = tsne.fit_transform(h)
                else:
                    embeddings['layer'+str(i+1)] = tsne.fit_transform(h)
            ## plot embeddings
            fig, axes = plt.subplots(1,len(embeddings),figsize=(12,2))
            col = data.y[data.test_mask].cpu().detach().numpy()
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
        w={}
        if self.train_complete:
            for name, param in self.model.named_parameters():
                #if re.search("lin.weight", name):
                w[name] = param
        else:
            self.test()
            for name, param in self.model.named_parameters():
                #if re.search("lin.weight", name):
                w[name] = param
        return w
      
    def plot_weights_of_last_layer(self, figsize=(10, 5)) -> None:
        if self.train_complete:
            w = self.get_weights()
            cim=w[list(w.keys())[-1]].cpu().detach().numpy()
            self.cim = pd.DataFrame(cim, columns=list(self.model.map.iloc[:, -1].unique()))
            f, ax = plt.subplots(figsize=figsize)
            ax = sns.heatmap(self.cim, linewidths=.5, cmap="YlGnBu", cbar_kws={'label': "Variable importance"})
            ax.set(xlabel='top-level layer', ylabel='Class')
        else:
            self.test()
            w = self.get_weights()
            cim=w[list(w.keys())[-1]].cpu().detach().numpy()
            self.cim = pd.DataFrame(cim, columns=list(self.model.map.iloc[:, -1].unique()))
            f, ax = plt.subplots(figsize=figsize)
            ax = sns.heatmap(self.cim, linewidths=.5, cmap="YlGnBu", cbar_kws={'label': "Variable importance"})
            ax.set(xlabel='top-level layer', ylabel='Class')

    def plot_subnetwork(self, names_df, pathway, figheight=500, figwidth=700):
        layerlist = list(self.model.map.columns)[1:]
        top_layer = layerlist[-1]
        cat_cols = layerlist.copy()
        layerlist.reverse()
        df = pd.DataFrame(self.model.map[self.model.map[top_layer] == pathway]).groupby(layerlist).size()
        df = df.reset_index()
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

