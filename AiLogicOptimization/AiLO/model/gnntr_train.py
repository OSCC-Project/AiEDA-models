
from sklearn.manifold import TSNE
# from tsnecuda import TSNE # Use this package if the previous one doesn't work
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statistics

import os
import torch
import gc
import torch.nn as nn
from .gnn_models import GNN, NodeEncoder
from .gnn_crosstr import TR, SynthesisTransformerEncoder
import torch.nn.functional as F

import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from vit_pytorch.recorder import Recorder
from dataset.utils import *

# criterion = nn.HuberLoss(delta=1.0)
criterion = nn.MSELoss()
# criterion = nn.L1Loss()

def optimizer_to(optim, device):
    # move optimizer to device
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def load_ckp(checkpoint_fpath, model, optimizer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device("cpu"))
    # device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_fpath, map_location = device)
    model.load_state_dict(checkpoint['state_dict'])
    print('model loaded from checkpoint file', checkpoint_fpath)
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.to(device)
      
    optimizer_to(optimizer, device)

    return model, optimizer, checkpoint['epoch']
    
def mean_squared_log_error(y_true, y_pred):
    epsilon = 1e-10
    return np.mean((np.log(y_true + epsilon) - np.log(y_pred + epsilon)) ** 2)

def mean_absolute_percentage_error(actuals, predictions):
    if len(actuals) != len(predictions):
        raise ValueError("实际值和预测值列表的长度必须相同。")
    
    mape = 100.0 * sum(abs((actuals[i] - predictions[i]) / max(actuals[i], 1e-8)) for i in range(len(actuals))) / len(actuals)
    return mape

def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    # print('y_pred:',y_pred)
    # print('y_true:',y_true)
    mae = mean_absolute_error(y_true, y_pred)
    # msle = mean_squared_log_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

def plot_tsne(nodes, labels, t):
    
    #Plot t-SNE visualizations
    
    labels_tox21 = ['SR-HSE', 'SR-MMP', 'SR-p53']
    #labels_sider = ['R.U.D.', 'P.P.P.C.', 'E.L.D.', 'C.D.', 'N.S.D.', 'I.S.P.C.']
     
    t+=1
    emb_tsne = np.asarray(nodes)
    y_tsne = np.asarray(labels).flatten()
    slipper_colour = pd.DataFrame({'colour': ['Blue', 'Orange'],
                       'label': [0, 1]})
    
    c_dict = {'Positive': '#ff7f0e','Negative': '#1f77b4' }
     
    z = TSNE(n_components=2, init='random').fit_transform(emb_tsne)
    label_vals = {0: 'Negative', 1: 'Positive'}
    tsne_result_df = pd.DataFrame({'tsne_dim_1': z[:,0], 'tsne_dim_2': z[:,1], 'label': y_tsne})
    tsne_result_df['label'] = tsne_result_df['label'].map(label_vals)
    fig, ax = plt.subplots(1)
    sns.set_style("ticks",{'axes.grid' : True})
    g1 = sns.scatterplot(x='tsne_dim_1', y='tsne_dim_2', hue='label', data=tsne_result_df, ax=ax,s=10, palette = c_dict, hue_order=('Negative', 'Positive'))
    lim = (z.min()-5, z.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal') 
    
    g1.legend(title=labels_tox21[t-1], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)    
    g1.set(xticklabels=[])
    g1.set(yticklabels=[])
    g1.set(xlabel=None)
    g1.set(ylabel=None)
    g1.tick_params(bottom=False) 
    g1.tick_params(left=False)
    plt.savefig('plots/'+labels_tox21[t-1])
    plt.show()
    plt.close(fig)
    
    return t
    
class CrossLO(nn.Module):
    def __init__(self, node_emb_size=2, gnnemb_size=128, synemb_size=128, batch_size=1,sm_size = 64, lg_size = 256):
        super(CrossLO,self).__init__()
        
        self.node_emb_size = node_emb_size
        self.gnnemb_size = gnnemb_size
        self.synemb_size = synemb_size
        self.batch_size = batch_size

        
        self.nodeencoder = NodeEncoder(self.node_emb_size)
        self.gnn = GNN(self.nodeencoder, self.node_emb_size*5, self.gnnemb_size)
        # self.SynFolwEncoder = SynthFlowEncoder(14, 128)
        self.synth_transformer = SynthesisTransformerEncoder(
            num_synthesis_actions= 14,
            embedding_dim=self.synemb_size,
            num_layers=4,  # 示例值，根据需要调整
            num_heads=4,  # 示例值，根据需要调整
            ff_dim=512,  # 示例值，根据需要调整
            dropout=0.4  # 示例值，根据需要调整
        )
        # self.synthesis_embedding_layer = nn.Embedding(16, 128)
        self.transformer = TR(self.gnn, self.synth_transformer, self.batch_size, self.gnnemb_size, self.synemb_size, 1, sm_size, lg_size)
        # self.optimizer = torch.optim.Adam(self.transformer.parameters(), lr=self.lr_update)
        # self.scheduler = ReduceLROnPlateau(self.optimizer, "min", verbose=True)
    
    def forward(self, batch):
        return self.transformer(batch)
        
            
def train(model, data, device, optimizer):
    torch.cuda.empty_cache()
    model.train()
    losses = []
    for _, batch in enumerate(tqdm(data, desc="Train:")):
        batch = batch.to(device)
        label = batch.target.view(-1)
        optimizer.zero_grad()
        pred = model.transformer(batch)
        pred = pred.view(-1)
        # print(f'pred:{pred}, label:{label}')
        loss = criterion(pred, label)
        # print(f'loss:{loss}')
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
    return np.mean(losses)

def test(model, data, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for _, batch in enumerate(tqdm(data, desc="Test:")):
            batch = batch.to(device)
            label = batch.target.view(-1)
            pred = model.transformer(batch)
            pred = pred.view(-1)
            loss = criterion(pred, label)
            losses.append(loss.detach().cpu().numpy())
    return np.mean(losses)

def dse(model,device,batch):
    model.eval()
    with torch.no_grad():
        batch = batch.to(device)
        pred = model.transformer(batch)
        pred = pred.reshape(-1)
    return pred, batch.opt_seq